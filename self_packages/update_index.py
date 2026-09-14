# build_index.py
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json
import time

# ==================== 配置 ====================
HERE = Path(__file__).parent.resolve()
PROJECT_ROOT = HERE.parent
DOCS_DIR = PROJECT_ROOT / "docs"

PERSIST_DIR = PROJECT_ROOT/ "chroma_rag"
# 增量更新必须保证以下几项与原rag一致
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# 记录文件状态的元数据文件（存每个文件的相对路径 + mtime）
STATE_FILE = Path(PERSIST_DIR) / "file_state.json"

# ==================== 工具函数 ====================
def load_state() -> dict:
    """读取上次构建时的文件状态"""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_state(state: dict):
    """保存当前文件状态"""
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

def get_file_state(path: Path) -> str:
    """用文件的 mtime + size 作为指纹"""
    stat = path.stat()
    return f"{stat.st_mtime_ns}_{stat.st_size}"

def make_loaders(paths: list[Path]):
    """根据扩展名创建对应 loader"""
    loaders = []
    for p in paths:
        ext = p.suffix.lower()
        if ext in [".md", ".txt"]:
            loaders.append(TextLoader(str(p), encoding="utf-8"))
        elif ext == ".pdf":
            loaders.append(PyPDFLoader(str(p)))
    return loaders

def load_documents(loaders: list) -> list:
    """加载文档"""
    documents = []
    for loader in loaders:
        try:
            docs = loader.load()
            documents.extend(docs)
            src = getattr(loader, "file_path", None) or getattr(loader, "path", "unknown")
            print(f"  ✅ {src}")
        except Exception as e:
            print(f"  ❌ {e}")
    return documents

# ==================== 主流程 ====================
def main():
    DOCS_DIR = Path("../docs").resolve()
    PERSIST_DIR = Path("../chroma_rag").resolve()

    if not DOCS_DIR.exists():
        raise ValueError(f"文档目录不存在: {DOCS_DIR}")

    # 1. 扫描当前所有文件
    all_files: list[Path] = []
    for ext in [".md", ".txt", ".pdf"]:
        all_files.extend(DOCS_DIR.rglob(f"*{ext}"))

    if not all_files:
        raise ValueError(f"{DOCS_DIR} 里没有任何文件！")

    # 2. 读取上次的状态，区分新增/修改/未变
    prev_state = load_state()
    curr_state: dict[str, str] = {}

    new_files: list[Path] = []      # 新增或修改过的文件
    unchanged_count = 0

    for p in all_files:
        rel = str(p.relative_to(DOCS_DIR))   # 相对路径作为 key
        fingerprint = get_file_state(p)
        curr_state[rel] = fingerprint
        if rel not in prev_state or prev_state[rel] != fingerprint:
            new_files.append(p)
        else:
            unchanged_count += 1

    print(f"📂 扫描到 {len(all_files)} 个文件")
    print(f"   - 未变动: {unchanged_count}")
    print(f"   - 新增/修改: {len(new_files)}")

    if not new_files:
        print("✅ 没有新文件，无需更新索引。")
        return

    # 3. 加载新文件
    print(f"\n📄 加载新文件:")
    loaders = make_loaders(new_files)
    documents = load_documents(loaders)
    print(f"   原始文档: {len(documents)} 篇")

    if not documents:
        print("⚠️ 新文件全部加载失败，退出。")
        return

    # 4. 切片
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""],
    )
    splits = splitter.split_documents(documents)
    print(f"   切片数: {len(splits)}")

    # 5. 写入向量库（增量 or 新建）
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    if Path(PERSIST_DIR).exists() and (Path(PERSIST_DIR) / "chroma.sqlite3").exists():
        # ---- 增量更新：加载已有库，追加新文档 ----
        print(f"\n🔄 增量更新 → {PERSIST_DIR}")
        vector_store = Chroma(
            persist_directory=str(PERSIST_DIR),
            embedding_function=embeddings,
        )
        vector_store.add_documents(splits)
        print(f"   ✅ 追加 {len(splits)} 个切片")
    else:
        # ---- 首次构建 ----
        print(f"\n🆕 首次构建 → {PERSIST_DIR}")
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=str(PERSIST_DIR),
        )
        print(f"   ✅ 创建 {len(splits)} 个切片")

    # 6. 持久化（from_documents 自动 persist；add_documents 需手动）
    if hasattr(vector_store, "persist"):
        vector_store.persist()

    # 7. 保存文件状态
    save_state(curr_state)
    print(f"\n✅ 索引更新完成 → {PERSIST_DIR}/")
    print(f"   📝 文件状态已保存: {STATE_FILE}")

    # 8. 维度验证
    test_embedding = embeddings.embed_query("测试")
    print(f"   📐 embedding 维度: {len(test_embedding)}")

if __name__ == "__main__":
    main()