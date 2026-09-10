# build_index.py
from pathlib import Path
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

HERE = Path(__file__).parent.resolve()
PROJECT_ROOT = HERE.parent
DOCS_DIR = PROJECT_ROOT / "docs"
PERSIST_DIR = "../chroma_rag"
EMBED_MODEL = "nomic-embed-text"    # 1.注意与tools.embeddings保持一致  2.Chroma 里存的 embedding 维度和当前模型输出维度一致

loaders = []
for ext in [".md", ".txt"]:
    for p in DOCS_DIR.rglob(f"*{ext}"):
        loaders.append(TextLoader(str(p), encoding="utf-8"))
for p in DOCS_DIR.rglob("*.pdf"):
    loaders.append(PyPDFLoader(str(p)))

if not loaders:
    raise ValueError("docs/ 里没有任何文件！")

documents = []
for loader in loaders:
    try:
        docs = loader.load()
        documents.extend(docs)
        print(f"✅ {getattr(loader,'file_path',None) or getattr(loader,'path','')}")
    except Exception as e:
        print(f"❌ {e}")

print(f"原始文档: {len(documents)}")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=150,
    separators=["\n\n","\n","。","！","？"," ",""],
)
splits = splitter.split_documents(documents)
print(f"切片数: {len(splits)}")

embeddings = OllamaEmbeddings(model=EMBED_MODEL)
vector_store = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory=PERSIST_DIR,
)
print(f"✅ 索引完成 → {PERSIST_DIR}/")
test_embedding = embeddings.embed_query("测试")
print(len(test_embedding))   # nomic-embed-text 是 768 维