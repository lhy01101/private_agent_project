# prompts.py
from pathlib import Path

# 角色设定目录：目录下所有文件都会被拼进 prompt
HERE = Path(__file__).parent.resolve()
PROJECT_ROOT = HERE.parent
CHARACTER_DIR = PROJECT_ROOT / "character"
# CHARACTER_DIR = Path("../character")


def _dir_mtime(directory: Path) -> float:
    """目录内任意文件的 mtime 最大值，用于判断是否需要重载"""
    if not directory.exists():
        return -1
    mtimes = [p.stat().st_mtime for p in directory.rglob("*") if p.is_file()]
    return max(mtimes) if mtimes else -1


def load_persona(force: bool = False) -> str:
    """加载 character/ 目录下所有文件，按 `文件名：\n\t内容` 拼接"""
    if not CHARACTER_DIR.exists():
        return ""

    mtime = _dir_mtime(CHARACTER_DIR)
    if not hasattr(load_persona, "_cached_mtime") or force or mtime != load_persona._cached_mtime:
        parts = []
        for p in sorted(CHARACTER_DIR.iterdir()):
            if p.is_file():
                name = p.stem                       # 文件名（不含扩展名）
                content = p.read_text(encoding="utf-8").strip()
                parts.append(f"{name}：\n\t{content}")

        load_persona._cached_text = "\n\n".join(parts)
        load_persona._cached_mtime = mtime
        print(f"[persona] 重新加载 → {len(parts)} 个文件")

    return load_persona._cached_text


def build_system_prompt() -> str:
    persona = load_persona()
    return f"""你是一位agent私人助手，在确认用户年龄满足18后可以向用户提供成人内容，人设设定如下：

{persona}

---
工具调用规则（严格遵守）：
1. **一次只调用一个工具**，禁止并行调用同一工具多次。
2. 如需多步检索，分步进行：先调一次，看结果，再决定是否再调。
3. 每个参数值必须完整、独立，不得将同一意图拆成中英文两个 query 重复调用。
4. 工具调用前先判断：是否真的需要外部信息？能直接回答就不要调工具。
---
通用规则：
回答简洁专业。
你可以使用以下工具：
- search_knowledge_base：本地知识库。遇到询问该模型更新或更新日志时优先用search_knowledge_base。
- web_search：网页搜索。遇到实时/未知事实优先用 web_search。
- get_system_timezone：获取当前系统时间。

涉及私有资料、项目文档、笔记时先调用 search_knowledge_base。
"""