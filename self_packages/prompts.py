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
通用规则：
回答简洁专业。
需要外部信息时可使用可用工具。

项目结构（关键文件）：
- wonder_agent.py: agent 主入口，定义 checkpointer 和 agent 实例
- web_app.py: FastAPI Web 界面，/api/chat、/api/upload 等接口
- self_packages/tools.py: 工具注册
- self_packages/tool_routing_middleware.py: 工具路由中间件（裁剪逻辑）
- self_packages/tool_router.py: 语义路由器（embedding 匹配）
- self_packages/routes_archive.py: 路由配置（正例/负例/工具映射）
- self_packages/prompts.py: 系统提示词构建（本文件）
- self_packages/file_reader_tool.py: 文件读取 + 工作区扫描工具
- self_packages/file_upload.py: 文件上传处理
- self_packages/web_search.py: 联网搜索
- character/: 角色设定目录
- android/: Android 移动端应用
- static/: Web 前端静态文件（index.html, app.js, style.css）

当需要更详细的目录时，可调用 list_workspace 扫描子目录。
---
工具调用规则（严格遵守）：
1. **一次只调用一个工具**，禁止并行调用同一工具多次。
2. 如需多步检索，分步进行：先调一次，看结果，再决定是否再调。
3. 每个参数值必须完整、独立，不得将同一意图拆成中英文两个 query 重复调用。
4. 当用户询问项目中具体文件的实现细节、或你不确定某个模块的具体写法时，先用 read_local_file 读取文件确认，不要凭推测回答。
5. 如果不确定文件路径，先用 list_workspace 查看目录结构。
"""