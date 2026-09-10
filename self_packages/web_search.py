import logging

# 1. 关掉 ddgs 的所有 INFO/WARNING
logging.getLogger("ddgs").setLevel(logging.ERROR)
# 2. 关掉 httpx 的 INFO（Ollama 调用会打很多）
logging.getLogger("httpx").setLevel(logging.WARNING)
# 3. 关掉 primp（ddgs 内部用的 HTTP 库）
logging.getLogger("primp").setLevel(logging.ERROR)
# 4. （可选）防止第三方库重新配置 root
logging.basicConfig(level=logging.ERROR)


from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / ".env"   # 和当前 py 文件同目录
load_dotenv(dotenv_path=env_path)


from self_packages.web_search_provider import _bocha_search,_ddg_search,_tavily_search
import re
import os

_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff66-\uff9f]")

def _is_cjk(q: str) -> bool:
    return bool(_CJK.search(q))

def routed_web_search(query: str) -> str:
    bocha_ok = bool(os.getenv("BOCHA_API_KEY"))
    tavily_ok = bool(os.getenv("TAVILY_API_KEY"))
    ddg_ok = os.getenv("DEV_DDG_FALLBACK") == "1"

    # 候选顺序：中文优先博查；英文博查也先试，失败转 Tavily
    plan = []
    if bocha_ok:
        plan.append(("bocha", _bocha_search))
    if tavily_ok:
        plan.append(("tavily", _tavily_search))
    if ddg_ok:
        plan.append(("ddg", _ddg_search))

    if not plan:
        return "（未配置任何搜索供应商：BOCHA_API_KEY / TAVILY_API_KEY）"

    last_err = None
    for name, fn in plan:
        try:
            out = fn(query)
            if out and "（搜索无结果）" not in out:
                return out
            last_err = f"{name}: empty"
        except Exception as e:  # 429 / timeout / JSON 错位都吞掉
            last_err = f"{name}: {e}"
            continue
    return f"搜索失败：{last_err}"



from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class SearchArgs(BaseModel):
    query: str = Field(description="搜索关键词")

web_tool = StructuredTool.from_function(
    func=routed_web_search,
    name="web_search",
    description="联网搜索最新网页信息，用于模型知识截止后的新事实、新闻、版本号、价格等。中文优先走博查，英文走 Tavily，均未配置时不可用。",
    args_schema=SearchArgs,
)