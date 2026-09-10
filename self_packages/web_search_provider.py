import requests, os


from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / ".env"   # 和当前 py 文件同目录
load_dotenv(dotenv_path=env_path)


# 博查和 Tavily 字段不一样，用一个小函数压平：
def _fmt(items: list[dict]) -> str:
    if not items:
        return "（搜索无结果）"
    return "\n".join(
        f"{x.get('title','')} ({x.get('url','')}): {x.get('content','')}"
        for x in items
    )


def _bocha_search(query: str, count: int = 5) -> str:
    r = requests.post(
        "https://api.bochaai.com/v1/web-search",   # 用 bochaai.com 这个 host，和你 .env 文档一致
        headers={
            "Authorization": f"Bearer {os.environ['BOCHA_API_KEY']}",
            "Content-Type": "application/json",
        },
        json={"query": query, "freshness": "noLimit", "summary": True, "count": count},
        timeout=15,
    )
    # 1. HTTP 层错误：交给路由降级，不让 ToolNode 炸
    if r.status_code != 200:
        raise RuntimeError(f"bocha http {r.status_code}: {r.text[:200]}")
    j = r.json()
    # 2. 业务码错误（博查用 code==200 表示成功）
    if j.get("code") != 200 or not j.get("data"):
        raise RuntimeError(f"bocha biz code={j.get('code')} msg={j.get('msg')}")
    pages = j["data"].get("webPages", {}).get("value", [])
    if not pages:
        return "（搜索无结果）"
    # 3. 字段兜底 + 归一成统一格式
    items = []
    for p in pages:
        content = p.get("summary") or p.get("snippet") or ""
        if not content:
            continue
        items.append({
            "title": p.get("name", ""),
            "url": p.get("url", ""),
            "content": content,
        })
    return _fmt(items)

def _tavily_search(query: str, count: int = 5) -> str:
    from tavily import TavilyClient
    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    resp = client.search(query=query, max_results=count, search_depth="basic")
    items = [
        {"title": x["title"], "url": x["url"], "content": x["content"]}
        for x in resp.get("results", [])
    ]
    return _fmt(items)

def _ddg_search(query: str) -> str:
    from langchain_community.tools import DuckDuckGoSearchRun
    try:
        return DuckDuckGoSearchRun().invoke(query)
    except Exception as e:
        return f"搜索失败：{e}"