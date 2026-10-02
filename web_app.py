"""
方塘 AI Chat - FastAPI Web Interface (Redis Persistence)
Run: python web_app.py
Env: REDIS_URL=redis://localhost:6379 (default)
"""

import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wonder_agent import agent, checkpointer

REDIS_URL = "redis://localhost:6379"

_redis = None
_conversations: dict[str, dict] = {}


@asynccontextmanager
async def lifespan(app):
    global _redis
    try:
        import redis.asyncio as aioredis
        _redis = aioredis.from_url(REDIS_URL, decode_responses=True)
        await _redis.ping()
        print(f"[metadata] Redis: {REDIS_URL}")
    except Exception as e:
        _redis = None
        print(f"[metadata] Redis unavailable ({e}), using in-memory")

    try:
        if hasattr(checkpointer, "setup"):
            checkpointer.setup()
    except Exception:
        pass

    yield

    if _redis:
        await _redis.aclose()


app = FastAPI(title="方塘 AI Chat", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str


class TitleRequest(BaseModel):
    title: str


# ── Conversation metadata (Redis or in-memory fallback) ──


@app.get("/api/conversations")
async def list_conversations():
    if _redis:
        ids = await _redis.zrevrange("ft:conversations", 0, -1)
        result = []
        for cid in ids:
            data = await _redis.hgetall(f"ft:conv:{cid}")
            if data:
                result.append(data)
        return result
    return sorted(_conversations.values(), key=lambda c: c["created_at"], reverse=True)


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    if _redis:
        data = await _redis.hgetall(f"ft:conv:{conv_id}")
        if not data:
            raise HTTPException(404, "Conversation not found")
        return data
    if conv_id not in _conversations:
        raise HTTPException(404, "Conversation not found")
    return _conversations[conv_id]


@app.post("/api/conversations")
async def create_conversation():
    conv_id = str(uuid.uuid4())
    conv = {"id": conv_id, "title": "新对话", "created_at": datetime.now().isoformat()}
    if _redis:
        await _redis.hset(f"ft:conv:{conv_id}", mapping=conv)
        await _redis.zadd("ft:conversations", {conv_id: datetime.now().timestamp()})
    else:
        _conversations[conv_id] = conv
    return conv


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    if _redis:
        await _redis.delete(f"ft:conv:{conv_id}")
        await _redis.zrem("ft:conversations", conv_id)
    else:
        _conversations.pop(conv_id, None)
    return {"ok": True}


@app.post("/api/conversations/{conv_id}/title")
async def update_title(conv_id: str, req: TitleRequest):
    if _redis:
        if not await _redis.exists(f"ft:conv:{conv_id}"):
            raise HTTPException(404)
        await _redis.hset(f"ft:conv:{conv_id}", "title", req.title)
    else:
        if conv_id not in _conversations:
            raise HTTPException(404)
        _conversations[conv_id]["title"] = req.title
    return {"ok": True}


@app.get("/api/history/{conv_id}")
async def get_history(conv_id: str):
    try:
        config = {"configurable": {"thread_id": conv_id}}
        state = await agent.aget_state(config)
        print(f"[history] state for {conv_id}: values_keys={list(state.values.keys()) if state.values else None}")
        messages = state.values.get("messages", []) if state.values else []
        print(f"[history] {conv_id}: {len(messages)} messages found")
        result = []
        for msg in messages:
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            msg_type = getattr(msg, "type", "")
            if msg_type == "human":
                result.append({"role": "user", "content": content})
            elif msg_type == "ai":
                if content:
                    result.append({"role": "assistant", "content": content})
            elif msg_type == "tool":
                result.append({
                    "role": "assistant",
                    "content": f"> 工具调用 ({getattr(msg, 'name', '')})\n\n{content}",
                })
        print(f"[history] {conv_id}: returning {len(result)} formatted messages")
        return {"messages": result}
    except Exception as e:
        print(f"[history] ERROR for {conv_id}: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return {"messages": []}


# ── Chat (agent history persisted by LangGraph checkpointer) ──


@app.post("/api/chat/{conv_id}")
async def chat(conv_id: str, req: ChatRequest):
    exists = (
        await _redis.exists(f"ft:conv:{conv_id}")
        if _redis
        else conv_id in _conversations
    )
    if not exists:
        raise HTTPException(404, "Conversation not found")

    message = req.message.strip()
    if not message:
        raise HTTPException(400, "Empty message")

    async def event_stream():
        def _sse(event: str, data: str) -> str:
            return f"event: {event}\ndata: {data}\n\n"

        yield _sse("user_message", json.dumps({"content": message}, ensure_ascii=False))

        partial = ""
        has_content = False

        try:
            async for event in agent.astream_events(
                {"messages": [{"role": "user", "content": message}]},
                config={"configurable": {"thread_id": conv_id}},
                version="v2",
            ):
                kind = event.get("event")

                if kind == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk:
                        content = getattr(chunk, "content", "")
                        if isinstance(content, list):
                            text = "".join(
                                p.get("text", "")
                                for p in content
                                if isinstance(p, dict) and p.get("type") == "text"
                            )
                        elif isinstance(content, str):
                            text = content
                        else:
                            text = ""

                        if text:
                            partial += text
                            has_content = True
                            yield _sse(
                                "text",
                                json.dumps({"content": text}, ensure_ascii=False),
                            )

                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    yield _sse(
                        "tool",
                        json.dumps(
                            {"content": f"正在调用: {tool_name}"}, ensure_ascii=False
                        ),
                    )

                elif kind == "on_tool_end":
                    yield _sse("tool_end", "{}")

        except Exception as e:
            yield _sse(
                "error", json.dumps({"content": f"Error: {e}"}, ensure_ascii=False)
            )
            return

        if not has_content:
            yield _sse(
                "text",
                json.dumps({"content": "（未生成文本输出）"}, ensure_ascii=False),
            )

        yield _sse("done", "{}")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


if __name__ == "__main__":
    import uvicorn
    # http://localhost:8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
