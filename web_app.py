"""
方塘 AI Chat - FastAPI Web Interface
Run: python web_app.py
"""

import uuid
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from wonder_agent import agent

app = FastAPI(title="方塘 AI Chat")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

conversations: dict[str, dict] = {}


class ChatRequest(BaseModel):
    message: str


class TitleRequest(BaseModel):
    title: str


@app.get("/api/conversations")
async def list_conversations():
    return sorted(
        conversations.values(),
        key=lambda c: c["created_at"],
        reverse=True,
    )


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str):
    if conv_id not in conversations:
        raise HTTPException(404, "Conversation not found")
    return conversations[conv_id]


@app.post("/api/conversations")
async def create_conversation():
    conv_id = str(uuid.uuid4())
    conversations[conv_id] = {
        "id": conv_id,
        "title": "新对话",
        "created_at": datetime.now().isoformat(),
    }
    return conversations[conv_id]


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str):
    conversations.pop(conv_id, None)
    return {"ok": True}


@app.post("/api/conversations/{conv_id}/title")
async def update_title(conv_id: str, req: TitleRequest):
    if conv_id not in conversations:
        raise HTTPException(404)
    conversations[conv_id]["title"] = req.title
    return {"ok": True}


@app.post("/api/chat/{conv_id}")
async def chat(conv_id: str, req: ChatRequest):
    if conv_id not in conversations:
        raise HTTPException(404, "Conversation not found")

    message = req.message.strip()
    if not message:
        raise HTTPException(400, "Empty message")

    async def event_stream():
        import json as _json

        def _sse(event: str, data: str) -> str:
            return f"event: {event}\ndata: {data}\n\n"

        yield _sse("user_message", _json.dumps({"content": message}, ensure_ascii=False))

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
                                _json.dumps({"content": text}, ensure_ascii=False),
                            )

                elif kind == "on_tool_start":
                    tool_name = event.get("name", "unknown")
                    display = f"正在调用: {tool_name}"
                    yield _sse(
                        "tool",
                        _json.dumps({"content": display}, ensure_ascii=False),
                    )

                elif kind == "on_tool_end":
                    yield _sse("tool_end", "{}")

        except Exception as e:
            yield _sse(
                "error",
                _json.dumps({"content": f"Error: {e}"}, ensure_ascii=False),
            )
            return

        if not has_content:
            yield _sse(
                "text",
                _json.dumps(
                    {"content": "（未生成文本输出）"}, ensure_ascii=False
                ),
            )

        yield _sse("done", "{}")

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
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
