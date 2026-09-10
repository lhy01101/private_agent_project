from wonder_agent import agent
import gradio as gr
import os

"""
网页版 Agent（Gradio）
- 模型：本地 Ollama (qwen3:8b)
- 能力：多轮对话 + RAG（检索本地 chroma_rag 知识库）+ 联网搜索
- 特性：token 级流式输出、新对话按钮、他人可访问（server_name=0.0.0.0）
启动：
    uv run python web_chat_box.py
然后在浏览器打开 http://<本机IP>:7860  （本机可访问 http://127.0.0.1:7860）
"""
# MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3:8b")
MODEL_NAME = os.getenv("deepseek", "deepseek-v4-flash")


from pathlib import Path
BASE_DIR = Path(__file__).parent

custom_css = """
.avatar-image {
    width: 64px !important;
    height: 64px !important;
    min-width: 64px !important;
}

.avatar-container {
    width: 64px !important;
    height: 64px !important;
    min-width: 64px !important;
}
"""

# ==================== 会话 thread_id 管理 ====================
_thread_counter = [0]

def new_thread_id() -> str:
    _thread_counter[0] += 1
    return f"web-user-{_thread_counter[0]}"

# ==================== 核心：流式响应 ====================
async def respond(message: str, history: list, thread_id: str):
    """
    核心逻辑：
    - history 是 list[dict]，格式 [{"role": "user", "content": "..."}, ...]
    - 手动 append assistant 消息，流式更新 history[-1]["content"]
    - 每次 yield history（list[dict]），Gradio 6.0 Chatbot 直接认
    """
    config = {"configurable": {"thread_id": thread_id}}

    # 把用户消息加入 history（UI 立刻显示）
    history.append({"role": "user", "content": message})
    # 先放一个空的 assistant 气泡
    history.append({"role": "assistant", "content": ""})

    partial = ""

    async for event in agent.astream_events(
        {"messages": [{"role": "user", "content": message}]},
        config=config,
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"].get("chunk")
            if chunk is None:
                continue
            content = getattr(chunk, "content", "")
            if isinstance(content, list):
                content = "".join(
                    c.get("text", "") for c in content if isinstance(c, dict)
                )
            if isinstance(content, str) and content:
                partial += content
                history[-1]["content"] = partial
                yield history, thread_id

        elif event["event"] == "on_tool_start":
            tool_name = event.get("name", "unknown")
            history[-1]["content"] = partial + f"\n\n🔧 *正在调用 {tool_name}...*"
            yield history, thread_id

        elif event["event"] == "on_tool_end":
            history[-1]["content"] = partial
            yield history, thread_id

    if not partial:
        history[-1]["content"] = "（未生成文本输出）"
    yield history, thread_id


# ==================== Gradio 界面 ====================
def make_app() -> gr.Blocks:
    with gr.Blocks(title="Agent 助手") as demo:
        gr.Markdown(f"# Agent 网页助手\n模型：`{MODEL_NAME}`　|　支持多轮对话 / 流式输出 / 新对话")

        thread_state = gr.State(value=new_thread_id())

        chatbot = gr.Chatbot(
            label="对话",
            height=520,
            avatar_images=(
                None,
                BASE_DIR / "pictures/bot2_compress.jpg",
            ),
        )

        msg = gr.Textbox(
            lines=1,
            placeholder="输入消息后回车发送…（输入 /new 开启新会话）",
            container=False,
            scale=7,
        )

        # 新对话按钮
        new_btn = gr.Button("🆕 新对话", variant="secondary")

        def clear_and_new():
            return [], new_thread_id()

        # 发送逻辑
        def handle_input(user_msg, history, tid):
            if user_msg.strip() == "/new":
                return "", [], new_thread_id()
            # 不清空 history，respond 里会自己 append
            return user_msg, history, tid

        msg.submit(
            handle_input, [msg, chatbot, thread_state], [msg, chatbot, thread_state]
        ).then(
            respond, [msg, chatbot, thread_state], [chatbot, thread_state]
        )

        new_btn.click(clear_and_new, None, [chatbot, thread_state])

        # 示例问题
        gr.Examples(
            examples=[
                "你好，介绍一下你自己",
                "帮我查看本地模型的更新日志",
            ],
            inputs=msg,
        )

    return demo

# ==================== 启动 ====================
if __name__ == "__main__":
    print("* Running on URL:  http://127.0.0.1:7860")
    PORT = int(os.getenv("PORT", "7860"))
    make_app().queue().launch(
        server_name="0.0.0.0",
        server_port=PORT,
        theme=gr.themes.Soft(),
        show_error=True,
        css=custom_css,
        # share=True,
    )