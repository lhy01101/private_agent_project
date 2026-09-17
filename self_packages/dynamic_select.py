# agent模型中间件，把模型剖开，可以用于动态选择模型/工具，后续可以在中间件上继续进行开发

# 现已弃用，由update_index.py完全承担上位功能

from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse, AgentMiddleware
from langchain.chat_models import init_chat_model
import os

from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent / ".env"   # 和当前 py 文件同目录
load_dotenv(dotenv_path=env_path)


# basic_model = init_chat_model("ollama:qwen3:8b")
# Reasoning_model = init_chat_model("ollama:deepseek-r1:7b")
basic_model = init_chat_model(
    "deepseek-v4-flash",
    model_provider="deepseek",   # 需 pip install langchain-deepseek
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)
advanced_model = init_chat_model(
    "deepseek-v4-flash",
    model_provider="deepseek",   # 需 pip install langchain-deepseek
    api_key=os.getenv("DEEPSEEK_API_KEY"),
)



class DynamicModelMiddleware(AgentMiddleware):
    name = "dynamic_model"
    trace_policy = None
    def wrap_model_call(self, request, call):
        msgs = request.state.get("messages", [])
        model = advanced_model if len(msgs) > 10 else basic_model
        request = request.override(model=model)
        return call(request)

    async def awrap_model_call(self, request, call):   # 顺序对齐规则：request 永远在前面，callable 永远在后面，和同步版完全对称。
        msgs = request.state.get("messages", [])
        if len(msgs) > 10:
            request = request.override(model=advanced_model)
        else:
            request = request.override(model=basic_model)
        return await call(request)
"""
# wrap_model_call会报错
def dynamic_model_selection(request: ModelRequest, handler) -> ModelResponse:
    # 根据对话复杂性选择模型。
    message_count = len(request.state["messages"])
    if message_count > 10:
        # 对较长的对话使用高级模型
        model = advanced_model
    else:
        model = basic_model
    request.model = model
    return handler(request)
"""
