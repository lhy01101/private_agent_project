from self_packages.tools import tools
from langchain.agents import create_agent
from self_packages.chat_box import chat_box as cb
from self_packages.dynamic_select import basic_model, DynamicModelMiddleware
from self_packages.tool_routing_middleware import ToolRoutingMiddleware
from self_packages.response_format import ResponseFormat
from langgraph.checkpoint.memory import InMemorySaver
from self_packages.prompts import build_system_prompt

import os


agent = create_agent(
    model=basic_model,  # 默认模型
    tools=tools,
    system_prompt=build_system_prompt(),
    middleware=[DynamicModelMiddleware(), ToolRoutingMiddleware()],
    checkpointer=InMemorySaver(),
    # response_format=ResponseFormat,   # deepseek-v4-flash不适配
)

# prev_len_ref = [0]
# cb(agent, prev_len_ref=prev_len_ref)
