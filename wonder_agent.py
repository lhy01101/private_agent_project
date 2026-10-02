from self_packages.tools import tools
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from self_packages.chat_box import chat_box as cb
from self_packages.dynamic_select import basic_model, DynamicModelMiddleware
from self_packages.tool_routing_middleware import ToolRoutingMiddleware
from self_packages.response_format import ResponseFormat
from self_packages.prompts import build_system_prompt

import os

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")

try:
    from langgraph_checkpoint_plainredis import AsyncRedisSaver
    checkpointer = AsyncRedisSaver(url=REDIS_URL)
    print(f"[checkpointer] Redis (plain): {REDIS_URL}")
except Exception as e:
    from langgraph.checkpoint.memory import InMemorySaver
    checkpointer = InMemorySaver()
    print(f"[checkpointer] Redis unavailable ({e}), using InMemorySaver")


agent = create_agent(
    model=basic_model,
    tools=tools,
    system_prompt=build_system_prompt(),
    middleware=[
        DynamicModelMiddleware(),
        ToolRoutingMiddleware(),
        SummarizationMiddleware(
            model=basic_model,
            trigger=("tokens", 8000),
            keep=("messages", 10),
        ),
    ],
    checkpointer=checkpointer,
    # response_format=ResponseFormat,   # deepseek-v4-flash不适配
)

# 单次invoke
# prev_len_ref = [0]
# cb(agent, prev_len_ref=prev_len_ref)
