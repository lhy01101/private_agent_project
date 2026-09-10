from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

import logging
# logging.basicConfig(level=logging.INFO)

#工具
from dataclasses import dataclass
from langchain.tools import tool, ToolRuntime

@tool # LangChain 的 @tool 装饰器 会添加元数据，并通过 ToolRuntime 参数启用运行时注入。
def get_weather_for_location(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"

@dataclass
class Context:
    """自定义运行时上下文模式。"""
    user_id: str

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """根据用户 ID 获取用户信息。"""
    user_id = runtime.context.user_id
    return "Florida" if user_id == "1" else "SF"

@tool
def get_weather(city: str) -> str:
    # 所有str都为类型提示，非强约束
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"

@tool
def cal(i: float, j:float) -> float:
    """除法计算工具"""
    if(j==0):
        print("输入了错误的除数！例如0")
        raise ValueError
    return i/j

from langchain.chat_models import init_chat_model
from langchain.agents.middleware import wrap_model_call, ModelRequest, ModelResponse

model = init_chat_model(
    "ollama:qwen3:1.7b",
    temperature=0.5,
    # temperature=0  # 要什么参数在这加
)
tools = [get_weather, get_weather_for_location, get_user_location]
SYSTEM_PROMPT = """你是一位擅长用双关语表达的专家天气预报员。

你可以使用两个工具：

- get_weather_for_location：用于获取特定地点的天气
- get_user_location：用于获取用户的位置

如果用户询问天气，请确保你知道具体位置。如果从问题中可以判断他们指的是自己所在的位置，请使用 get_user_location 工具来查找他们的位置。"""
from langgraph.checkpoint.memory import InMemorySaver
checkpointer = InMemorySaver()

from dataclasses import dataclass
# 这里使用 dataclass，但也支持 Pydantic 模型。
@dataclass
class ResponseFormat:
    """代理的响应模式。"""
    # 带双关语的回应（始终必需）
    punny_response: str
    # 天气的任何有趣信息（如果有）
    weather_conditions: str | None = None

agent = create_agent(
    model = model,
    tools=tools,
    system_prompt=SYSTEM_PROMPT,
    checkpointer=checkpointer,
    response_format=ResponseFormat,
    debug=True,
)

# 运行代理
config = {"configurable": {"thread_id": "user-001"}}
# user_id = 490829b3-7ab8-49a3-9e6f-cabae8bcdda6
prev_len = 0
while True:
    lines = []
    while True:
        line = input(">>> ")
        if line == "":
            break
        lines.append(line)

    user_input = "\n".join(lines)
    if user_input.lower() in {"exit", "quit"}:
        break

    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
        context=Context(user_id="1")
    )
    new_messages = result["messages"][prev_len:]
    prev_len = len(result["messages"])
    for msg in new_messages:
        if isinstance(msg, AIMessage):
            if msg.tool_calls:
                print("\n🛠️ Tool Calls：")
                for tc in msg.tool_calls:
                    print(f"  • {tc['name']}({tc['args']})")
            print("\n🤖 AI：")
            msg.pretty_print()

        elif isinstance(msg, ToolMessage):
            print(f"\n🔧 Tool Result({msg.name})：")
            print(msg.content)
        else:
            print("\n👤 User：")
            print(msg.content)

