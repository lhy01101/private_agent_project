#用于模型输出，使用result保存完整对话内容（包含记忆），使用prev_len得到当前对话内容
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from self_packages.permission import build_context

import logging
logging.basicConfig(level=logging.INFO)


# 用法：prev_len_ref = [0]
def chat_box(agent, prev_len_ref, user_id: str = "1"):
    """CLI 对话入口。user_id 决定权限等级（查 permission 表）与独立对话记忆（thread）。"""
    prev_len = prev_len_ref[0]
    config = {"configurable": {"thread_id": f"user-{user_id}"}}
    """
    引用传递:Python 的整数是 不可变对象，函数内赋值不会影响外部
    但 list / dict 是可变的，可以“模拟引用传递”
    """
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
            context=build_context(user_id),
        )
        new_messages = result["messages"][prev_len:]
        prev_len_ref[0] = len(result["messages"])
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