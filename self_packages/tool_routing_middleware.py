"""
工具路由中间件（ToolRoutingMiddleware）
===========================================

【这个中间件是干什么的？】
--------------------------
在 Agent 真正调用大模型之前，拦截这次请求，根据用户最新的提问，
用 tool_router 算出"本次该暴露哪些工具"，然后把 agent 可用的工具列表
裁剪到只剩这些。模型看不到无关工具 → 减少幻觉、降低 token、提升准确率。

【核心机制：你必须理解这几点】
---------------------------------
1. 框架在"调工具"、"调模型"这两个时机，会回调对应的 wrap_* / awrap_* 方法。
   - wrap_tool_call / awrap_tool_call ：拦截「工具执行」
   - wrap_model_call / awrap_model_call：拦截「模型调用」  ← 我们在这里裁剪工具

2. 每个方法都会拿到：
   - request：本次请求。这是一个 ModelRequest 对象，核心字段：
       · request.messages：本次要发给模型的消息列表（list[BaseMessage]）
       · request.tools   ：当前可用工具列表
       · request.tool_choice：本次的工具选择策略（None 表示交给模型，即 auto）
       · request.state   ：agent 状态；request.runtime：运行时上下文
     ⚠️ 注意：是 request.messages，不是 request.input["messages"]！
   - call   ：一个"继续往下走"的函数。你不调用 call(request)，整条链路就断了。

3. 想改参数？用 request.override(字段=新值) 生成一个新 request，再传给 call()。
   ⚠️ 永远不要直接修改 request.tools，要用 override 返回新对象（不可变思想）。

4. 同步方法（wrap_*）给「同步 agent」用，异步方法（awrap_*）给「异步 agent」用。
   两者逻辑必须一致，否则同步/异步跑出不同结果，极难排查。

【关于 select_tools 的注意事项】
--------------------------------
select_tools() 内部会跑 embedder.embed_query()，这是一次模型推理（CPU/GPU 计算）。
- 在同步环境（wrap_model_call）里直接调用没问题。
- 在异步环境（awrap_model_call）里【不能】直接调用，否则会阻塞整个事件循环。
  → 解决：用 asyncio 的 run_in_executor 把它丢到线程池里去跑。

【为什么"直接回答"不能用 tools=[]】（DSML 的根因）
--------------------------------------------------
DeepSeek 的工具能力必须靠请求里显式带 tools 才启用。一旦 tools 被覆盖成空列表：
- langchain/agents/factory.py 里是 `if final_tools: bind_tools(...)`，
  空列表时连 bind_tools 都不会调用，模型这一轮彻底没有工具通道；
- 但 system prompt / 上下文仍在要求它调工具 → 它只能把调用写成正文，也就是 DSML 文本。
所以「直接回答」改为：工具声明照旧给全，只把 tool_choice 置为 "none"
（DeepSeek 语义：不调用任何工具，而是生成一条消息）。

→ 不变式：本中间件永远不会把 request.tools 覆盖为空列表。
"""

import asyncio
import logging
from typing import Optional

from langchain.agents.middleware import AgentMiddleware

from self_packages.tool_router import select_tools  # 返回 List[BaseTool]，即工具对象列表

logger = logging.getLogger(__name__)

# 「直接回答」分支使用的 tool_choice 取值。
# DeepSeek 语义：模型不会调用任何工具，而是生成一条消息。
NO_TOOL_TOOL_CHOICE = "none"


class ToolRoutingMiddleware(AgentMiddleware):
    name = "tool_routing"
    # trace_policy：是否需要追踪/埋点。None 表示不启用，保持默认即可。
    trace_policy = None

    # ------------------------------------------------------------------ #
    # 工具调用拦截（本中间件不需要动工具执行，直接透传即可）
    # ------------------------------------------------------------------ #
    def wrap_tool_call(self, request, call):
        """同步：拦截工具执行。我们只裁剪工具列表，不干预执行过程，直接放行。"""
        return call(request)

    async def awrap_tool_call(self, request, call):
        """异步：同上，直接放行。"""
        return await call(request)

    # ------------------------------------------------------------------ #
    # 模型调用拦截（核心：在这里裁剪工具列表）
    # ------------------------------------------------------------------ #
    def wrap_model_call(self, request, call):
        """
        同步版本：先算决策，再按分支放行。
        逻辑与异步版完全一致，区别只在「怎么调用 select_tools」。
        """
        # 1. 取出用户最新一条消息的文本
        content = self._extract_last_user_content(request)

        # 2. 算出本次决策（同步调用，同步环境里 OK）
        #    allowed_names 为 None 表示"路由失败/取不到内容"，此时不做任何干预
        allowed_names, tool_choice = self._compute_decision(content)

        # 3. 降级分支：路由不可用 → 不裁剪、不改 tool_choice，原样放行
        if allowed_names is None:
            return call(request)

        # 4. 直接回答分支：工具声明照旧给全，只把 tool_choice 置为 "none"
        #    ★ 关键：绝不用 tools=[] 表达"直接回答"。
        #      工具为空时 factory 不会调 bind_tools，模型失去工具通道 → 退化成正文 DSML。
        if not allowed_names:   # set()
            if not request.tools:
                logger.warning(
                    "[ToolRouting] 判定直接回答，但 request.tools 已为空，"
                    "模型将失去工具通道（DSML 风险）"
                )
            return call(request.override(tool_choice=tool_choice))  # request.override(tool_choice="none")

        # 5. 需要工具分支：裁剪到相关工具，tool_choice 交回模型自己决定（None → auto）
        filtered = [t for t in request.tools if t.name in allowed_names]

        #    裁剪结果意外为空（工具名未注册 / 同名冲突）时同样不下发空集，降级为"声明照旧 + 禁止调用"
        if not filtered:
            logger.warning(
                "[ToolRouting] 裁剪结果为空，降级为 tool_choice=%s：%s",
                NO_TOOL_TOOL_CHOICE,
                sorted(allowed_names),
            )
            return call(request.override(tool_choice=NO_TOOL_TOOL_CHOICE))

        return call(request.override(tools=filtered))

    async def awrap_model_call(self, request, call):
        """
        异步版本：先算决策，再按分支放行。
        与同步版唯一区别：select_tools 含模型推理，必须丢到线程池避免阻塞事件循环。
        """
        content = self._extract_last_user_content(request)

        # ★ 关键点：select_tools 是同步且有 CPU/GPU 计算的，不能在 async 里直接 await 它，
        #   而是把它提交到默认线程池，让出事件循环给其他协程。
        allowed_names, tool_choice = await asyncio.get_event_loop().run_in_executor(
            None,                          # 使用默认的 ThreadPoolExecutor
            self._compute_decision,        # 要执行的函数
            content,                       # 函数的参数
        )

        # 降级分支：路由不可用 → 不裁剪、不改 tool_choice，原样放行
        if allowed_names is None:
            return await call(request)

        # 直接回答分支：工具声明照旧给全，只把 tool_choice 置为 "none"
        if not allowed_names:
            if not request.tools:
                logger.warning(
                    "[ToolRouting] 判定直接回答，但 request.tools 已为空，"
                    "模型将失去工具通道（DSML 风险）"
                )
            return await call(request.override(tool_choice=tool_choice))

        # 需要工具分支：裁剪到相关工具，tool_choice 交回模型自己决定（None → auto）
        filtered = [t for t in request.tools if t.name in allowed_names]
        if not filtered:
            logger.warning(
                "[ToolRouting] 裁剪结果为空，降级为 tool_choice=%s：%s",
                NO_TOOL_TOOL_CHOICE,
                sorted(allowed_names),
            )
            return await call(request.override(tool_choice=NO_TOOL_TOOL_CHOICE))

        return await call(request.override(tools=filtered))

    # ------------------------------------------------------------------ #
    # 下面是抽取出来的公共逻辑（同步/异步共用，保证一致性）
    # ------------------------------------------------------------------ #
    def _compute_decision(self, content: Optional[str]) -> tuple[Optional[set], Optional[str]]:
        """
        根据用户输入文本，算出本次的『工具裁剪方案』。
        抽成独立方法，是为了让同步/异步两个入口都走同一份逻辑。

        参数:
            content: 用户最新一条消息的文本；取不到时为 None。
        返回 (allowed_names, tool_choice)：
            - (None, None)          ：路由失败 / 取不到有效内容 → 调用方原样放行，不做干预
            - (set(), "none")       ：判定「直接回答」→ 工具声明照旧给全，只禁止调用
            - ({工具名, ...}, None) ：判定「要调工具」→ 裁剪到这些工具，tool_choice 交回模型
        """
        # 取不到有效文本 → 保守降级：放行全部（宁可多给，别误杀）
        if not content:
            logger.debug("[ToolRouting] empty user content, fallback to all tools")
            return None, None

        try:
            # select_tools 返回 List[BaseTool]（工具对象），取 .name 得到名字集合
            picked = select_tools(content)
        except Exception as e:  # noqa
            # 路由出错时不要硬挂掉整个请求，降级为"给全部工具"
            logger.exception("[ToolRouting] select_tools failed, fallback to all tools: %s", e)
            return None, None

        # 空工具集 = 判定为「直接回答」。
        # ⚠️ 这里返回空 set 的含义是"声明照旧、只禁调用"，绝不是"把 tools 覆盖成空列表"——
        #    后者会让模型失去工具通道，退化成正文 DSML（见文件头部说明）。
        if not picked:
            return set(), NO_TOOL_TOOL_CHOICE

        return {t.name for t in picked}, None

    def _extract_last_user_content(self, request) -> Optional[str]:
        """
        从 request 里安全地取出『最近一条用户消息』的文本内容。

        【关键：ModelRequest 的真实结构】
        ---------------------------------
        LangChain 1.3 的 ModelRequest 是 @dataclass，消息直接在 `request.messages`，
        且里面的元素是 langchain 的 BaseMessage 对象（如 HumanMessage/AIMessage），
        用 `.type` 区分角色、"user" 对应的是 `HumanMessage`（type == "human"）。

        常见踩坑：
           - ❌ request.input              → 根本没有这个属性，会 AttributeError
           - ❌ msg["role"] / msg.role     → BaseMessage 用 .type，不是 .role
           - ❌ 假设 content 一定是 str     → 多模态时是 list[dict]

        所以这里做了多重兼容：既能处理 BaseMessage 对象，也能处理裸 dict。
        """
        # 1. 取消息列表：标准字段是 request.messages（list[BaseMessage]）
        messages = getattr(request, "messages", None)
        if not isinstance(messages, (list, tuple)):
            # 极端兜底：万一未来结构变化，尝试其他位置
            logger.debug("[ToolRouting] request has no list-like .messages, fallback")
            return None

        # 2. 倒序遍历，找最后一条用户消息
        for msg in reversed(messages):
            # --- 判断角色：兼容 BaseMessage 对象 与 裸 dict 两种形态 ---
            if isinstance(msg, dict):
                role = msg.get("role")                       # dict 形态："user"
                content = msg.get("content", "")
            else:
                # BaseMessage 对象：用 .type，"user" 消息的 type 是 "human"
                msg_type = getattr(msg, "type", None)
                role = "user" if msg_type == "human" else msg_type  # human→user 归一化
                content = getattr(msg, "content", "")

            if role != "user":
                continue

            # 3. content 可能是 str，也可能是多模态 list[dict]
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                # 多模态格式：[ {"type": "text", "text": "..."}, ... ]
                texts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                joined = "".join(texts).strip()
                if joined:
                    return joined

        # 找不到有效的用户文本 → 返回 None，上层降级为"放行全部工具"
        return None


"""
使用示例
========

    from langchain.agents import create_agent  # 按你实际构造 agent 的方式导入
    from tool_routing_middleware import ToolRoutingMiddleware

    agent = create_agent(
        model=...,
        tools=all_tools,                 # 注册全部工具
        middleware=[ToolRoutingMiddleware()],   # ← 挂上这个中间件即可
    )

    # 异步调用时，会自动走 awrap_model_call，embedding 在线程池里跑，不阻塞事件循环
    await agent.ainvoke({"messages": [("user", "帮我查一下天气")]})
"""
