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
"""

import asyncio
import logging
from typing import Optional

from langchain.agents.middleware import AgentMiddleware

from self_packages.tool_router import select_tools  # 返回 List[BaseTool]，即工具对象列表

logger = logging.getLogger(__name__)


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
        同步版本：裁剪工具后再放行。
        逻辑与异步版完全一致，区别只在「怎么调用 select_tools」。
        """
        # 1. 取出用户最新一条消息的文本
        content = self._extract_last_user_content(request)

        # 2. 根据内容算出本次允许的工具（同步调用，同步环境里 OK）
        #    allowed_names 为 None 表示"路由失败/取不到内容"，此时保守放行全部工具
        allowed_names = self._compute_allowed(content)

        # 3. 用算出的名字去 request.tools 里过滤（保留 request 里的原始工具对象）
        #    若为 None → 降级：保留全部工具（宁可多给，别误杀）
        if allowed_names is None:
            return call(request)
        filtered = [t for t in request.tools if t.name in allowed_names]

        # 4. 放行（工具为空 = 不给工具，让模型直接文本回答）
        request = request.override(tools=filtered)
        if not filtered:
            logger.warning("[ToolRouting] no tools allowed, falling back to text-only answer")
        return call(request)

    async def awrap_model_call(self, request, call):
        """
        异步版本：裁剪工具后再放行。
        与同步版唯一区别：select_tools 含模型推理，必须丢到线程池避免阻塞事件循环。
        """
        content = self._extract_last_user_content(request)

        # ★ 关键点：select_tools 是同步且有 CPU/GPU 计算的，不能在 async 里直接 await 它，
        #   而是把它提交到默认线程池，让出事件循环给其他协程。
        allowed_names = await asyncio.get_event_loop().run_in_executor(
            None,                          # 使用默认的 ThreadPoolExecutor
            self._compute_allowed,         # 要执行的函数
            content,                       # 函数的参数
        )

        if allowed_names is None:
            return await call(request)
        filtered = [t for t in request.tools if t.name in allowed_names]
        request = request.override(tools=filtered)
        if not filtered:
            logger.warning("[ToolRouting] no tools allowed, falling back to text-only answer")
        return await call(request)

    # ------------------------------------------------------------------ #
    # 下面是抽取出来的公共逻辑（同步/异步共用，保证一致性）
    # ------------------------------------------------------------------ #
    def _compute_allowed(self, content: Optional[str]) -> Optional[set]:
        """
        根据用户输入文本，算出『本次允许使用的工具名集合』。
        抽成独立方法，是为了让同步/异步两个入口都走同一份逻辑。

        参数:
            content: 用户最新一条消息的文本；取不到时为 None。
        返回:
            - set[str]          ：允许的工具名集合（可为空 = 不给任何工具）
            - None              ：路由失败 / 取不到有效内容，调用方应"放行全部"作为降级
        """
        # 取不到有效文本 → 保守降级：放行全部（宁可多给，别误杀）
        if not content:
            logger.debug("[ToolRouting] empty user content, fallback to all tools")
            return None

        try:
            # select_tools 返回 List[BaseTool]（工具对象），取 .name 得到名字集合
            allowed = {tool.name for tool in select_tools(content)}
        except Exception as e:  # noqa
            # 路由出错时不要硬挂掉整个请求，降级为"给全部工具"
            logger.exception("[ToolRouting] select_tools failed, fallback to all tools: %s", e)
            return None

        return allowed

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
    await agent.ainvoke({"input": "帮我查一下天气"})
"""
