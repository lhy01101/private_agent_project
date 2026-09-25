"""
工具路由中间件（ToolRoutingMiddleware）
===========================================

【这个中间件是干什么的？】
--------------------------
在 Agent 真正调用大模型之前，拦截这次请求，根据用户最新的提问，
用 tool_router 算出"本次该暴露哪些工具"，然后把 agent 可用的工具列表
裁剪到只剩这些。模型看不到无关工具 → 减少幻觉、降低 token、提升准确率。

注意：本中间件【只做裁剪】，不做"禁止调用"。为什么，见下面第 5 节。

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
   → 本文件的做法：把"决策"（_compute_decision）和"组装"（_plan）都抽成纯逻辑，
     同步/异步两个入口只保留 call / await call 的区别，逻辑不可能漂移。

5. 为什么只保留"裁剪权"，放弃"否决权"？
---------------------------------------
   两种权力的误判代价是不对称的：
   - 裁剪权（软）：判错无非多给/少给一个工具，auto 下模型可以不用它 → 可恢复。
   - 否决权（tool_choice="none"）：判错则这一轮模型彻底没有信息通道，
     只能硬答或编造 → 不可恢复，用户直接看到错答案。

   而"没有任何路由过闸"是一个【残差判定】：它同时覆盖"真不需要工具"（闲聊）和
   "漏检"（"现在几点了"这种必然需要工具的），两者无法区分、正确性不可验证。
   残差判定不该持有不可恢复的权力 → 所以它只行使裁剪权，放弃否决权。

   残差时的处理是【不干预】：工具声明照旧给全 + 默认 auto，让模型自己决定。
   这等于把"该不该调"这个决策交回给掌握完整上下文的模型，中间件只负责
   在路由有把握时缩小候选集。

   ⚠️ 唯一必须守住的不变式：本中间件永远不会把 request.tools 覆盖为空列表。
      一旦 tools 为空，langchain/agents/factory.py 里的 `if final_tools:`
      会让 bind_tools 整段跳过、连 tool_choice 字段都不会进请求体，
      模型失去工具通道，而 prompt 还在要求它调工具 → 它只能把调用写成正文，
      也就是 DSML 文本。
      「没有候选」的正确表达是【不干预】（决策返回 None，原样放行），不是【关闸】。
"""

import asyncio
import logging
from typing import Optional, Sequence

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
        同步版本：先算决策 → 再算本次请求怎么组装 → 放行。
        逻辑与异步版完全一致，区别只在「怎么调用 select_tools」（这里可以直接调）。
        """
        # 1. 取用户最新一条消息的文本
        content = self._extract_last_user_content(request)

        # 2. 本次调用是否处在一次工具回路中间（见 _in_tool_loop）
        in_tool_loop = self._in_tool_loop(request)

        # 3. 算出本次要保留哪些工具（同步调用，同步环境里 OK）
        allowed_names = self._compute_decision(content, in_tool_loop)

        # 4. 把决策翻译成"这次请求怎么组装"
        filtered = self._plan(request, allowed_names)

        # 5a. None → 不干预：工具声明照旧给全，tool_choice 也不动（沿用默认 auto）
        if filtered is None:
            return call(request)

        # 5b. 裁剪：只暴露相关工具。tool_choice 一概不动，交回模型自己决定。
        #     ★ 关键：filtered 必然非空，永远不会产生 tools=[] 的请求。
        return call(request.override(tools=filtered))

    async def awrap_model_call(self, request, call):
        """
        异步版本：与同步版唯一区别——select_tools 含模型推理（embedder.embed_query），
        必须丢到线程池，避免阻塞整个事件循环。
        """
        content = self._extract_last_user_content(request)
        in_tool_loop = self._in_tool_loop(request)

        # ★ 关键点：select_tools 是同步且有 CPU/GPU 计算的，不能在 async 里直接 await 它，
        #   而是把它提交到默认线程池，让出事件循环给其他协程。
        #   注意：_compute_decision 只接收"纯数据"（content + 一个布尔量），不碰 request，
        #   所以在别的线程里跑是安全的。
        allowed_names = await asyncio.get_event_loop().run_in_executor(
            None,                          # 使用默认的 ThreadPoolExecutor
            self._compute_decision,        # 要执行的函数
            content,                       # 参数 1
            in_tool_loop,                  # 参数 2
        )

        filtered = self._plan(request, allowed_names)

        # 不干预：工具声明照旧给全，tool_choice 也不动（沿用默认 auto）
        if filtered is None:
            return await call(request)

        # 裁剪：只暴露相关工具（filtered 必然非空，永不产生 tools=[]）
        return await call(request.override(tools=filtered))

    # ------------------------------------------------------------------ #
    # 下面是抽取出来的公共逻辑（同步/异步共用，保证一致性）
    # ------------------------------------------------------------------ #
    def _compute_decision(
        self,
        content: Optional[str],
        in_tool_loop: bool = False,
    ) -> Optional[set]:
        """
        根据用户输入文本，算出本次要保留的工具名集合。这是纯粹的"决策"层，
        不含任何 langchain 对象操作，因此可以安全地丢进线程池。

        参数:
            content: 用户最新一条消息的文本；取不到时为 None。
            in_tool_loop: 本次调用是否处在工具回路中间（见 _in_tool_loop）。

        返回:
            - None        ：不干预 → 调用方原样放行（工具声明给全 + 默认 auto）
                            四种来源：取不到内容 / 正处在工具回路中 /
                                      路由异常 / 没有任何工具路由过闸（残差空）
            - {工具名, ...}：裁剪到这些工具，tool_choice 交回模型（默认 auto）

        ⚠️ 契约约束：本方法【不返回空集合】。
           "没有候选"必须表达成 None（不干预），而不是 set()。
           理由见文件头部第 5 节：否决权已被整体移除，set() 只剩语义漂移的隐患
           （空 set 是 falsy，会悄悄命中调用方的"禁止调用"分支）。
        """
        # 取不到有效文本 → 不干预（宁可多给，别误杀）
        if not content:
            logger.debug("[ToolRouting] empty user content → pass-through")
            return None

        # 正处在工具回路中 → 不干预。
        # select_tools 只看"最后一条用户消息"的文本；回路中间那几轮文本没变，
        # 但模型很可能要复用刚调过的工具。此时若裁剪，容易把刚用过的工具裁掉，
        # 模型就会"接着上文硬调"（正文里出现假调用 / DSML）。
        if in_tool_loop:
            logger.debug("[ToolRouting] inside tool loop → pass-through")
            return None

        try:
            # select_tools 返回 List[BaseTool]（工具对象），取 .name 得到名字集合
            picked = select_tools(content)
        except Exception as e:  # noqa
            # 路由出错时不要硬挂掉整个请求，降级为"不干预"
            logger.exception("[ToolRouting] select_tools failed → pass-through: %s", e)
            return None

        # 残差空：没有任何工具路由过闸。这里混着"真不需要工具"与"路由漏检"两类，
        # 无法区分，所以只放弃裁剪、绝不关闸 —— 声明给全 + auto，让模型自己判断。
        if not picked:
            logger.debug("[ToolRouting] no route passed threshold → pass-through")
            return None

        return {t.name for t in picked}

    def _plan(self, request, allowed_names: Optional[set]) -> Optional[Sequence]:
        """
        把「决策」翻译成「本次请求怎么组装」，同时也是唯一那条不变式的守门人。

        返回:
            - None      ：本次不 override tools（原样放行）
            - [工具对象] ：裁剪到这些工具（保证非空）

        这里集中处理两种"兜底放行"的情况：
          1) 上游决策本来就是 None；
          2) 决策给了名字，但和 request.tools 对不上（工具名未注册 / 同名冲突）
             —— 这是**数据 bug**，不是路由决策，不能拿它去封死工具通道。
                放行让模型用全量工具兜住，同时打 warning 提醒修数据。
        """
        if not allowed_names:
            return None

        if not request.tools:
            # agent 压根没注册工具：无事可做，也不该报警
            logger.debug("[ToolRouting] request.tools is empty → pass-through")
            return None

        filtered = [t for t in request.tools if t.name in allowed_names]
        if not filtered:
            logger.warning(
                "[ToolRouting] 裁剪结果为空（工具名未注册/同名冲突？）→ 放行全部：%s",
                sorted(allowed_names),
            )
            return None

        logger.debug(
            "[ToolRouting] trim: %s/%s → %s",
            len(filtered),
            len(request.tools),
            sorted(t.name for t in filtered),
        )
        return filtered

    # ------------------------------------------------------------------ #
    # 工具回路判定
    # ------------------------------------------------------------------ #
    @staticmethod
    def _in_tool_loop(request) -> bool:
        """
        本次模型调用是否处在一次「工具回路」中间？

        判据：**最后一条用户消息之后**，已经出现了工具调用（ai 消息带 tool_calls）
              或工具结果（ToolMessage）。

        为什么只看"最后一条用户消息之后"，而不是扫整个历史：
        agent 的工具回路就是在最后一条 user 消息之后不断追加
        ai(tool_calls) → tool(result) → ai(...)，直到给出最终答复。
        用户一发新消息，回路就结束了，此时应该恢复正常裁剪。
        如果扫整个历史，那么一个长会话里只要调过一次工具，之后每一轮都会永远跳过裁剪
        —— 中间件等于永久失效。这个边界很容易写错，务必留意。

        兼容 BaseMessage 对象与裸 dict 两种形态。
        """
        messages = getattr(request, "messages", None)
        if not isinstance(messages, (list, tuple)):
            return False

        # 1. 先定位最后一条用户消息的位置
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if ToolRoutingMiddleware._is_user_message(msg):
                last_user_idx = i

        # 2. 只看它之后的部分有没有工具活动
        for msg in messages[last_user_idx + 1:]:
            if ToolRoutingMiddleware._has_tool_activity(msg):
                return True
        return False

    @staticmethod
    def _is_user_message(msg) -> bool:
        """兼容两种形态的消息角色判定（与 _extract_last_user_content 保持一致）。"""
        if isinstance(msg, dict):
            return msg.get("role") == "user"
        return getattr(msg, "type", None) == "human"

    @staticmethod
    def _has_tool_activity(msg) -> bool:
        """这条消息是否携带了工具调用（ai.tool_calls）或工具结果（ToolMessage）。"""
        if isinstance(msg, dict):
            return bool(msg.get("tool_calls")) or msg.get("role") == "tool"
        if getattr(msg, "tool_calls", None):
            return True
        return getattr(msg, "type", None) == "tool"

    # ------------------------------------------------------------------ #
    # 文本提取
    # ------------------------------------------------------------------ #
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

【行为速查（改版后）】
======================
    路由命中         → 只暴露命中的工具（tool_choice 保持默认 auto）
    路由没命中       → 不干预：全部工具 + auto，模型自己决定（不再禁止调用）
    处在工具回路中   → 不干预（最后一条用户消息之后已有 tool_calls / tool 结果）
    工具名对不上     → 不干预 + warning（这是数据 bug，不该封死工具通道）
"""
