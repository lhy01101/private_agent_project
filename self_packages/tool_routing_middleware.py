"""
工具路由中间件（ToolRoutingMiddleware）
===========================================

【这个中间件是干什么的？】
--------------------------
在 Agent 真正调用大模型之前 / 之后拦截请求，做两件事：

  (A) 调模型【前】—— 用 tool_router 裁剪工具列表（原有能力）
      → 模型只看到本次需要的工具，减少幻觉、节省 token

  (B) 调模型【后】—— 检测 & 规范化模型的工具调用（本次新增）
      → DeepSeek 有时会输出 DSML 多调用块（<｜｜DSML｜｜ calls> ...），
        框架默认解析器不认识 → 工具调用被静默丢弃。
        本中间件把 DSML 块解析成标准 tool_call 结构，并对「同一工具重复调用」
        （如中/英文两个 query）做合并去重。

【核心机制：你必须理解这几点】
---------------------------------
1. 框架在"调工具"、"调模型"这两个时机，会回调对应的 wrap_* / awrap_* 方法。
   - wrap_tool_call / awrap_tool_call ：拦截「工具执行」（本中间件直接透传）
   - wrap_model_call / awrap_model_call：拦截「模型调用」  ← 裁剪 + DSML 规范化都在这里

2. wrap_model_call 的完整流程：
       request ──▶ [裁剪工具] ──▶ call(request) ──▶ response
                                                     │
                                          [检测 DSML 块] ──▶ 解析 / 合并 / 重写 response
                                                     │
                                                 返回 response

3. 想改参数？用 request.override(字段=新值) / response.override(...) 生成新对象。
   ⚠️ 永远不要原地修改，要用 override 返回新对象（不可变思想）。

4. 同步方法（wrap_*）给「同步 agent」用，异步方法（awrap_*）给「异步 agent」用。
   两者逻辑必须一致，否则同步/异步跑出不同结果，极难排查。

【关于 select_tools 的注意事项】
--------------------------------
select_tools() 内部会跑 embedder.embed_query()，是一次模型推理（CPU/GPU 计算）。
- 在同步环境（wrap_model_call）里直接调用没问题。
- 在异步环境（awrap_model_call）里【不能】直接调用，否则会阻塞整个事件循环。
  → 用 asyncio 的 run_in_executor 把它丢到线程池里去跑。

【关于 DSML 规范化的注意事项】
------------------------------
- DSML 出现在【模型输出】里，因此规范化逻辑放在「call(request) 之后」处理 response。
- 解析出的 tool_calls 会尝试写回 response，让框架正常执行；若你的 LangChain 版本
  的 ModelResponse 不支持 .override(tool_calls=...)，请参照下方「接入说明」自行适配。
- 合并策略：同一工具名只保留一个调用，query 取「更长 / 信息量更大」的那条
  （通常是更完整的表达，丢弃其翻译/重复版本）。
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from langchain.agents.middleware import AgentMiddleware

from self_packages.tool_router import select_tools  # 返回 List[BaseTool]，即工具对象列表

logger = logging.getLogger(__name__)


# ====================================================================== #
# 一、DSML 解析 & 合并（与中间件主体解耦，方便单独单测）
# ====================================================================== #

@dataclass
class ParsedToolCall:
    """从 DSML 块中解析出的单个工具调用。"""
    name: str
    arguments: dict = field(default_factory=dict)


class DSMLParser:
    """
    DeepSeek DSML 多调用块的解析器。

    支持的输入形态（均会被匹配）：
        <｜｜DSML｜｜ calls>
            <｜｜DSML｜｜ invoke name="search_knowledge_base">
                <｜｜DSML｜｜ parameter name="query" string="true">工具提示词</｜｜DSML｜｜ parameter>
            </｜｜DSML｜｜ invoke>
            ...
        </｜｜DSML｜｜ calls>

    说明：
        - 一个 <calls> 块内可有多个 <invoke>（即一次声明多个并行调用）。
        - 同一 <invoke> 可能有多个 <parameter>（本类会把同名参数合并为 list）。
    """

    # 整段 calls 块（分隔符是全角竖线 ｜ = U+FF5C，此处直接写字面量，避免编码歧义）
    _CALLS_BLOCK = re.compile(
        r"<｜｜DSML｜｜\s*calls\s*>"
        r"(.*?)"
        r"</｜｜DSML｜｜\s*calls\s*>",
        re.DOTALL,
    )
    # 单个 invoke（name 用命名分组，供 parse 取值）
    _INVOKE = re.compile(
        r'invoke\s+name\s*=\s*"(?P<name>[^"]+)"',
        re.DOTALL,
    )
    # 单个 parameter：name="..." [type="..."] > value <
    _PARAM = re.compile(
        r'parameter\s+name\s*=\s*"([^"]+)"'
        r'(?:\s+[^\s>]+)?'          # 容忍 string="true" / type="..." 等额外属性
        r"\s*>"
        r"(.*?)"
        r"</",
        re.DOTALL,
    )

    @classmethod
    def contains_dsml(cls, text: str) -> bool:
        """快速判断一段文本是否包含 DSML calls 块。"""
        if not text:
            return False
        return "<｜｜DSML｜｜ calls>" in text or bool(cls._CALLS_BLOCK.search(text))

    @classmethod
    def parse(cls, text: str) -> list[ParsedToolCall]:
        """
        把 DSML 文本解析成结构化调用列表。
        找不到任何块时返回 []（不抛异常，保证中间件不中断主链路）。

        处理策略：先定位 <calls> 块，再逐个 <invoke>...</invoke> 子块解析，
        每块内收集所有 <parameter>，同名参数合并为 list。
        """
        if not text:
            return []

        calls: list[ParsedToolCall] = []
        for block in cls._CALLS_BLOCK.findall(text):
            # 逐个 <invoke name="...">...</invoke> 块处理。
            # 用 finditer 拿到每个 invoke 的起止：起始=name 后，终止=下一个 invoke 起点或块尾。
            # 区间内的 <parameter> 统一交给 _PARAM 正则提取（兼容简写闭合）。
            invoke_spans = list(cls._INVOKE.finditer(block))
            for i, invoke_match in enumerate(invoke_spans):
                name = invoke_match.group("name").strip()
                start = invoke_match.end()
                # 终止：下一个 invoke 的起点；没有则用块尾
                if i + 1 < len(invoke_spans):
                    end = invoke_spans[i + 1].start()
                else:
                    end = len(block)
                chunk = block[start:end]

                args: dict = {}
                for p_name, p_value in cls._PARAM.findall(chunk):
                    p_name = p_name.strip()
                    p_value = p_value.strip()
                    # 同名参数 → 合并成 list（保留顺序）
                    if p_name in args:
                        existing = args[p_name]
                        if isinstance(existing, list):
                            existing.append(p_value)
                        else:
                            args[p_name] = [existing, p_value]
                    else:
                        args[p_name] = p_value

                calls.append(ParsedToolCall(name=name, arguments=args))

        return calls


def merge_duplicate_calls(calls: list[ParsedToolCall]) -> list[ParsedToolCall]:
    """
    合并「同一工具」的重复调用。

    规则：相同 name 只保留一个。当多个调用争用时，query 取「信息量更大」的那条——
    用 (长度, 是否像英文/翻译) 综合判断：优先保留更长的中文原句，丢弃其翻译/重复版本。

    例：
        search_knowledge_base("工具提示词 负例 机制")
        search_knowledge_base("tool prompt 负例 示例 工具选择")   ← 英文翻译，丢弃
        web_search("iPhone 18 发布 价格 配置")
        web_search("iPhone 18 release date specs price 2026")   ← 英文翻译，丢弃
        → 各保留 1 个
    """
    def _query_info(call: ParsedToolCall) -> str:
        # 兼容参数名叫 query / q / input 的常见情况
        for key in ("query", "q", "input", "question"):
            val = call.arguments.get(key)
            if isinstance(val, str) and val:
                return val
        # 兜底：取第一个字符串参数
        for val in call.arguments.values():
            if isinstance(val, str) and val:
                return val
        return ""

    def _score(text: str) -> tuple:
        """
        信息量评分，用于「同一工具的多个调用」争用时决定保留哪一个。

        优先级：
          1) CJK 字符占比高 → 视为「原文」，优先保留（剔除其翻译副本）；
          2) 占比相当时，取更长（信息更完整）的那条。

        用「占比」而非「个数」，是为了区分「纯中文原文」与
        「英文翻译里夹了几个中文词」这类混合文本。

        返回 tuple 可直接比较大小：(cjk_ratio, length)
        """
        if not text:
            return (-1.0, 0)
        cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
        ratio = cjk / len(text)
        return (ratio, len(text))

    seen: dict[str, ParsedToolCall] = {}
    order: list[str] = []
    for call in calls:
        name = call.name
        if name in seen:
            # 争用：保留信息量更大的那条
            old_info = _query_info(seen[name])
            new_info = _query_info(call)
            if _score(new_info) > _score(old_info):
                seen[name] = call
        else:
            seen[name] = call
            order.append(name)

    return [seen[n] for n in order]


# ====================================================================== #
# 二、中间件主体
# ====================================================================== #

class ToolRoutingMiddleware(AgentMiddleware):
    name = "tool_routing"
    # trace_policy：是否需要追踪/埋点。None 表示不启用，保持默认即可。
    trace_policy = None

    # ---- 可通过构造参数调整的行为 ----
    def __init__(
        self,
        *,
        enable_dsml: bool = True,        # 是否启用 DSML 检测/规范化
        dsml_merge: bool = True,         # 是否合并同一工具的重复调用
        dsml_strip_from_text: bool = True,  # 规范化后是否从文本里剥离 DSML 原文
    ):
        self.enable_dsml = enable_dsml
        self.dsml_merge = dsml_merge
        self.dsml_strip_from_text = dsml_strip_from_text

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
    # 模型调用拦截（核心：裁剪工具 + DSML 规范化）
    # ------------------------------------------------------------------ #
    def wrap_model_call(self, request, call):
        # —— 阶段 1：裁剪工具（调模型前）——
        request = self._filter_tools(request)

        # —— 阶段 2：调用模型，拿到 response ——
        response = call(request)

        # —— 阶段 3：规范化 DSML（调模型后）——
        if self.enable_dsml:
            response = self._normalize_dsml(response)

        return response

    async def awrap_model_call(self, request, call):
        # —— 阶段 1：裁剪工具（调模型前）——
        #   select_tools 含模型推理，必须丢到线程池避免阻塞事件循环
        content = self._extract_last_user_content(request)
        allowed_names = await asyncio.get_event_loop().run_in_executor(
            None,                          # 使用默认的 ThreadPoolExecutor
            self._compute_allowed,         # 要执行的函数
            content,                       # 函数的参数
        )
        if allowed_names is None:
            # 路由失败 → 降级放行全部工具（在 override 里保留原 tools）
            pass
        else:
            filtered = [t for t in request.tools if t.name in allowed_names]
            request = request.override(tools=filtered)
            if not filtered:
                logger.warning("[ToolRouting] no tools allowed, falling back to text-only answer")

        # —— 阶段 2：调用模型 ——
        response = await call(request)

        # —— 阶段 3：规范化 DSML（调模型后）——
        if self.enable_dsml:
            response = await asyncio.get_event_loop().run_in_executor(
                None, self._normalize_dsml, response
            )

        return response

    # ------------------------------------------------------------------ #
    # 阶段 1：工具裁剪（同步/异步共用）
    # ------------------------------------------------------------------ #
    def _filter_tools(self, request):
        """同步版的裁剪入口（供 wrap_model_call 使用）。"""
        content = self._extract_last_user_content(request)
        allowed_names = self._compute_allowed(content)
        if allowed_names is None:
            return request  # 降级：保留全部工具
        filtered = [t for t in request.tools if t.name in allowed_names]
        if not filtered:
            logger.warning("[ToolRouting] no tools allowed, falling back to text-only answer")
        return request.override(tools=filtered)

    # ------------------------------------------------------------------ #
    # 阶段 3：DSML 规范化
    # ------------------------------------------------------------------ #
    def _normalize_dsml(self, response) -> object:
        """
        检测 response 里的 DSML 块 → 解析 → 合并 → 尝试写回标准 tool_calls。

        设计原则：
          - 任何异常都不中断主链路（最多记日志 + 保留原始 response）。
          - 优先写回 response.tool_calls（让框架正常执行）；
            同时可选地剥离文本里的 DSML 原文，避免模型把它当正文复述。
        """
        # 3.1 取出模型输出的文本内容（兼容多种 response 结构）
        text = self._extract_response_text(response)
        if not DSMLParser.contains_dsml(text):
            return response  # 没有 DSML，原样返回

        # 3.2 解析
        raw_calls = DSMLParser.parse(text)
        if not raw_calls:
            logger.debug("[DSML] detected DSML marker but parsed 0 calls; keep original")
            return response

        # 3.3 合并重复调用
        if self.dsml_merge:
            before = len(raw_calls)
            raw_calls = merge_duplicate_calls(raw_calls)
            if len(raw_calls) != before:
                logger.info("[DSML] merged duplicate calls: %d → %d", before, len(raw_calls))

        # 3.4 转成框架通用的 dict 结构 {name, arguments}
        normalized = [
            {"name": c.name, "arguments": c.arguments} for c in raw_calls
        ]
        logger.info("[DSML] normalized %d call(s): %s",
                    len(normalized), [c["name"] for c in normalized])

        # 3.5 写回 response（尽量用 override；字段名随 LangChain 版本而异）
        response = self._inject_tool_calls(response, normalized)

        # 3.6 可选：从正文里剥离 DSML 原文，防止被当作普通文本复述
        if self.dsml_strip_from_text:
            response = self._strip_dsml_from_response(response, text)

        return response

    def _inject_tool_calls(self, response, tool_calls: list[dict]) -> object:
        """
        把规范化后的 tool_calls 写回 response。
        优先尝试 response.override(tool_calls=...)；失败则尝试直接赋值 / 设置属性。
        （不同 LangChain 版本的 ModelResponse 字段名略有差异，此处做兼容。）
        """
        # 情况 A：支持 override（推荐路径，不可变）
        override = getattr(response, "override", None)
        if callable(override):
            try:
                return override(tool_calls=tool_calls)
            except TypeError:
                # override 不接受 tool_calls 关键字 → 退到情况 B
                pass

        # 情况 B：可直接设置属性（dataclass-like）
        if hasattr(response, "tool_calls"):
            try:
                response.tool_calls = tool_calls
                return response
            except Exception:  # noqa
                pass

        # 情况 C：都失败 → 保留原始 response，仅记录（不中断链路）
        logger.warning(
            "[DSML] cannot inject tool_calls into response of type %s; "
            "fallback to original response. tool_calls=%s",
            type(response).__name__, tool_calls,
        )
        return response

    def _strip_dsml_from_response(self, response, original_text: str) -> object:
        """把正文里的 <｜｜DSML｜｜ calls>...</...calls> 块剥离，替换为简短占位。"""
        cleaned = DSMLParser._CALLS_BLOCK.sub(
            "\n[已解析的工具调用]\n", original_text
        )
        if cleaned == original_text:
            return response

        override = getattr(response, "override", None)
        if callable(override):
            try:
                return override(content=cleaned)
            except TypeError:
                pass
        if hasattr(response, "content"):
            try:
                response.content = cleaned
                return response
            except Exception:  # noqa
                pass
        return response

    # ------------------------------------------------------------------ #
    # 下面是抽取出来的公共逻辑（同步/异步共用，保证一致性）
    # ------------------------------------------------------------------ #
    def _compute_allowed(self, content: Optional[str]) -> Optional[set]:
        """
        根据用户输入文本，算出『本次允许使用的工具名集合』。
        抽成独立方法，是为了让同步/异步两个入口都走同一份逻辑。

        返回:
            - set[str] : 允许的工具名集合（可为空 = 不给任何工具）
            - None     : 路由失败 / 取不到有效内容，调用方应"放行全部"作为降级
        """
        if not content:
            logger.debug("[ToolRouting] empty user content, fallback to all tools")
            return None

        try:
            allowed = {tool.name for tool in select_tools(content)}
        except Exception as e:  # noqa
            logger.exception("[ToolRouting] select_tools failed, fallback to all tools: %s", e)
            return None

        return allowed

    def _extract_last_user_content(self, request) -> Optional[str]:
        """
        从 request 里安全地取出『最近一条用户消息』的文本内容。

        LangChain 1.3 的 ModelRequest 是 @dataclass，消息在 `request.messages`，
        元素是 BaseMessage（HumanMessage/AIMessage），用 `.type` 区分角色，
        "user" 对应的是 HumanMessage（type == "human"）。

        常见踩坑：
           - ❌ request.input              → 没有这个属性
           - ❌ msg["role"] / msg.role     → BaseMessage 用 .type
           - ❌ 假设 content 一定是 str     → 多模态时是 list[dict]
        """
        messages = getattr(request, "messages", None)
        if not isinstance(messages, (list, tuple)):
            logger.debug("[ToolRouting] request has no list-like .messages, fallback")
            return None

        for msg in reversed(messages):
            if isinstance(msg, dict):
                role = msg.get("role")
                content = msg.get("content", "")
            else:
                msg_type = getattr(msg, "type", None)
                role = "user" if msg_type == "human" else msg_type
                content = getattr(msg, "content", "")

            if role != "user":
                continue

            if isinstance(content, str):
                return content
            if isinstance(content, list):
                texts = [
                    p.get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                joined = "".join(texts).strip()
                if joined:
                    return joined

        return None

    def _extract_response_text(self, response) -> str:
        """
        从模型响应里取出文本内容，兼容多种结构：
           - response.content (str / list)
           - response["content"]
           - response.text
        取不到则返回 ""（不抛异常）。
        """
        # 对象属性
        for attr in ("content", "text"):
            val = getattr(response, attr, None)
            if isinstance(val, str) and val:
                return val
            if isinstance(val, list):  # 多模态 content 列表
                joined = "".join(
                    p.get("text", "") for p in val
                    if isinstance(p, dict) and p.get("type") == "text"
                )
                if joined:
                    return joined

        # dict 形态
        if isinstance(response, dict):
            val = response.get("content") or response.get("text")
            if isinstance(val, str):
                return val

        return ""


"""
使用示例
========

    from langchain.agents import create_agent  # 按你实际构造 agent 的方式导入
    from tool_routing_middleware import ToolRoutingMiddleware

    agent = create_agent(
        model=...,
        tools=all_tools,
        middleware=[ToolRoutingMiddleware(
            enable_dsml=True,        # 开启 DSML 检测/规范化
            dsml_merge=True,         # 合并同一工具的重复调用（中/英文去重）
            dsml_strip_from_text=True,
        )],
    )

    await agent.ainvoke({"input": "帮我查一下天气"})

接入说明（重要）
================
1. DSML 规范化发生在「模型调用之后」。如果你的 LangChain 版本里，
   ModelResponse 不支持 response.override(tool_calls=...)，
   请检查日志中的 "[DSML] cannot inject tool_calls ..." 警告，
   并按你的版本把 _inject_tool_calls 里的「情况 B/C」补上对应字段。

2. 更彻底的方案是从模型侧禁用 DSML、走标准 function calling
   （ollama 部署可配 tool_choice / format），从源头不产生 DSML。
   本中间件的规范化逻辑可作为「兜底兼容层」长期保留。

3. 建议先用一次请求打印验证：
       logger.setLevel(logging.INFO)
   观察是否出现 "[DSML] normalized N call(s): [...]" 日志，
   确认写回是否生效；若未生效，按说明 1 调整字段名。
"""
