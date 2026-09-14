## 技术路径

### 大模型输出
- result["messages"] = history + [user_new, ai_new, tool_new, ai_final]
- prev_len = 0 # 用来记录上一次的消息数量 
- 只打印新增的消息 
- new_messages = result["messages"][prev_len:]
- prev_len = len(result["messages"])

### 添加记忆
- from langgraph.checkpoint.memory import InMemorySaver 
- 使用checkpointer = InMemorySaver()

### 循环+规则判断
  - "自己发现错误并换路子"的意愿和智力在模型里，"能不能继续跑下去"的机制在人写的循环里。真正的 Agent 能力 = 模型的推理 × 框架的持久化执行。
  - 如果你在考虑自己搭 Agent 框架，这个分工决定了：调优时策略质量改 prompt 和模型，稳定性和鲁棒性改循环逻辑和异常处理。两边问题容易混，定位时要先分清是哪层的事。
  - 实际工程里的推荐路径：
*第一层：system prompt + few-shot*
    *↓ 效果不够*
*第二层：框架层兜底（循环里加规则判断）*
  - *连续相同 action → 强制打断，注入"你重复了，必须换策略"*
  - *特定 error → 代码直接给 hint 塞回 context*
  - *步数超阈值 → 强制输出"当前方案不可行，请重新规划"*
    *↓ 还是不够*
*第三层：针对特定子任务做 SFT（不是全 agent 循环微调）*
  - *只微调"看报错→改参数"这个子步骤*
  - *其他步骤仍用通用模型*
    *↓ 最后才考虑*
*第四层：RL/RLHF on trajectory（成本高，一般团队不碰）*

### 语义路由（分层调模型/工具）
- 为什么做语义路由+Agent管线：工具越多，Agent 越慢越贵；工具太多 → LLM 选错；控制力：有些事你不想让 LLM 自己决定
- Agent 管线：把"回答问题"从"一步到位"变成"多步骤协作"，能处理复杂任务 ；语义路由：Agent 的"前台调度员"，用零成本的向量距离决定走哪条管线，避免工具爆炸 + 省钱 + 提准确率
- 也就是说我想给什么工具的时候我再给LLM，而不是一开始就塞20个工具进去，通过语义路由（是不是也类似RAG embedding的retrieve来选择工具？），在具体提问时选择部分工具交给LLM，相当于一定程度上解决了agent工具调用的问题


*一个基本的流程：*
- *用户问题*
- *用 bge-m3 做语义路由（选知识库）   ← 快，CPU 跑*
- *用 bge-m3 检索文档 chunk            ← 向量库*
- *用 bge-reranker 精排 top3(对于RAG就是top3文本，对于Agent管线就是top3工具)          ← CPU*
- *用 Qwen2.5-7B（GPU）生成答案        ← 只干一件事：说人话*

*设计思路*
- 用户问题
- embed（复用 nomic-embed-text，离线、免费）
- 与每个"路由"的示例句做余弦相似度
- 取 ≥ threshold 的路由（支持多选 → 组合工具）
- 汇总成工具列表 [rag_tool_1, web_search, ...]
- 只把这些工具交给 agent
- - 关键点：
- - 每个路由 = 一组示例句 + 对应工具（或工具组合）
- - 向量距离 = 示例句的 embedding 与 query 的最大余弦相似度
- - 多选 = 组合工具（如 rag_tool_1 + web_search）
- - 低于阈值 → 不暴露任何工具 → 直接回答（省钱）

### DeepSeek V4-Flash 的“缓存命中”到底是什么？
- 它做的是 Prompt Cache（前缀 KV 缓存），不是语义缓存，也不是工具路由：
- 机制：你发请求时，从第一个 token 开始的前缀如果和之前某次请求逐字节完全一致，服务端就复用之前算好的 KV 中间状态，不再重算 prefill
- 计费：命中部分按 prompt_cache_hit_tokens 算，价格约是未命中的 1/50（Flash 命中 0.02 元/M，未命中 1 元/M）
- 关键约束：是“精确前缀匹配”，不是语义相似匹配。系统提示词里加个 "今天是2026-09-06"、工具顺序变一下、RAG 召回文档顺序变一下 → 前缀断了 → 缓存失效
- 默认开启，不用你写代码，但命中率完全取决于你怎么排 prompt
- 所以它省的是：“同一段长系统提示 / 同一份固定工具 schema / 同一段不变上下文”被反复发给模型时，输入侧重算的钱和延迟。
- 语义路由解决什么： 20 个工具里挑 3 个给 LLM，降决策错误率+省 token
- DeepSeek Prompt Cache解决什么：选定工具后，系统提示/工具 schema 不变 → 省 prefill 钱
- Prompt Cache：**不用“做”，但要“顺手蹭”**
- - 你不需要写缓存代码，但prompt 排版不对 = 命中率 0：
- - 稳定内容放最前：system 提示 → 工具 schema（排序固定）→ 固定参考文档
- - 动态内容放最后：检索到的 chunk → 多轮 history → 当前用户问题
- - 禁止在 system 里塞 时间戳 / session_id / 随机 trace_id
- - 工具列表用固定顺序序列化（dict 别乱迭代）

### 联网搜索机制
- 用户问 → ① 调 Bing/Google API（**关键词检索**/query 改写后的关键词）
- → ② 拿到 Top 10 网页 URL + 快照
- → ③ 爬正文、清 HTML、**分块**
- → ④ 每块 bge-m3 **embedding**（动态，不预存）
- → ⑤ 用户 query 也 embedding
- → ⑥ 向量检索 Top-3 块
- → ⑦ 可选 rerank
- → ⑧ LLM 生成

### Rerank / 重排 API
- 输入"1 个 query + N 条已检索到的候选文本"，输出这 N 条重新打分后的顺序。​ 它不联网、不找新东西，只在你给的那堆里挑。
- 典型：Cohere Rerank、Jina Rerank、BGE-Reranker（本地）
- 调用形态：rerank(query=..., documents=[cand1, cand2, ...], top_n=5) → 返回重排后的 index + relevance_score
- 模型机制是 cross-encoder：把 query 和每条 candidate 拼在一起过一遍 transformer，比向量检索（bi-encoder 各算各的）精度高很多，但慢，所以只跑在"第一轮已经捞出的 top50 候选"上
- query → 向量库/搜索API 捞 top50（快，保召回） → Rerank 精排成 top5（准） → 喂 LLM

### RAG机制
- 将文本进行对齐（embedding），就是把“人话”和“文档”翻译成同一种“数学语言”，让它们能在同一个向量空间里比距离。
- Embedding 模型（如bge-m3）是怎么“学会对齐”的？
这是关键——embedding 模型是“被训练过”的，专门为了对齐查询和文档。
训练时它在学什么？
用一个简化版的对比学习（Contrastive Learning）例子：
训练样本：
  查询：怎么退钱
  正样本：退款流程说明
  负样本：猫为什么爱睡觉
训练目标：
让“怎么退钱”的向量 和 “退款流程说明”的向量 靠近
让“怎么退钱”的向量 和 “猫为什么爱睡觉”的向量 远离
经过海量这样的三元组训练后，模型就学会了：
“退钱” ≈ “退款” ≈ “退货” ≈ “返款”
这就是对齐——把不同表述但同义的东西，压到向量空间里相近的位置。
- 计算向量相似度，返回top50~top100，（只靠相似度不够准确）
- 为什么查询和文档要用“同一个”embedding 模型？
这是很多人忽略的点：
查询和文档必须用同一个 embedding 模型，否则向量空间不同，距离没意义。
类比：
你用中文翻译把“hello”翻成“你好”
你用日语翻译把“world”翻成“世界”
然后问：中文“你好”和日语“世界”距离近不近？
没法比，因为翻译规则不同，空间不同
所以：
文档入库时：用 bge-m3 向量化
用户查询时：也必须用同一个 bge-m3​ 向量化
这样两边才在同一个空间里，距离才有意义
- 为什么需要重排序？问题在于：压缩过程中丢掉了词序、重点、细粒度匹配信号。
- 重排序不用"压缩成向量"，而是：
把 (查询, 文档) 当成一对，直接过一遍 Cross-Encoder，输出一个"相关分数"
- 计算向量相似度得到的 top50进行重排序，返回top10，取top3进行输出

### 多 Agent 协作解决的四个核心问题

1. **关注点分离 → 解决"角色冲突"：**
一个 Agent 既要当研究员又要当审稿人，等于让同一个人又写又审，自我批评力度天然弱。拆成 Researcher + Critic，对抗式博弈能把事实错误率显著压低（典型如 debate / self-refine 架构）。

2. **上下文隔离 → 解决"长链路污染"：**
每个子 Agent 只拿自己那段的上下文，主 Agent 只汇总结论。百万级长任务里，这比把所有东西塞进一个窗口更稳也更省 token。

3. **并行 + 专业化 → 解决"吞吐与深度不可兼得"：**
并行：多个 Worker 同时查不同数据源，时延从串行 O(n) 降到 O(1)；
专业化：每个 Agent 配专属工具集和 system prompt（法律 Agent / 代码 Agent / 数据 Agent），比"通才 prompt"效果好得多。

4. **容错与可控 → 解决"一步错全盘崩"：**
Orchestrator 可以重试单个 Worker、做投票/仲裁、设超时降级。单 Agent 一旦在某步跑偏，只能靠反思硬拉回来。

- **经验法则**：能用单 Agent + 好工具解决的事，就不要拆。只有当任务同时满足 ① 可自然拆分、② 子任务可并行/专业化、③ 对准确率或时延敏感​ 时，多 Agent 的收益才明显大于成本。

- ### 中间件
- wrap_model_call 是干嘛的？每个中间件都必须有吗？
- wrap_model_call 是同步版本的中间件钩子。当你的 agent 在同步上下文中调用模型时走这个方法；awrap_model_call 是异步版本的钩子，agent 在 async 上下文中走这个。
  - LangChain 的 AgentMiddleware 基类定义了这些方法作为可选钩子：
  
  | 钩子 |用途|
  |-----|:----------------:|
  |wrap_model_call|同步模型调用拦截|
  |awrap_model_call |异步模型调用拦截|
  |wrap_tool_call|同步工具调用拦截|
  |awrap_tool_call|异步工具调用拦截|
  - 不是每个都必须有，只用异步就只写 async 版本没问题。
  
### agent人格
Pawer Gateway
   │  每次发请求前，把这一堆拼进 system prompt：
   │    ├─ AGENTS.md（工作手册：规则、记忆用法、红线）
   │    ├─ SOUL.md（人格：方塘的性格、说话风格）  ← 主人看的"性格文档"
   │    ├─ IDENTITY.md（身份：名字、物种、设定）
   │    ├─ USER.md + MEMORY.md（记忆）
   │    └─ 相关记忆片段（Active Memory 检索出来的）
   ▼
DeepSeek V4 API ← 纯黑盒，只收 prompt 出 text
更优雅：做成"人格中间件"（每次请求动态注入） 如果你想让 system prompt 跟着每次请求走（而不是 agent 构建时固定死），可以用中间件在 before_model 里注入：

### 图节点并行问题（如工具）
1. 大多数 RAG 场景串行就够了： 
- 原因 1：RAG 的瓶颈不在工具调用延迟
- 向量检索：~50ms（Chroma，本地）
- LLM 生成：~2-5s（取决于输出长度）
- 联网搜索：~1-3s
- 真正慢的是 LLM 生成，不是工具调用。并行省的那 1-2s 用户感知不明显。
- 原因 2：串行让 LLM 有"纠错机会"
- 并行：一次规划 3 个子查询 → 全部执行 → 发现第 2 个查错了 → 没法补救
- 串行：查第 1 个 → 看到结果不对 → 换个问法再查 → 更鲁棒
2. 两种并行策略： 
- 语义路由 → 拆子查询 → 并行检索（asyncio）→ 合并结果 → 一次 LLM 生成
- 注意：并行只放在"检索层"，不要并行 LLM 生成（没意义，生成必须串行）。
- （复杂 Agent）： LangGraph 里定义：某些节点并行执行 → 汇聚节点合并 → 下一节点
- 这是图计算框架的活，不是简单 asyncio 能 cover 的。

3. 可并行（无串行依赖）的操作：
- [联网搜索 ‖ RAG 检索]
- [路由计算 ‖ Prompt 渲染]
4. 两种并行实现方式
- 方式 1：Agent 框架自动并行（LangChain AgentExecutor 支持）
- 如果 LLM 一次输出多个工具调用（OpenAI 格式支持 parallel tool calls），AgentExecutor 可以并行执行：
  - 方式 2：应用层手动并行（你控制更细）
  - 
        import asyncio
        async def parallel_retrieve(queries):
            tasks = [asyncio.to_thread(retriever.invoke, q) for q in queries]
            results = await asyncio.gather(*tasks)
            return results
        
        #用户问题拆成子查询
        sub_queries = ["部署流程", "回滚方案"]
        docs = asyncio.run(parallel_retrieve(sub_queries))
        #合并后一次性喂给 LLM 生成
5. 优化策略（按 ROI 排序）：
- 联网 + RAG 并行（省 50ms，心理安慰为主）
- （已做）流式输出：LLM 一边生成一边返回给前端，用户感知延迟从 5s → 1s（体感最大提升！）
- 预检索：用户打字时就发起检索
- 缓存：相同 query 的检索结果 + LLM 回答缓存
6. 工业级 Agent 的做法——"路由后扇出，不互相依赖的工具分支并行，汇聚后再进 LLM"。

7. 关于关键路径：
> Plan-and-Execute     先规划再执行，规划用小模型、执行用大模型
> LLMCompiler    LLM 输出"任务图"，编译器自动并行无依赖任务
- 规划的价值：把能并行的都甩到非关键路径上。
- 简单问题（"部署流程是什么"→ 一次 RAG 就够）也走全套规划，反而更慢更贵。
- 对策：Fast Path 短路
>       query → 先过一个极简分类器
>       ├─ 简单（单工具）→ 直接执行，跳过 Planner
>       └─ 复杂（多子查询/多源）→ 才进 Planner
- 静态规划 vs 动态修正
Planner 一开始定的图，执行中发现"T1 结果不够，需要补搜"怎么办？
- 对策：允许 Plan 中途追加节点（动态 DAG）。LLMCompiler 的做法是：执行过程中若某任务输出触发"需要更多信息"，往图里插入新节点，重新拓扑调度。这是进阶功能，初期可以先做"失败重试"就够了。

8. 未来并行扇出的正确架构
>       用户 query
>           │
>           ▼
>       [tool_router]  ──→  scores / selected / tools
>           │
>           ▼  （编排层：确定性扇出）
>       fanout(query, selected_tools):
>           tasks = [call_tool(t, build_arg(query)) for t in selected_tools]
>           results = await asyncio.gather(*tasks)   # ← 真正的并行在这里
>           return merge(results)
>           │
>           ▼
>       汇聚结果 → 喂回模型生成最终回答

### Markdown 切分思路
一个可直接用的 Markdown 切分思路（经验值）
text
1. 先按 # / ## / ### 拆
2. 每个 section：
   - < 200 token → 不切
   - 200–600 → 一个 chunk
   - > 600 → 按段落切
3. overlap 只在“长段落切分”时用
4. 列表 / 表格 / 定义块：不切
5. 总结：Markdown 不要先想 chunk_size，先想结构；碎片化文档不要硬拼，小 chunk + 父子结构是王道。

---

## 费用
- 模型调用：deepseek-v4-flash（Flash 命中 0.02 元/M，未命中 1 元/M）
- 网络搜索：博查 ¥0.036 / 次（即 ¥36 / 千次），可购买资源包
- 网络搜索：tavily