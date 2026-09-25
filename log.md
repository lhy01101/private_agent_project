# 日志文档

## **7月29日日志**
- **模型动态选择以及中间件**
  - 定义中间件参数让智能体自动选择模型 
  - middleware=[dynamic_model_selection]
  - > 中间件不能与预绑定模型（已调用 bind_tools 的模型）一起使用
  - > 此外，中间件参数与工具参数绑定使用
- 封装了输出模块chat_box
- 封装了动态选择模型模块DynamicModelMiddleware()

## **7月31日日志**
- 使用duckduckgo-search进行模型联网

## 8月1日日志
- 封装了tools

## 8月5日日志
- 使用agent搭建了股票信息查询的agent测试用例
- 包含两个模块stock_tools.py和stock_response.py
- 以后调权限只改 permission.py 的 USER_PERMISSIONS 一张表；接真实权限服务时把这张 dict 换成数据库查询即可，调用方零改动。                      # 无 permission → basic → 延迟15分钟

今日反馈：（已完成）需要换更好的模型，即调用接口，来优化输出；其次是优化网络搜索工具

## 8月17日日志
- 包了rag进入工具中
- 新建了build_index文本对齐脚本
- 输入模型的文件存储在./chroma_rag中
- 使用方式：
- - mkdir docs
- - 把你的 md/txt/pdf 丢进 docs/
- - cp ~/notes/*.md docs/
- - cp ~/project/*.pdf docs/
- - python build_index.py

## 8月19日日志
- 对README.md文档embedding
- 成功调用RAG

## 8月20日日志
- 新增了流式输入与流式输出，暂弃旧的self_packages.chat_box.py
- 使用gardio搭建了简易问答网页
- ### **语义路由**
- 语义路由在 RAG 里的典型用法（8G 显存很适合）

|     用户问题     |   路由到    |
|:------------:|:--------:|
| “怎么配置 nginx” | 技术文档知识库  |
|   “年假还剩几天”   | 人事制度知识库  |
|   “帮我写周报”    | 纯 LLM（不走 RAG）  |
|   “你是谁”           | 固定回答（不走 LLM）  |
语义路由 = 用“向量相似度”代替 if/elif 做请求分发

- ### embedding vs LLM embedding，为什么不一样？
- 它们“产出的东西”看起来一样，但训练目标完全不同

|类型| 训练目标|擅长什么|
|---|---|---|
|通用 embedding（bge / m3）|对比学习：让“相似句”靠近，“无关句”远离|✅ 检索、语义匹配|
|LLM embedding（hidden state）|预测下一个 token|❌ 检索弱，✅ 生成强|
- bge 是为了“找文档”被专门训练的
- LLM 是为了“说人话”被训练的


## 8月28日志
- 成功在web上运行

## 8月29日日志
- 成功调用deepseek-v4-flash API接口（API-key在site_packages 的.env文件中）
- 测试：上下文记忆出问题了，联网也有问题


## 8月30日日志
- 重新加入参数checkpointer=InMemorySaver(),成功添加短期记忆
- 去除tools列表中的get_user_location(会被误认为用户信息)
- 修改了头像，增加了图片压缩模块Picture_compression.py，修改了gradio页面的css格式
- 测试：联网功能在链接GGDD时可用，也就是说现在的源可能是外网


## 8月31日日志
- 成功实现国内网站搜索
- 添加了新的两个收费网络搜索提供商：博查（bocha）、tavily
- 添加了网络搜索语义路由
- ### 博查网络搜索
- 基于 IP 的本地化搜索 
  - 搜索引擎会通过你发起请求时的 IP 地址做粗略定位（通常精确到城市级别），然后自动把本地新闻排到前面。这是很常见的功能，目的是让结果更"接地气"。不过——这个定位是在搜索引擎服务端发生的，AI看不到你的 IP，也拿不到定位结果。
- ### 语义路由怎么写
- 所谓"语义路由"在这里分两层，不引入额外 embedding 模型，靠轻量启发式 + 配置态决定，避免每次搜索多花一次 LLM 调用：
- 配置层路由（硬路由）：看哪个 Key 存在。
  - 有 BOCHA_API_KEY → 博查可用
  - 有 TAVILY_API_KEY → Tavily 可用
  - DEV_DDG_FALLBACK=1 → DDG 才进候选
- 查询层路由（软/语义路由）：对 query 做轻量判断——
  - 含中日韩字符 → 优先博查（中文索引强）
  - 纯 ASCII 且像英文短语/技术名 → 博查失败再走 Tavily
  - 博查调用异常（429/超时/空结果）→ 自动跳下一个可用供应商
- 容错链：供应商按顺序 try，全部失败返回固定错误串，不让 LangGraph ToolNode 抛异常炸图。

## 9月5日日志
- 原RAG为提取向量库文档top3，现改为top50
- 在rag_tools中添加了cross_encoder重排序，重排序top3为新的rag查询结果
- 在.env添加了HF_TOKEN并设置了hf离线模式

## 9月6日日志
- 实现build_index.py增量更新
- - 扫描 docs/ 所有文件
- - → 对比 file_state.json
- - → 只处理「新文件」和「mtime/size 变了的文件」
- - → 加载 → 切片 → add_documents() 追加进已有 Chroma
- - → 更新 file_state.json
- chunksize更改大小不一致导致检索出错，语义近似度精度降低。若要更改切片大小可以rm -r /chroma_rag删除后再运行update_index.py

## 9月7日日志
- 添加工具语义路由（不是把用户提问分配到不同agent管线，而是把用户提问做一次embedding，根据语义距离选择工具输入）
- tool_router.py 工具管线，解决工具过多容易选错工具的问题
- calibrate.py 用于tool_router的threshold校准，threshold用于选出分数超过阈值的工具
- 添加中间件ToolRoutingMiddleware

## 9月8日日志
- langchain的中间件好像把agent剖开了，可以看看能玩出什么花样

## 9月9日日志
- 把人格拼进prompt里面,新增prompt.py

##  9月10日日志
- 添加工具timezone
- 添加了工具提示词的负例机制
- 大幅提高了工具调用速度和正确率（具体原因去看optimize.md 三、）
打分逻辑从单一"正分"变成：正分 − 负分抑制：
pos_score = 0.7 × example_sim + 0.3 × desc_sim     # 正例 + 描述（和原来一样）
neg_sim   = 与 negative_examples 的最大余弦相似度   # 新增：越像负例越高
net_score = pos_score − 0.5 × neg_sim              # 净分（被负例拉低）
- 将项目push到git上

## 9月13日日志
- 更改了self_packages/tool_router.py中select_tools()函数，加入赢家兜底机制
- DSML问题终于告一段落，本质上是tool_router中参数与模型意愿相差过大的问题
- （已解决）问题：兜底意味着"几乎任何 query 都会强制绑一个工具"
- 1. 兜底机制中加入底线 fallback_min_net=0.30，净分太低就返回空让模型直接回答，从此"哈哈哈笑死"不会再绑时间工具，同时保留对 "AI 新闻"(0.522) 这类边界 query 的兜底救援。
- 还需要调整参数：fallback_min_net和系列路由参数
- 2. 推荐方案：加一个 direct_answer 路由，在 routes_archive.py 里新增一个路由，正例 = 应该直接回答的 query，负例 = 任何需要工具的 query。
- 这样"直接回答"就有了自己的正分，可以和工具路由同台竞技

## 9月14日日志
- 调整了chunk_size与chunk_overlap，以适应碎片化文本
- markdown文本可以考虑结构化切分方式
- 测试结果良好，但是比较脆弱，依赖上下文

## 9月15日日志
- 了解了agent读取文件的机制（不仅仅要读取，并要进行管理）
- 生成了可直接交付工程师的file_achieve.md文件

## 9月16日日志
- 修了dynamic_select.py中同步流程的一个问题
- 安装了file_ingest的依赖项
- 查看元宝；我不确定项目是否有chroma_client，之前一直用chroma.sqlite3文件直接存RAG切片

## 9月17日日志
- 写了github README文档与LICENSE许可证

## 9月22日志
- 老大又回来了，软件不上线誓不做人，555
- 解决了direct_answer导致的DSML问题，不再将tools = []直接call传给模型；使用call(request.override(tool_choice="none"))告诉模型不使用工具，弃用direct_answer；
- select_tools只做工具裁剪，tool_choice 交回模型自己决定（None → auto）
- 还是偶发DSML问题，怀疑是neg_threshold误杀了正确的工具，将其设置为none后情况好很多
- tool_choice约束的是输出通道，不是模型意图；tool_choice = None模型自主决定|auto模型自主决定|"none"模型侧不走function-call输出通道；在allowed_names = set()即工具集为空时tool_choice是直接设为"none"
- "模型认为有工具可调" 与 "输出通道不可用" 同时成立
- 根因在于embedding对短中文的区分能力弱，1.换一个中文可靠的 embedding（治本），候选：bge-m3；2.需要继续优化routes_archive的精度；3.将提示词里："你可以使用以下工具"修改为动态的

## 9月23日日志
- 修改了prompt，把所有工具描述集中到工具的description中，不再在prompt声明工具。之后可以通过优化description去优化模型调用工具的意愿
- 更新了中间件设计
  1. 正常路由裁剪得到工具，tool_choice = None，模型手里只有这个工具，且让它决定要不要用这个工具
  2. 裁剪后若工具集为空，暴露所有工具集（把直接回答合并到路由失败那条路），tool_choice = None，让模型自己决定
  3. 之后工具路由的意义就在于缩短工具选择范围，不再有硬控制直接回答
- "全量"传工具是兜底，不是常态；常态应该是路由命中后裁剪，只有残差时才兜底。
- 新增_has_tool_activity：用于预防多轮粘性，「最后一条用户消息之后，是否出现过 ai.tool_calls 或 ToolMessage」。命中就跳过裁剪，不干预。


## 9月25日日志
- 更新了README.md描述


















---
# **方向调整**：
## 可选方向
- (1)先问问Claudecode意见，再添加direct_anser路由以避免工具过度调用
- (1)测试工具路由过程中名字错误的问题（如实际调用timezone但显示调用rag_2）
- (2)能够操作文件的工具：我想给我的智能体不仅能“读”，还能“写”，我想给他加上能写文档，写入各种文件的权限及能力
- (3)对话记忆：按日期分，把重点对话和主动记忆放入本地文档
- (3)记忆：用户角色、对话记忆（长期对话保依赖服务器运行吗？我需要实现记忆主动写入文档）
- （可选）将embedding换成bge-m3以更适配中英混合的场景
- (4)先搞一个简单的fastapi界面？然后再加上文件输入和多轮对话功能
- (4)用FastAPI做页面，异步处理
- 动态系统提示（根据用户输入选择提示词）
- 循环+规则判断，实现多步骤协作（AgentExecutor/langGraph）
- AI询问用户的能力
- 小灵感：模拟主动输出的算法：特定时段/情感需求期/聊天后置时段 的特定输入或者空白输入，输入时根据近期细节记忆，向用户主动发出聊天邀请


---
# 出现的问题
1. 调用工具时快速返回输出：（已经在一定程度上解决，提示词限制+中间件拦截（但是中间件貌似没起作用））
   (hi之后稳定触发DSML问题)
DSML 不是"模型抽风非要用的私有协议"，而是模型在"手里没有可用工具"时的一种退化行为——它想调工具但无处下手，就把调用写成正文文本。
兜底保证了工具集永远不会为空，模型因此始终走"结构化 function calling"，DSML 就从源头消失了。
但是：兜底意味着"几乎任何 query 都会强制绑一个工具"
<｜｜DSML｜｜ calls>
<｜｜DSML｜｜ invoke name="web_search">
<｜｜DSML｜｜ parameter name="query" string="true">iPhone 18 发布 价格 配置</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
<｜｜DSML｜｜ invoke name="web_search">
<｜｜DSML｜｜ parameter name="query" string="true">iPhone 18 release date specs price 2026</｜｜DSML｜｜ parameter>
</｜｜DSML｜｜ invoke>
</｜｜DSML｜｜ calls>