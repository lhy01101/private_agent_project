# 工具定义，tools需要全部传入注册
from self_packages.web_search import web_tool # 优先导入
from langchain.agents.middleware import wrap_tool_call
from langchain_core.messages import ToolMessage
from langchain.tools import tool, ToolRuntime
from dataclasses import dataclass

@wrap_tool_call
def handle_tool_errors(request, handler):
    """使用自定义消息处理工具执行错误。"""
    try:
        return handler(request)
    except Exception as e:
        # 向模型返回自定义错误消息
        return ToolMessage(
            content=f"工具错误：请检查您的输入并重试。({str(e)})",
            tool_call_id=request.tool_call["id"]
        )


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
    # 可能的错误chat_box里面有config包含了user_id,但是新的web_chat_box里面没有:
    # invoke 传入的 context 可能是 dict，ToolRuntime 不会自动转成 Context
    ctx = runtime.context
    user_id = ctx["user_id"] if isinstance(ctx, dict) else ctx.user_id
    return "Florida" if user_id == "1" else "SF"

from datetime import datetime
@tool
def get_system_timezone():
    """返回系统时间。"""
    return datetime.now()

# rag_tool
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.vectorstores import Chroma   # 或 FAISS
from langchain_openai import OpenAIEmbeddings         # 本地可用 Ollama nomic-embed-text 替代
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="nomic-embed-text") # embedding模型
# test_embedding = embeddings.embed_query("测试")
# print(len(test_embedding))   # nomic-embed-text 是 768 维
vector_store = Chroma(persist_directory="./chroma_rag", embedding_function=embeddings)
# 第一步：底层 retriever 拿 50 个
base_retriever = vector_store.as_retriever(search_kwargs={"k": 50})   # 粗排：向量库检索 Top-K 文档

# 第二步：重排序器（cross-encoder 比 embedding 的 cosine 准得多）
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from dotenv import load_dotenv
load_dotenv()

ce = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")  # 重排序模型
compressor = CrossEncoderReranker(
    model=ce,   # 中文效果好，本地跑
    top_n=3,
)
# 第三步：组合
retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=base_retriever,
)

rag_tool_1 = create_retriever_tool(
    retriever, "search_knowledge_base",
    "检索本地项目文档与笔记，回答私有知识问题前必须调用。"
)

rag_tool_2 = create_retriever_tool(
    retriever, "search_knowledge_base",
    "检索本地项目文档与笔记，回答私有知识问题前必须调用。"
)

tools = [get_weather_for_location, rag_tool_1, rag_tool_2, web_tool, get_system_timezone]

