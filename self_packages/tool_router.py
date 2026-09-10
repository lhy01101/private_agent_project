import numpy as np
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()

EMBED_MODEL = "nomic-embed-text"

# 正例权重
W_EXAMPLE = 0.7
W_DESC = 0.3


# 1. 定义路由：正例(examples) + 负例(negative_examples) + 互斥规则 + 工具组合
from self_packages.routes_archive import ROUTES



class SemanticToolRouter:
    """
    带负面提示词的语义工具路由器。

    打分逻辑（每个路由）：
        pos_score = W_EXAMPLE * example_sim + W_DESC * desc_sim   # 正例 + 描述
        neg_sim   = 与负例矩阵的最大余弦相似度（无负例则为 0）
        net_score = pos_score - neg_alpha * neg_sim              # 净分（受负例抑制）

    入选条件（同时满足）：
        1. net_score >= threshold            # 净分够高
        2. neg_sim  <  neg_threshold         # 未触碰"硬排除"红线（可选，设 None 关闭）
    """

    def __init__(
        self,
        routes=None,
        threshold=0.55,
        neg_alpha=0.5,          # 负例抑制强度：越大，负例越能拉低分数
        neg_threshold=None,     # 硬排除红线：neg_sim 超过此值直接淘汰（None=关闭）
        embed_model=EMBED_MODEL,
    ):
        self.routes = routes or ROUTES
        self.threshold = threshold
        self.neg_alpha = neg_alpha
        self.neg_threshold = neg_threshold
        self.embedder = OllamaEmbeddings(model=embed_model)
        self._build_index()

    # ---------- 索引构建 ----------
    def _embed_norm(self, text: str) -> np.ndarray:
        """单句 embed 并 L2 归一化"""
        vec = np.array(self.embedder.embed_query(text), dtype=np.float32)
        return vec / (np.linalg.norm(vec) + 1e-9)

    def _build_index(self):
        """启动时把所有路由的正例 / 负例 / 描述离线 embed 成矩阵（只跑一次）"""
        self.route_names = []
        self.exemplar_vecs = []     # 正例：(n_examples, dim)
        self.negative_vecs = []     # 负例：(n_neg, dim)，无负例则为 None
        self.descriptions = {}      # 描述：(dim,)
        self.tool_map = {}
        self.exclusive_map = {}     # 互斥规则

        for name, cfg in self.routes.items():
            # 正例
            vecs = self.embedder.embed_documents(cfg["examples"])
            mat = np.array(vecs, dtype=np.float32)
            mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-9)

            # 负例（可选）
            neg_ex = cfg.get("negative_examples") or []
            neg_mat = None
            if neg_ex:
                neg_vecs = self.embedder.embed_documents(neg_ex)
                neg_mat = np.array(neg_vecs, dtype=np.float32)
                neg_mat = neg_mat / (np.linalg.norm(neg_mat, axis=1, keepdims=True) + 1e-9)

            # 描述
            desc_vec = self._embed_norm(cfg["description"])

            self.route_names.append(name)
            self.exemplar_vecs.append(mat)
            self.negative_vecs.append(neg_mat)
            self.descriptions[name] = desc_vec
            self.tool_map[name] = cfg["tools"]
            self.exclusive_map[name] = set(cfg.get("exclusive_with", []))

    # ---------- 路由推理 ----------
    def route(self, query: str) -> dict:
        q = self._embed_norm(query)

        scores = {}        # 正例+描述 的原始分
        neg_scores = {}    # 负例相似度
        net_scores = {}    # 净分（抑制后）

        for i, name in enumerate(self.route_names):
            # 正例最高分
            example_sim = float(np.dot(self.exemplar_vecs[i], q).max())
            # 描述分
            desc_sim = float(np.dot(self.descriptions[name], q))
            pos_score = W_EXAMPLE * example_sim + W_DESC * desc_sim

            # 负例最高分
            neg_mat = self.negative_vecs[i]
            neg_sim = float(np.dot(neg_mat, q).max()) if neg_mat is not None else 0.0

            net = pos_score - self.neg_alpha * neg_sim

            scores[name] = pos_score
            neg_scores[name] = neg_sim
            net_scores[name] = net

        # 初选：净分 >= 阈值
        candidates = {k: v for k, v in net_scores.items() if v >= self.threshold}

        # 硬排除：触碰负例红线的一律剔除
        if self.neg_threshold is not None:
            candidates = {
                k: v for k, v in candidates.items()
                if neg_scores[k] < self.neg_threshold
            }

        # 互斥消解：若 A 与 B 互斥且同时入选，保留净分更高的
        selected = dict(candidates)
        for name in list(selected.keys()):
            for rival in self.exclusive_map.get(name, ()):
                if rival in selected and name in self.exclusive_map.get(rival, ()):
                    if net_scores[name] < net_scores[rival]:
                        selected.pop(name, None)
                        break

        # 合并工具、去重、保持优先级顺序
        tools = []
        for name in self.route_names:
            if name in selected:
                for t in self.tool_map[name]:
                    if t not in tools:
                        tools.append(t)

        return {
            "scores": scores,          # 原始正分
            "neg_scores": neg_scores,  # 负例相似度
            "net_scores": net_scores,  # 抑制后的净分
            "selected": selected,
            "tools": tools,
        }


from self_packages.tools import rag_tool_1, rag_tool_2, web_tool, get_system_timezone

# 2. 工具注册表：名字 -> 真实 tool 对象
TOOL_REGISTRY = {
    "rag_tool_1": rag_tool_1,
    "rag_tool_2": rag_tool_2,
    "web_search": web_tool,
    "get_system_timezone": get_system_timezone,
}

# neg_threshold=0.72 ：一旦 query 与某路由负例的相似度超过 0.72，强制排除（可按校准结果调整）
router = SemanticToolRouter(threshold=0.55, neg_alpha=0.3, neg_threshold=None)


def select_tools(query: str) -> list:
    """返回本次调用应该暴露给 agent 的 tool 对象列表"""
    res = router.route(query)
    print(f"[router] scores    ={ {k: round(v, 3) for k, v in res['scores'].items()} }")
    print(f"[router] neg_scores={ {k: round(v, 3) for k, v in res['neg_scores'].items()} }")
    print(f"[router] net_scores={ {k: round(v, 3) for k, v in res['net_scores'].items()} }")
    print(f"[router] selected  ={list(res['selected'].keys())} -> {res['tools']}")
    return [TOOL_REGISTRY[t] for t in res["tools"]]
