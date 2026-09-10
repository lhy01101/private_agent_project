# -*- coding: utf-8 -*-
"""
calibrate.py —— 工具语义路由校准脚本（agg = max）

功能：
  1. 加载 routes_archive.py 中的最新 ROUTES 配置
  2. 对每个测试用例跑一遍路由，打印「期望 vs 实际」+ 四类分数
  3. 统计：各路由召回率、误触发率、纯 LLM（不应调工具）的正确拒绝率
  4. 自动网格扫描 (neg_alpha, neg_threshold, example_agg) 的最优组合

用法：
  uv run python calibrate.py            # 默认 max 聚合，跑测试集 + 自动扫描
  uv run python calibrate.py --no-scan   # 只跑测试集，不扫描
  uv run python calibrate.py --agg top3  # 对比 top-3 均值聚合

前置：ollama serve 已启动，且已拉取 nomic-embed-text
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
from langchain_ollama import OllamaEmbeddings

# ============================================================
# 0. 加载 routes_archive.py 里的 ROUTES（不依赖项目包结构）
# ============================================================
ROUTES_FILE = Path(__file__).parent / ".." / "inputs" / "routes_archive.py"
ROUTES_FILE = ROUTES_FILE.resolve()
if not ROUTES_FILE.exists():
    # 兜底：也允许放在本目录下
    ROUTES_FILE = Path(__file__).parent / "routes_archive.py"

spec = importlib.util.spec_from_file_location("routes_archive", str(ROUTES_FILE))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
ROUTES = mod.ROUTES

# ============================================================
# 1. 校准参数（可按需调整）
# ============================================================
EMBED_MODEL = "nomic-embed-text"
THRESHOLD = 0.55        # 净分入选阈值
NEG_ALPHA = 0.5         # 负例抑制强度
NEG_THRESHOLD = 0.72    # 硬排除红线（None=关闭）
EXAMPLE_AGG = "max"     # 正例聚合：max / mean / topk
TOPK = 3                # agg=topk 时取前 K 个求平均


# ============================================================
# 2. 测试集
#    - TEST_CASES: 期望命中的路由 -> 该路由的正例 query
#    - SHOULD_BE_NONE: 纯 LLM 问题，期望「不调任何工具」
#    - EXPECT_NOT: 针对难负例——这些 query 绝不能被指定路由选中
# ============================================================
TEST_CASES = {
    "rag_1": [
        "项目某段时间的日志是什么？",
        "8月17日那天做了什么？",
        "项目的 API 接口是怎么定义的？",
        "代码规范里对命名有什么要求？",
        "语义路由是怎么实现的？",
        "tool_router 的 threshold 是怎么校准的？",
        "RAG 里 rerank 用的什么模型？",
        "build_index 怎么支持增量更新？",
        "人格是怎么拼进 prompt 的？",
        "chunksize 改了为什么会检索出错？",
        "ToolRoutingMiddleware 是干什么的？",
        "博查和 Tavily 有什么区别？",
        "这个项目一个月大概花多少钱？",
    ],
    "rag_2": [
        "原神有什么角色？",
        "星铁里银枝的配装怎么搭？",
        "原神深境螺旋 12 层怎么打？",
        "这个 boss 的弱点是什么？",
        "原神 5.x 版本更新了什么？",
        "下一个卡池是谁？",
        "雷电将军和纳西妲哪个值得抽？",
    ],
    "web": [
        "帮我搜索一下",
        "帮我调查这件事",
        "今天有什么新闻",
        "DeepSeek V4 最新版本是多少",
        "现在美元兑人民币汇率是多少",
        "昨天 NBA 比赛结果如何",
        "帮我核实一下这个数据是真的吗",
        "北京到上海的高铁票多少钱",
        "what happened today in the news",
    ],
    "system_timezone": [
        "现在几点了",
        "今天几月几号",
        "系统时区是哪个",
        "今天星期几",
        "距今还有一个星期是几号",
        "what time is it now",
    ],
    "rag1_plus_web": [
        "我们文档里的方案和现在主流做法有什么区别？",
        "这个项目用的框架最新版本更新了什么？",
        "内部方案相比业界最新进展落后在哪？",
        "文档里这个组件的推荐配置，现在还是最佳实践吗？",
        "别人家是怎么做的，和我们比优劣如何？",
    ],
}

SHOULD_BE_NONE = [
    "帮我写一段 Python 代码",        # 纯 LLM，无需工具
    "Python 怎么读取文件",           # 通用知识
    "Flask 的路由装饰器怎么用",      # 通用知识
]

# 难负例断言：query -> 绝不能被这些路由选中
EXPECT_NOT = {
    "今天有什么新闻": ["system_timezone", "rag_1"],
    "现在美元兑人民币汇率是多少": ["system_timezone", "rag_1"],
    "今天发布的版本更新了什么": ["system_timezone"],
    "现在最火的游戏是什么": ["system_timezone"],
    "明天天气怎么样": ["system_timezone", "rag_1"],
    "帮我搜索一下原神的攻略": ["rag_2"],          # 「搜索」显式意图 → web，不应走 rag_2
    "Python 怎么读取文件": ["rag_1"],             # 通用知识，不属本项目
    "Flask 的路由装饰器怎么用": ["rag_1"],
    "帮我写一段 Python 代码": ["web"],            # 纯 LLM，连 web 都不该走
    "会议定在今天的几点": ["system_timezone"],     # 日程/时间解析，非「此刻」
}


# ============================================================
# 3. 路由器（自包含实现，与 tool_router.py 逻辑一致，可切换聚合）
# ============================================================
class CalibRouter:
    """仅用于校准：复刻 SemanticToolRouter 的打分，便于切换 example_agg"""

    def __init__(self, routes, embed_model=EMBED_MODEL, threshold=THRESHOLD,
                 neg_alpha=NEG_ALPHA, neg_threshold=NEG_THRESHOLD,
                 agg=EXAMPLE_AGG, topk=TOPK):
        self.routes = routes
        self.threshold = threshold
        self.neg_alpha = neg_alpha
        self.neg_threshold = neg_threshold
        self.agg = agg
        self.topk = topk
        self.embedder = OllamaEmbeddings(model=embed_model)
        self._build_index()

    def _norm(self, vec):
        v = np.array(vec, dtype=np.float32)
        return v / (np.linalg.norm(v) + 1e-9)

    def _agg(self, sims: np.ndarray) -> float:
        if sims.size == 0:
            return 0.0
        if self.agg == "max":
            return float(sims.max())
        if self.agg == "mean":
            return float(sims.mean())
        if self.agg == "topk":
            k = min(self.topk, sims.size)
            return float(np.sort(sims)[-k:].mean())
        raise ValueError(f"unknown agg: {self.agg}")

    def _build_index(self):
        self.route_names = []
        self.exemplar_vecs = []
        self.negative_vecs = []
        self.descriptions = {}
        self.exclusive_map = {}
        for name, cfg in self.routes.items():
            ex = cfg.get("examples", [])
            mat = np.stack([self._norm(self.embedder.embed_query(s)) for s in ex])
            neg_ex = cfg.get("negative_examples", []) or []
            neg_mat = None
            if neg_ex:
                neg_mat = np.stack([self._norm(self.embedder.embed_query(s)) for s in neg_ex])
            desc = self._norm(self.embedder.embed_query(cfg["description"]))
            self.route_names.append(name)
            self.exemplar_vecs.append(mat)
            self.negative_vecs.append(neg_mat)
            self.descriptions[name] = desc
            self.exclusive_map[name] = set(cfg.get("exclusive_with", []))

    def route(self, query: str) -> dict:
        q = self._norm(self.embedder.embed_query(query))
        scores, neg_scores, net_scores = {}, {}, {}
        for i, name in enumerate(self.route_names):
            example_sim = self._agg(np.dot(self.exemplar_vecs[i], q))
            desc_sim = float(np.dot(self.descriptions[name], q))
            pos_score = 0.7 * example_sim + 0.3 * desc_sim
            neg_mat = self.negative_vecs[i]
            neg_sim = self._agg(np.dot(neg_mat, q)) if neg_mat is not None else 0.0
            net = pos_score - self.neg_alpha * neg_sim
            scores[name] = pos_score
            neg_scores[name] = neg_sim
            net_scores[name] = net

        candidates = {k: v for k, v in net_scores.items() if v >= self.threshold}
        if self.neg_threshold is not None:
            candidates = {k: v for k, v in candidates.items()
                          if neg_scores[k] < self.neg_threshold}
        selected = dict(candidates)
        # 互斥消解
        for name in list(selected.keys()):
            for rival in self.exclusive_map.get(name, ()):
                if rival in selected and name in self.exclusive_map.get(rival, ()):
                    if net_scores[name] < net_scores[rival]:
                        selected.pop(name, None)
                        break

        return {
            "scores": scores,
            "neg_scores": neg_scores,
            "net_scores": net_scores,
            "selected": selected,
        }


# ============================================================
# 4. 评估逻辑
# ============================================================
def collect_queries():
    """把所有测试 query 摊平，并记录其「期望命中的路由集合」"""
    cases = []
    for label, qs in TEST_CASES.items():
        for q in qs:
            cases.append((q, {label}))   # 期望至少命中 label
    for q in SHOULD_BE_NONE:
        cases.append((q, set()))         # 期望为空
    return cases


# ============================================================
# 5. 网格扫描：自动寻找最优 (neg_alpha, neg_threshold, agg)
# ============================================================
def grid_search(base_routes):
    alphas = [0.3, 0.5, 0.7, 1.0]
    thresholds = [None, 0.60, 0.65, 0.70, 0.72, 0.75]
    aggs = ["max", "mean", "topk"]

    print("\n" + "#" * 96)
    print("开始网格扫描（首次会 embed 所有示例，稍慢）...\n")
    results = []
    for agg in aggs:
        for alpha in alphas:
            for thr in thresholds:
                router = CalibRouter(base_routes, agg=agg, neg_alpha=alpha,
                                     neg_threshold=thr, topk=TOPK)
                s = evaluate(router, verbose=False)   # 静默跑，明细存入 _LAST_METRICS
                mr = _LAST_METRICS.get("mean_recall", 0.0)
                nr = _LAST_METRICS.get("none_recall", 0.0)
                ft = _LAST_METRICS.get("false_triggers", 0)
                thr_str = f"{thr:.2f}" if thr is not None else "off"
                print(f"  agg={agg:<5} alpha={alpha:.1f} neg_thr={thr_str:<5} "
                      f"mean_recall={mr:.1%} none_recall={nr:.1%} "
                      f"false_trig={ft:<3} score={s:.3f}")
                results.append((s, agg, alpha, thr))

    # 按综合得分排序
    results.sort(key=lambda x: x[0], reverse=True)
    print("\n" + "=" * 96)
    print("Top-5 参数组合：")
    for rank, (s, agg, alpha, thr) in enumerate(results[:5], 1):
        thr_str = f"{thr:.2f}" if thr is not None else "off"
        print(f"  #{rank} score={s:.3f}  agg={agg:<5} neg_alpha={alpha:.1f}  neg_threshold={thr_str}")
    best = results[0]
    print(f"\n★ 推荐：agg={best[1]}, neg_alpha={best[2]:.1f}, neg_threshold="
          f"{f'{best[3]:.2f}' if best[3] is not None else 'off'}")
    return best


# evaluate 会把明细存入这里，供 grid_search 读取
_LAST_METRICS = {}


def evaluate(router: CalibRouter, verbose: bool = True):
    """重载版本：同时把明细存进 _LAST_METRICS 供 grid_search 使用"""
    cases = collect_queries()
    per_route_hits = {lbl: [0, 0] for lbl in TEST_CASES}
    false_triggers = 0
    none_total = len(SHOULD_BE_NONE)
    none_correct = 0

    if verbose:
        print("\n" + "=" * 96)
        print(f"{'query':<34} {'expected':<18} {'actual':<26} {'OK':<4} neg~max")
        print("-" * 96)

    for q, expected in cases:
        res = router.route(q)
        selected = set(res["selected"].keys())

        if len(expected) == 0:
            ok = (len(selected) == 0)
            if ok:
                none_correct += 1
            tag = "OK " if ok else "MISS"
        else:
            label = next(iter(expected))
            per_route_hits[label][1] += 1
            ok = label in selected
            if ok:
                per_route_hits[label][0] += 1
            tag = "OK " if ok else "MISS"

        for fb in EXPECT_NOT.get(q, []):
            if fb in selected:
                false_triggers += 1

        if verbose:
            exp_str = ",".join(sorted(expected)) if expected else "(none)"
            act_str = ",".join(sorted(selected)) if selected else "(direct)"
            top_neg = max(res["neg_scores"].items(), key=lambda kv: kv[1])
            print(f"{q:<32} {exp_str:<18} {act_str:<26} {tag:<4} {top_neg[0]}={top_neg[1]:.3f}")

    recalls = {lbl: (hit / total if total else 0.0)
               for lbl, (hit, total) in per_route_hits.items()}
    mean_recall = np.mean(list(recalls.values()))
    none_recall = none_correct / none_total if none_total else 0.0
    score = mean_recall * 0.5 + none_recall * 0.3 - false_triggers * 0.1

    if verbose:
        print("\n" + "=" * 96)
        print("【各路由召回率】")
        for label, rec in recalls.items():
            print(f"  {label:<18} {per_route_hits[label][0]}/{per_route_hits[label][1]} = {rec:.1%}")
        print(f"【正确拒绝率】 {none_correct}/{none_total} = {none_recall:.1%}")
        print(f"【误触发次数】 {false_triggers}")
        print(f"综合得分 = {score:.3f}")

    _LAST_METRICS["mean_recall"] = mean_recall
    _LAST_METRICS["none_recall"] = none_recall
    _LAST_METRICS["false_triggers"] = false_triggers
    return score


# ============================================================
# 6. 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="工具语义路由校准脚本")
    parser.add_argument("--no-scan", action="store_true", help="只跑测试集，不网格扫描")
    parser.add_argument("--agg", default=EXAMPLE_AGG, choices=["max", "mean", "topk"],
                        help="正例聚合方式（默认 max）")
    args = parser.parse_args()

    print(f"加载路由配置：{ROUTES_FILE}")
    print(f"路由数：{len(ROUTES)}  (agg={args.agg})\n")

    if args.no_scan:
        router = CalibRouter(ROUTES, agg=args.agg, neg_alpha=NEG_ALPHA,
                             neg_threshold=NEG_THRESHOLD, topk=TOPK)
        evaluate(router, verbose=True)
    else:
        grid_search(ROUTES)


if __name__ == "__main__":
    main()
