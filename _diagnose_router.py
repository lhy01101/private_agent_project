"""
工具路由诊断脚本（离线可用，绕开 self_packages.tools 的 HuggingFace cross-encoder）
================================================================================
用途：排查「该调工具却没调 / 不该调却调了 / 路由被门槛截断」这类问题。

分区：
  ① embedding 体检 —— 先确认向量能不能区分中文，不能的话后面所有阈值都无意义
  ② 逐路由拆解   —— 对每个 query 打印 pos / neg / net，并标出被哪道门拦下

用法：
    cd <项目根>
    ./.venv/Scripts/python.exe _diagnose_router.py

判读标准（①）：
    中文两两 cos 若出现 1.0000 且「向量完全相同=N」为 True，
    说明这些句子被映射成了同一个向量 → 语义路由在该长度上完全失效。
"""
import sys
import types

import numpy as np

RAG = r"C:\Users\11976.old\projects\RAG"
sys.path.insert(0, RAG)


class FakeTool:
    def __init__(self, n):
        self.name = n


tm = types.ModuleType("self_packages.tools")
tm.rag_tool_1 = FakeTool("search_knowledge_base")
tm.rag_tool_2 = FakeTool("search_knowledge_base")
tm.web_tool = FakeTool("web_search")
tm.get_system_timezone = FakeTool("get_system_timezone")
sys.modules["self_packages.tools"] = tm

import self_packages.tool_router as tr  # noqa: E402

router = tr.router


# ----------------------------------------------------------------- ① embedding 体检
def embed_sanity():
    groups = {
        "英文": ["what time is it", "hello there my", "apple banana pie"],
        "中文 3 字": ["你好呀", "几点了", "吃了吗"],
        "中文 5 字": ["哈哈哈笑死", "现在几点了", "苹果香蕉梨"],
        "中文 6 字": ["今天天气真好", "苹果香蕉橘子", "完全无关内容"],
    }
    for title, texts in groups.items():
        vecs = np.array(router.embedder.embed_documents(texts), dtype=np.float32)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
        c = vecs @ vecs.T
        n = len(texts)
        off = [c[i][j] for i in range(n) for j in range(i + 1, n)]
        dup = [(texts[i], texts[j])
               for i in range(n) for j in range(i + 1, n)
               if np.allclose(c[i][j], 1.0)]
        verdict = "塌缩!" if dup else "正常"
        print(f"  [{verdict}] {title:<8} 非对角线 cos: "
              f"min={min(off):.4f} max={max(off):.4f}  完全相同{len(dup)}对")
        for a, b in dup:
            print(f"           └ '{a}' 与 '{b}' 向量完全相同")


# ----------------------------------------------------------------- ② 逐路由拆解
QUERIES = [
    "现在几点了", "看一下时间", "今天几号了",          # 时间类
    "哈哈哈笑死", "你好呀", "1+1等于几",                # 闲聊类
    "今天有什么新闻", "明天天气怎么样", "原神有什么角色",   # 应走工具
    "项目的 API 接口是怎么定义的",
]


def explain():
    r = router
    print(f"\n门槛: threshold={r.threshold}  neg_alpha={r.neg_alpha}  "
          f"neg_threshold={r.neg_threshold}")
    for q in QUERIES:
        res = r.route(q)
        print(f"\n  --- {q}")
        for name in r.route_names:
            pos, neg = res["scores"][name], res["neg_scores"][name]
            net = res["net_scores"][name]
            why = []
            if net < r.threshold:
                why.append(f"net<{r.threshold}")
            if r.neg_threshold is not None and neg >= r.neg_threshold:
                why.append(f"neg>={r.neg_threshold}")
            print(f"      {name:<16} pos={pos:.3f} neg={neg:.3f} net={net:.3f}  "
                  f"{'PASS' if not why else 'DROP(' + ','.join(why) + ')'}")


if __name__ == "__main__":
    print("=== ① embedding 体检 ===")
    embed_sanity()
    print("\n=== ② 逐路由拆解 ===")
    explain()
