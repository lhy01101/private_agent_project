"""
test_code_fallback.py —— 代码切块双路径保障（验收 #6）

设计：TreeSitterLoader 在 import 失败时自动降级为 CodeRegexLoader。
本环境无 tree-sitter，验证：
  (a) 降级路径：正则切块仍产出 name + lineno（验收 #6 的最小契约）
  (b) 若环境装了 tree-sitter，自动启用 AST 切块（更精确）

无论哪条路径，metadata 都必须含 name + lineno —— 这是契约核心。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from file_ingest.loader import get_loader, TreeSitterLoader, CodeRegexLoader


def main():
    results = []
    def check(name, cond, detail=""):
        results.append(bool(cond))
        print(f"  {'✓' if cond else '✗'} {name}" + (f"  ({detail})" if detail else ""))

    with tempfile.TemporaryDirectory() as d:
        py = Path(d) / "module.py"
        py.write_text(
            '"""模块文档字符串"""\n'
            "# 顶层注释\n"
            "\n"
            "import os\n"
            "\n"
            "CONSTANT = 42\n"
            "\n"
            "def fetch_data(url):\n"
            '    """拉取数据"""\n'
            "    return url\n"
            "\n"
            "\n"
            "class DataProcessor:\n"
            '    """数据处理"""\n'
            "\n"
            "    def __init__(self, name):\n"
            "        self.name = name\n"
            "\n"
            "    def run(self):\n"
            "        return True\n"
            "\n"
            "\n"
            "def main():\n"
            "    p = DataProcessor('x')\n"
            "    print(p.run())\n"
        )

        print("[A] get_loader 分发（.py）")
        loader = get_loader(str(py))
        check("返回 Loader 实例", loader is not None)
        # 取决于环境：有 tree-sitter -> TreeSitterLoader，否则 CodeRegexLoader
        path_name = type(loader).__name__
        print(f"    实际使用的 loader: {path_name}")

        chunks = loader.load(str(py))
        names = [c.metadata.get("name") for c in chunks]

        print("\n[B] 契约：metadata 必须含 name + lineno（验收 #6 核心）")
        annotated = [c for c in chunks if c.metadata.get("name") and c.metadata.get("lineno")]
        check("至少一个块带 name + lineno", len(annotated) >= 3,
              f"names={names}")
        check("所有定义块都有 lineno",
              all("lineno" in c.metadata for c in annotated))
        # lineno 单调递增（定义顺序）
        linenos = [c.metadata["lineno"] for c in annotated]
        check("lineno 单调递增", all(linenos[i] < linenos[i+1] for i in range(len(linenos)-1)),
              str(linenos))

        # 关键符号都被捕获
        for sym in ("fetch_data", "DataProcessor", "main"):
            check(f"捕获符号 {sym}", sym in names, f"现有={names}")

        print("\n[C] TreeSitterLoader 降级行为")
        ts = TreeSitterLoader("python", ".py")
        check("降级标志 _available 存在", hasattr(ts, "_available"))
        if ts._available:
            print("    → 环境有 tree-sitter，走 AST 切块")
            ts_chunks = ts.load(str(py))
            check("AST 切块非空", bool(ts_chunks))
            check("AST 也带 name+lineno",
                  any(c.metadata.get("name") and c.metadata.get("lineno") for c in ts_chunks))
        else:
            print("    → 无 tree-sitter，已降级为 CodeRegexLoader")
            check("降级后仍是有效 Loader", isinstance(ts._regex_fallback, CodeRegexLoader))
            # 降级路径下 load 仍能工作
            fb = ts.load(str(py))
            check("降级路径 load 不报错且非空", bool(fb))

        print("\n[D] 其他语言降级（.go/.ts/.java）")
        for ext, lang in [(".go", "go"), (".ts", "typescript"), (".java", "java")]:
            f = Path(d) / f"a{ext}"
            f.write_text(f"package main\nfunc main() {{}}\nclass A {{}}")
            loader = get_loader(str(f))
            check(f"{ext} 有 loader（不抛异常）", loader is not None)
            # 即使 tree-sitter grammar 缺，也应降级到正则
            chunks2 = loader.load(str(f))
            check(f"{ext} load 产出带 lineno 的块", any("lineno" in c.metadata for c in chunks2))

    print("\n" + "=" * 50)
    print(f"总计: {len(results)} 项, 通过 {sum(results)}, 失败 {len(results)-sum(results)}")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
