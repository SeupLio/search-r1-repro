# -*- coding: utf-8 -*-
"""一键跑通全部可离线执行的产物（不需要 GPU）。

    python run_all.py

等价于依次执行：
    analysis.paper_numbers          论文数字 -> CSV
    content_platform.build_corpus   语料构建
    content_platform.build_evalset  自建评测集
    content_platform.run_content_eval --offline   检索/评测口径自检
    eval.run_dryrun                 四系统对比 + 归因
    eval.export_failure_cases       失败案例库
    analysis.make_figures           10 张图
    analysis.deviation              偏差预算

全部步骤都是确定性的（不依赖 PYTHONHASHSEED），重复运行结果一致。
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ("论文数字 -> CSV", [sys.executable, "-m", "analysis.paper_numbers"]),
    ("构建内容平台语料", [sys.executable, "-m", "content_platform.build_corpus",
                    "--out", "data/content_corpus.jsonl"]),
    ("构建自建评测集", [sys.executable, "-m", "content_platform.build_evalset",
                   "--out", "data/content_eval.jsonl"]),
    ("检索/评测口径自检", [sys.executable, "-m", "content_platform.run_content_eval",
                    "--offline", "--out", "results/eval/content_bench.json"]),
    ("四系统对比 + 归因", [sys.executable, "-m", "eval.run_dryrun"]),
    ("失败案例库", [sys.executable, "-m", "eval.export_failure_cases"]),
    ("生成图表", [sys.executable, "-m", "analysis.make_figures"]),
    ("偏差预算", [sys.executable, "-m", "analysis.deviation"]),
]


def main():
    t0 = time.time()
    failed = []
    for i, (name, cmd) in enumerate(STEPS, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(STEPS)}] {name}\n{'=' * 70}", flush=True)
        r = subprocess.run(cmd, cwd=HERE)
        if r.returncode != 0:
            failed.append(name)
            print(f"[FAIL] {name}", flush=True)
    print(f"\n{'=' * 70}")
    print(f"完成，用时 {time.time() - t0:.1f}s")
    if failed:
        print("失败步骤:", failed)
        sys.exit(1)
    print("全部通过。产物在 results/ 下：figures/ dryrun/ failure_attribution/")


if __name__ == "__main__":
    main()
