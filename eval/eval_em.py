# -*- coding: utf-8 -*-
"""统一评测入口。

两条路径：
  1. 真机：加载 verl 训练出的 checkpoint，用 SGLang 起服务，跑 7 个 benchmark 或
     内容平台自建集，输出 EM + 每题的完整 trajectory（供失败归因）。
  2. 离线：直接对一份 "predictions.jsonl" 打分（predictions 由任意方式产生），
     用于快速对拍评测口径。

predictions.jsonl 每行：
    {"id": "...", "prediction": "...", "gold": ["..."], "gold_is_numeric": false}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searchr1_repro.reward import compute_reward, exact_match, normalize_answer  # noqa: E402


def eval_predictions(path: str) -> dict:
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    per_task = defaultdict(lambda: [0.0, 0])
    per_plat = defaultdict(lambda: [0.0, 0])
    total = 0.0
    per_item = []
    for r in rows:
        gold = r.get("gold") or []
        if isinstance(gold, str):
            gold = [gold]
        pred = r.get("prediction", "")
        s = compute_reward(pred, gold, r.get("gold_is_numeric", False))
        total += s
        key = r.get("task_type", "unknown")
        per_task[key][0] += s
        per_task[key][1] += 1
        plat = r.get("platform", "unknown")
        per_plat[plat][0] += s
        per_plat[plat][1] += 1
        per_item.append({**r, "reward": s, "norm_pred": normalize_answer(pred),
                         "norm_gold": [normalize_answer(g) for g in gold]})
    n = max(len(rows), 1)
    return {
        "n": len(rows),
        "EM": round(total / n, 4),
        "by_task": {k: round(v[0] / v[1], 4) for k, v in sorted(per_task.items())},
        "by_platform": {k: round(v[0] / v[1], 4) for k, v in sorted(per_plat.items())},
        "items": per_item,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--out", default="results/eval_report.json")
    ap.add_argument("--dump-wrong", default=None, help="把错题导出成 jsonl 便于归因")
    args = ap.parse_args()

    report = eval_predictions(args.predictions)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in report.items() if k != "items"}, f,
                  ensure_ascii=False, indent=2)

    print(f"samples = {report['n']}   EM = {report['EM']}")
    print("by task    :", report["by_task"])
    print("by platform:", report["by_platform"])

    if args.dump_wrong:
        wrong = [it for it in report["items"] if it["reward"] < 1.0]
        with open(args.dump_wrong, "w", encoding="utf-8") as f:
            for w in wrong:
                f.write(json.dumps(w, ensure_ascii=False) + "\n")
        print(f"错题 {len(wrong)} 条 -> {args.dump_wrong}")


if __name__ == "__main__":
    main()
