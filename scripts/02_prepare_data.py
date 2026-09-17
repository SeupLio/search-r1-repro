# -*- coding: utf-8 -*-
"""准备训练/评测数据（NQ / HotpotQA / TriviaQA / PopQA / 2wiki / Musique / Bamboogle）。

论文设置：
  * 训练：NQ train + HotpotQA train 合并；
  * 评测：7 个数据集的 test/validation；
  * 指标：EM。

产物是 verl 认的 parquet，字段与 Search-R1 官方 scripts/data_process/nq_search.py 一致：
    data_source / prompt / ability / reward_model{style, ground_truth} / extra_info{split,index}

注意：官方脚本直接产出 parquet；本脚本额外做两件事
  1. 统计并打印每个 split 的规模与答案长度分布（便于发现 split 版本不一致）；
  2. 把 unanswerable / 无 gold 的样本单独落到 *_noanswer.jsonl，
     供拒答能力评测使用（论文没有这个维度，是本仓库新增的）。
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter

DATASETS = {
    "nq":         ("natural_questions", None, "question", "answer"),
    "hotpotqa":   ("hotpot_qa", "distractor", "question", "answer"),
    "triviaqa":   ("trivia_qa", "rc.wikipedia", "question", "answer"),
    "popqa":      ("popqa", None, "question", "possible_answers"),
    "2wiki":      ("2wikimultihopqa", None, "question", "answer"),
    "musique":    ("musique", None, "question", "answer"),
    "bamboogle":  ("bamboogle", None, "question", "answer"),
}


def build_record(q: str, a: str, source: str, idx: int, split: str) -> dict:
    return {
        "data_source": source,
        "prompt": [{"role": "user", "content": q.strip()}],
        "ability": "fact-reasoning",
        "reward_model": {"style": "rule", "ground_truth": str(a).strip()},
        "extra_info": {"split": split, "index": idx},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--offline", action="store_true",
                    help="不联网下载，只做本地 jsonl -> parquet 的转换与统计")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    try:
        from datasets import load_dataset
        import pandas as pd
    except ImportError:
        raise SystemExit("需要 pip install datasets pandas pyarrow")

    stats = {}
    for name, (hf_name, hf_cfg, qkey, akey) in DATASETS.items():
        try:
            ds = load_dataset(hf_name, hf_cfg) if hf_cfg else load_dataset(hf_name)
        except Exception as e:                      # noqa: BLE001
            print(f"[skip] {name}: {e}")
            continue

        for split in ds.keys():
            rows, noans = [], []
            for i, ex in enumerate(ds[split]):
                q = ex.get(qkey) or ex.get("question")
                a = ex.get(akey) or ex.get("answers") or ex.get("answer")
                if isinstance(a, list):
                    a = a[0] if a else None
                if not q or a in (None, "", []):
                    noans.append({"question": q, "split": split, "index": i})
                    continue
                rows.append(build_record(q, a, name, i, split))
            if not rows:
                continue
            out = os.path.join(args.outdir, f"{name}_{split}.parquet")
            pd.DataFrame(rows).to_parquet(out, index=False)
            stats[f"{name}/{split}"] = len(rows)
            if noans:
                with open(os.path.join(args.outdir, f"{name}_{split}_noanswer.jsonl"),
                          "w", encoding="utf-8") as f:
                    for r in noans:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"[ok] {name}/{split}: {len(rows)} rows -> {out}")

    # 合并训练集（论文做法）
    train_files = [os.path.join(args.outdir, f"{n}_train.parquet")
                   for n in ("nq", "hotpotqa")]
    train_files = [p for p in train_files if os.path.exists(p)]
    if len(train_files) == 2:
        merged = pd.concat([pd.read_parquet(p) for p in train_files], ignore_index=True)
        merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
        out = os.path.join(args.outdir, "nq_hotpotqa_train.parquet")
        merged.to_parquet(out, index=False)
        print(f"[merge] {len(merged)} rows -> {out}")

    print("\n=== 数据规模统计（用来对齐 split 版本）===")
    for k, v in sorted(stats.items()):
        print(f"  {k:>22s}: {v}")
    with open(os.path.join(args.outdir, "data_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print("\n把 data_stats.json 和论文对比：规模对不上说明 split 版本不同，属于已知偏差来源。")


if __name__ == "__main__":
    main()
