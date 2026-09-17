# -*- coding: utf-8 -*-
"""把真实训练曲线导出成 analysis/make_figures.py 认识的 CSV。

支持两种来源：
  1. wandb（推荐，verl 默认就接）
  2. tensorboard event 文件

导出的 CSV 放到 results/runs/<name>.csv，make_figures.py 会自动优先使用它，
从而把图 2/3/4 从"论文重建曲线"切换成"你的真实曲线"——不需要改任何绘图代码。

命名约定（必须与 analysis/make_figures.py 的 _load_curve 对得上）：
    ppo_vs_grpo.csv      列: step, PPO, GRPO
    base_vs_instruct.csv 列: step, Base, Instruct
    loss_mask_7b.csv     列: step, w. mask, w.o. mask
    loss_mask_3b.csv     列: step, w. mask, w.o. mask
    topk.csv             列: step, topk=1, topk=3, topk=5
    group_size.csv       列: step, size=1, size=3, size=5
    response_length.csv  列: step, Response Length, Train Reward
    valid_search.csv     列: step, # Valid Search, Train Reward

用法：
    python scripts/07_export_run_logs.py --source wandb --entity me --project search_r1_repro
    python scripts/07_export_run_logs.py --source tb --logdir logs/ --out results/runs
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List

# wandb 里的 run name -> 目标 csv 名
RUN_TO_FILE = {
    "ppo_vs_grpo": "ppo_vs_grpo",
    "base_vs_instruct": "base_vs_instruct",
    "loss_mask_7b": "loss_mask_7b",
    "loss_mask_3b": "loss_mask_3b",
    "topk": "topk",
    "group_size": "group_size",
    "response_length": "response_length",
    "valid_search": "valid_search",
}


def export_wandb(entity: str, project: str, outdir: str) -> List[str]:
    import wandb  # 延迟导入，没装也不影响其他功能
    api = wandb.Api()
    runs = api.runs(f"{entity}/{project}")
    # 按 run 前缀分组（如 "ppo_vs_grpo/PPO" -> 组 ppo_vs_grpo，列名 PPO）
    groups: Dict[str, Dict[str, Dict[int, float]]] = {}
    for r in runs:
        name = r.name or r.id
        if "/" not in name:
            print(f"[skip] run '{name}' 不符合 '<group>/<series>' 命名，跳过")
            continue
        g, series = name.split("/", 1)
        if g not in RUN_TO_FILE:
            print(f"[skip] 未知分组 '{g}'")
            continue
        hist = r.history(keys=["_step", "train/reward", "critic/rewards/mean",
                               "response_length/mean", "search_r1/valid_search"],
                         pandas=True)
        if hist is None or hist.empty:
            continue
        col = "train/reward"
        if col not in hist.columns:
            col = "critic/rewards/mean" if "critic/rewards/mean" in hist.columns else None
        if col is None:
            continue
        groups.setdefault(g, {})[series] = dict(zip(hist["_step"], hist[col]))

    written = []
    os.makedirs(outdir, exist_ok=True)
    for g, series in groups.items():
        steps = sorted({s for v in series.values() for s in v})
        path = os.path.join(outdir, f"{RUN_TO_FILE[g]}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["step"] + list(series.keys()) + ["provenance"])
            for s in steps:
                w.writerow([s] + [series[k].get(s, "") for k in series] + ["real_run"])
        written.append(path)
        print("wrote", path)
    return written


def export_tb(logdir: str, outdir: str, tags: Dict[str, str]) -> List[str]:
    from tensorboard.backend.event_processing import event_accumulator
    os.makedirs(outdir, exist_ok=True)
    rows: Dict[str, Dict[str, Dict[int, float]]] = {}
    for root, _dirs, files in os.walk(logdir):
        for fn in files:
            if not fn.startswith("events.out"):
                continue
            ea = event_accumulator.EventAccumulator(os.path.join(root, fn))
            ea.Reload()
            for group, tag in tags.items():
                if tag not in ea.Tags().get("scalars", []):
                    continue
                series_name = os.path.basename(root)
                d = rows.setdefault(group, {}).setdefault(series_name, {})
                for e in ea.Scalars(tag):
                    d[e.step] = e.value
    written = []
    for g, series in rows.items():
        steps = sorted({s for v in series.values() for s in v})
        path = os.path.join(outdir, f"{RUN_TO_FILE.get(g, g)}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["step"] + list(series.keys()) + ["provenance"])
            for s in steps:
                w.writerow([s] + [series[k].get(s, "") for k in series] + ["real_run"])
        written.append(path)
    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["wandb", "tb"], default="wandb")
    ap.add_argument("--entity", default=None)
    ap.add_argument("--project", default="search_r1_repro")
    ap.add_argument("--logdir", default="logs")
    ap.add_argument("--out", default="results/runs")
    args = ap.parse_args()

    if args.source == "wandb":
        if not args.entity:
            sys.exit("--entity 必填")
        files = export_wandb(args.entity, args.project, args.out)
    else:
        files = export_tb(args.logdir, args.out, {
            "ppo_vs_grpo": "train/reward",
            "response_length": "response_length/mean",
            "valid_search": "search_r1/valid_search",
        })
    print(f"\n导出 {len(files)} 个文件。现在运行 python -m analysis.make_figures 即可用真实曲线出图。")


if __name__ == "__main__":
    main()
