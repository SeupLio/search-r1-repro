# -*- coding: utf-8 -*-
"""统一出图。

数据来源优先级（自动切换，无需改代码）：
  1. results/runs/*.csv      —— 真机训练导出的真实曲线（有就用它）
  2. results/reconstructed_paper_curves/*.csv —— 依据论文 figure 重建的示意曲线
  3. analysis/paper_numbers.py —— 论文表格数字（金标准，永远用它做 reference）

所有含重建数据的图都会打 "RECONSTRUCTED FROM PAPER" 水印，避免误读。
"""
from __future__ import annotations

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from analysis import paper_numbers as PN   # noqa: E402

plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.labelcolor": "#111111",
    "text.color": "#111111",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "font.size": 10,
})

FIGDIR = os.path.join(ROOT, "results", "figures")
CFIGDIR = os.path.join(ROOT, "results", "reconstructed_paper_curves")
RUNSDIR = os.path.join(ROOT, "results", "runs")

# 浅色主题配色（中国习惯：涨红跌绿；这里用于"提升/下降"语义）
C_MAIN = "#2F6FB5"
C_ACC = "#D9534F"
C_OK = "#C0392B"       # 提升（红）
C_BAD = "#2E8B57"      # 下降（绿）
C_GREY = "#8C8C8C"
PALETTE = ["#2F6FB5", "#D9534F", "#5CB85C", "#F0AD4E", "#7B68A6", "#17A2B8", "#C0392B"]


def _ensure():
    os.makedirs(FIGDIR, exist_ok=True)
    os.makedirs(CFIGDIR, exist_ok=True)


def _stamp(ax_prov=True, text="RECONSTRUCTED FROM PAPER — 非实验数据"):
    if ax_prov:
        plt.gcf().text(0.99, 0.005, text, ha="right", va="bottom",
                       fontsize=7.5, color="#B00020", alpha=0.85)


def _load_curve(name: str):
    """优先真实 run，其次重建曲线。返回 (data_dict, is_reconstructed)。"""
    real = os.path.join(RUNSDIR, f"{name}.csv")
    rec = os.path.join(CFIGDIR, f"{name}.csv")
    path, is_rec = (real, False) if os.path.exists(real) else (rec, True)
    if not os.path.exists(path):
        return None, True
    import csv as _csv
    with open(path, encoding="utf-8-sig") as f:
        rows = list(_csv.DictReader(f))
    cols = [c for c in rows[0].keys() if c not in ("step", "provenance")]
    data = {"step": [int(float(r["step"])) for r in rows]}
    for c in cols:
        data[c] = [float(r[c]) for r in rows]
    return data, is_rec


def _smooth(y, w=9):
    """边界安全的滑动平均。

    ⚠️ 不能直接用 np.convolve(y, ones(w)/w, mode="same")：
    两端会用零填充，导致曲线在首尾被"拉低"，看起来像突然崩了。
    这里用除以有效点数的方式做归一化（等价于 edge padding）。
    """
    y = np.asarray(y, dtype=float)
    if w <= 1 or len(y) < w:
        return y
    k = np.ones(w)
    num = np.convolve(y, k, mode="same")
    den = np.convolve(np.ones_like(y), k, mode="same")
    return num / den


# ---------------------------------------------------------------------------
# 图 1：论文 Table 2 主结果（7B），reference
# ---------------------------------------------------------------------------
def fig1_paper_main():
    methods = ["RAG", "SFT", "R1-base", "Rejection Sampling", "Search-R1-base", "Search-R1-instruct"]
    keys = [f"Qwen2.5-7b / {m}" for m in methods]
    x = np.arange(len(PN.DATASETS))
    w = 0.13
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for i, k in enumerate(keys):
        vals = [PN.TABLE2[k][d] for d in PN.DATASETS]
        ax.bar(x + (i - len(keys) / 2 + 0.5) * w, vals, w,
               label=k.replace("Qwen2.5-7b / ", ""), color=PALETTE[i % len(PALETTE)],
               edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(PN.DATASETS)
    ax.set_ylabel("Exact Match")
    ax.set_title("论文 Table 2 主结果（Qwen2.5-7B）— 复现时的对照基准", fontsize=12)
    ax.legend(fontsize=8, ncol=3, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "f1_paper_table2_7b.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 图 2：PPO vs GRPO
# ---------------------------------------------------------------------------
def fig2_ppo_grpo():
    data, is_rec = _load_curve("ppo_vs_grpo")
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    if data:
        for name, color in (("PPO", C_MAIN), ("GRPO", C_ACC)):
            if name in data:
                ax.plot(data["step"], _smooth(data[name]), color=color, lw=2, label=name)
                ax.fill_between(data["step"], _smooth(data[name], 21) - 0.012,
                                _smooth(data[name], 21) + 0.012, color=color, alpha=0.10)
    # 论文 Table 3 的终值锚点（7b-base）
    ax.axhline(PN.TABLE3["Qwen2.5-7b / Search-R1-base (PPO)"]["Avg"], ls="--", lw=1, color=C_MAIN, alpha=.6)
    ax.axhline(PN.TABLE3["Qwen2.5-7b / Search-R1-base (GRPO)"]["Avg"], ls="--", lw=1, color=C_ACC, alpha=.6)
    ax.text(505, PN.TABLE3["Qwen2.5-7b / Search-R1-base (PPO)"]["Avg"], " PPO avg .431",
            va="center", fontsize=8, color=C_MAIN)
    ax.text(505, PN.TABLE3["Qwen2.5-7b / Search-R1-base (GRPO)"]["Avg"], " GRPO avg .350",
            va="center", fontsize=8, color=C_ACC)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Train Reward")
    ax.set_title("PPO vs GRPO：GRPO 收敛更快但后期易 collapse，PPO 更稳", fontsize=12)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    if is_rec:
        _stamp()
    p = os.path.join(FIGDIR, "f2_ppo_vs_grpo.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 图 3：消融三连（loss mask / topk / group size）
# ---------------------------------------------------------------------------
def fig3_ablations():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    specs = [
        ("loss_mask_7b", ["w. mask", "w.o. mask"], "Retrieved Token Loss Mask (7b)",
         {"w. mask": .431, "w.o. mask": .343}),
        ("topk", ["topk=1", "topk=3", "topk=5"], "Top-k Retrieved Passages",
         {"topk=1": .375, "topk=3": .431, "topk=5": .400}),
        ("group_size", ["size=1", "size=3", "size=5"], "GRPO Group Size",
         {"size=1": .410, "size=3": .363, "size=5": .350}),
    ]
    any_rec = False
    for ax, (name, series, title, finals) in zip(axes, specs):
        data, is_rec = _load_curve(name)
        any_rec = any_rec or is_rec
        if data:
            for i, s in enumerate(series):
                if s in data:
                    ax.plot(data["step"], _smooth(data[s]), lw=2,
                            color=PALETTE[i], label=f"{s} (avg {finals[s]:.3f})")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Train Reward")
        ax.legend(fontsize=8, frameon=False)
        ax.grid(alpha=0.25)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    if any_rec:
        _stamp()
    p = os.path.join(FIGDIR, "f3_ablations.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 图 4：response length / #valid search / base-vs-instruct 动态
# ---------------------------------------------------------------------------
def fig4_dynamics():
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    any_rec = False

    # (a) response length
    d, r = _load_curve("response_length"); any_rec |= r
    ax = axes[0]
    if d and "Response Length" in d:
        ax.plot(d["step"], _smooth(d["Response Length"], 7), color=C_MAIN, lw=2, label="Response Length")
        ax.set_ylabel("Response Length (tokens)")
        ax2 = ax.twinx()
        ax2.plot(d["step"], _smooth(d["Train Reward"], 7), color=C_ACC, lw=1.6, ls="--", alpha=.85,
                 label="Train Reward")
        ax2.set_ylabel("Train Reward", color=C_ACC)
        ax2.tick_params(axis="y", colors=C_ACC)
    ax.set_title("(a) 长度：下降→上升→稳定", fontsize=11)
    ax.set_xlabel("Step")
    ax.grid(alpha=0.25)

    # (b) valid search
    d, r = _load_curve("valid_search"); any_rec |= r
    ax = axes[1]
    if d and "# Valid Search" in d:
        ax.plot(d["step"], _smooth(d["# Valid Search"], 7), color=C_ACC, lw=2, label="# Valid Search")
        ax.set_ylabel("# Valid Search")
        ax2 = ax.twinx()
        ax2.plot(d["step"], _smooth(d["Train Reward"], 7), color=C_MAIN, lw=1.6, ls="--", alpha=.85)
        ax2.set_ylabel("Train Reward", color=C_MAIN)
        ax2.tick_params(axis="y", colors=C_MAIN)
    ax.set_title("(b) 检索次数随训练增加", fontsize=11)
    ax.set_xlabel("Step")
    ax.grid(alpha=0.25)

    # (c) base vs instruct
    d, r = _load_curve("base_vs_instruct"); any_rec |= r
    ax = axes[2]
    if d:
        for s, c in (("Base", C_MAIN), ("Instruct", C_ACC)):
            if s in d:
                ax.plot(d["step"], _smooth(d[s]), color=c, lw=2, label=s)
    ax.set_title("(c) Base vs Instruct：最终趋同", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Train Reward")
    ax.legend(fontsize=9, frameon=False)
    ax.grid(alpha=0.25)

    for ax in axes:
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    fig.tight_layout()
    if any_rec:
        _stamp()
    p = os.path.join(FIGDIR, "f4_training_dynamics.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 图 5-9：内容平台自建评测集（真实 dry-run 结果）
# ---------------------------------------------------------------------------
def _load_summary():
    p = os.path.join(ROOT, "results", "dryrun", "summary.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def fig5_content_overall(summary):
    if not summary:
        return None
    names = list(summary.keys())
    vals = [summary[n]["EM"] for n in names]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    colors = [C_GREY, "#F0AD4E", C_MAIN, C_OK]
    bars = ax.bar(names, vals, color=[colors[i % len(colors)] for i in range(len(names))],
                  edgecolor="white", width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", fontsize=10)
    base = summary.get("RAG", {}).get("EM", 0)
    if base:
        ax.axhline(base, ls="--", lw=1, color=C_GREY)
        ax.text(len(names) - 0.4, base + 0.012, f"RAG 基线 {base:.3f}", fontsize=8, color=C_GREY)
    ax.set_ylabel("Exact Match (EM)")
    ax.set_ylim(0, max(vals) * 1.22 + 0.05)
    ax.set_title("自建内容平台评测集（ContentSearch-Bench, 212 题）总体 EM", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "f5_content_overall.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig6_content_by_difficulty(summary):
    if not summary:
        return None
    diffs = ["easy", "medium", "hard"]
    names = list(summary.keys())
    x = np.arange(len(diffs))
    w = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for i, n in enumerate(names):
        vals = [summary[n]["by_difficulty"].get(d, 0) for d in diffs]
        ax.bar(x + (i - len(names) / 2 + 0.5) * w, vals, w, label=n,
               color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.6)
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - len(names) / 2 + 0.5) * w, v + 0.012, f"{v:.2f}",
                    ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(["简单 (单跳)", "中等 (多跳/聚合)", "困难 (跨平台/时效/拒答)"])
    ax.set_ylabel("EM")
    ax.set_ylim(0, 1.16)
    ax.set_title("分难度 EM：检索带来的增益集中在中高难度", fontsize=12)
    ax.legend(fontsize=9, ncol=4, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "f6_content_by_difficulty.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


TASK_CN = {
    "single_hop_author": "单跳·作者",
    "single_hop_numeric": "单跳·数值",
    "single_hop_date": "单跳·日期",
    "aggregate_max": "聚合·最大值",
    "aggregate_avg": "聚合·平均",
    "aggregate_count": "聚合·计数",
    "multi_hop_compare": "多跳·比较",
    "cross_platform_compare": "跨平台·对比",
    "unanswerable": "不可答·拒答",
}


def fig7_content_by_task(summary):
    if not summary:
        return None
    tasks = [t for t in TASK_CN if t in summary[list(summary)[0]]["by_task"]]
    names = list(summary.keys())
    x = np.arange(len(tasks))
    w = 0.8 / max(len(names), 1)
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, n in enumerate(names):
        vals = [summary[n]["by_task"].get(t, 0) for t in tasks]
        ax.bar(x + (i - len(names) / 2 + 0.5) * w, vals, w, label=n,
               color=PALETTE[i % len(PALETTE)], edgecolor="white", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_CN[t] for t in tasks], rotation=20, ha="right")
    ax.set_ylabel("EM")
    ax.set_ylim(0, 1.16)
    ax.set_title("分题型 EM：单跳已饱和，瓶颈在聚合推理与拒答", fontsize=12)
    ax.legend(fontsize=9, ncol=4, frameon=False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "f7_content_by_task.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


FAIL_CN = {
    "correct": "正确",
    "format_error": "格式错误",
    "no_search": "未检索",
    "over_search": "检索预算耗尽",
    "retrieval_miss": "检索未召回",
    "query_poor": "query 质量差",
    "unfaithful": "不忠实/幻觉",
    "aggregation_error": "聚合推理错误",
    "gold_strict": "评测口径偏严",
}
FAIL_COLOR = {
    "correct": "#5CB85C", "format_error": "#7B68A6", "no_search": "#F0AD4E",
    "over_search": "#17A2B8", "retrieval_miss": "#C0392B", "query_poor": "#E08A00",
    "unfaithful": "#B00020", "aggregation_error": "#2F6FB5", "gold_strict": "#8C8C8C",
}


def fig8_failure_attribution(summary):
    if not summary:
        return None
    names = [n for n in summary if n != "Direct"]
    keys = [k for k in FAIL_CN if any(summary[n]["failure_counts"].get(k, 0) for n in names)]
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    bottom = np.zeros(len(names))
    for k in keys:
        vals = np.array([summary[n]["failure_counts"].get(k, 0) for n in names], dtype=float)
        totals = np.array([sum(summary[n]["failure_counts"].values()) for n in names], dtype=float)
        frac = vals / np.maximum(totals, 1)
        ax.bar(names, frac, bottom=bottom, label=FAIL_CN[k], color=FAIL_COLOR[k],
               edgecolor="white", linewidth=0.5)
        for i, (f, v) in enumerate(zip(frac, vals)):
            if f > 0.03:
                ax.text(i, bottom[i] + f / 2, f"{int(v)}", ha="center", va="center",
                        fontsize=8, color="white")
        bottom += frac
    ax.set_ylabel("样本占比")
    ax.set_ylim(0, 1.0)
    ax.set_title("失败案例归因：错在哪里决定了下一步该调什么", fontsize=12)
    ax.legend(fontsize=8, ncol=5, frameon=False, bbox_to_anchor=(1.0, -0.12))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "f8_failure_attribution.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def fig9_search_efficiency(summary):
    if not summary:
        return None
    names = list(summary.keys())
    xs = [summary[n]["avg_valid_search"] for n in names]
    ys = [summary[n]["EM"] for n in names]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    for i, n in enumerate(names):
        ax.scatter(xs[i], ys[i], s=140, color=PALETTE[i % len(PALETTE)], zorder=3,
                   edgecolor="white", linewidth=1.2)
        ax.annotate(f"{n}\n{ys[i]:.3f}", (xs[i], ys[i]), textcoords="offset points",
                    xytext=(10, -4), fontsize=9)
    ax.set_xlabel("平均有效检索次数 / 题（B=4 预算）")
    ax.set_ylabel("EM")
    ax.set_title("检索次数 × 准确率：多检索本身不是目的，能召回+能过滤才是", fontsize=12)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    p = os.path.join(FIGDIR, "f9_search_efficiency.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 图 10：偏差分析
# ---------------------------------------------------------------------------
def fig10_deviation():
    from analysis import deviation as DV
    rows = sorted(DV.FACTORS, key=lambda f: f.delta_low)
    labels = [f.name for f in rows]
    lows = [f.delta_low for f in rows]
    highs = [f.delta_high for f in rows]
    y = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    for i, f in enumerate(rows):
        color = C_BAD if f.delta_high <= 0 else (C_MAIN if f.delta_low >= 0 else "#F0AD4E")
        ax.barh(y[i], highs[i] - lows[i], left=lows[i], height=0.55,
                color=color, alpha=0.85, edgecolor="white")
        ax.text(highs[i] + 0.25, y[i], f"[{f.delta_low:+.1f}, {f.delta_high:+.1f}]",
                va="center", fontsize=8.5, color="#333")
    ax.axvline(0, color="#333", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("对平均 EM 的影响（百分点）")
    ax.set_title("偏差来源分解：复现值 vs 论文值（工程量级估计，非实测）", fontsize=12)
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    plt.gcf().text(0.99, 0.005, "ESTIMATED IMPACT — 非实测，依据见 analysis/deviation.py",
                   ha="right", va="bottom", fontsize=7.5, color="#B00020", alpha=0.85)
    p = os.path.join(FIGDIR, "f10_deviation.png")
    fig.savefig(p, dpi=170)
    plt.close(fig)
    return p


def main():
    _ensure()
    from analysis import paper_curves as PC
    if not os.path.exists(os.path.join(CFIGDIR, "ppo_vs_grpo.csv")):
        PC.dump_csv(CFIGDIR)
    PN.dump_all_csv(os.path.join(ROOT, "results", "paper_reference"))

    summary = _load_summary()
    made = []
    made.append(fig1_paper_main())
    made.append(fig2_ppo_grpo())
    made.append(fig3_ablations())
    made.append(fig4_dynamics())
    for fn in (fig5_content_overall, fig6_content_by_difficulty, fig7_content_by_task,
               fig8_failure_attribution, fig9_search_efficiency):
        p = fn(summary)
        if p:
            made.append(p)
    made.append(fig10_deviation())
    for p in made:
        print("figure:", os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
