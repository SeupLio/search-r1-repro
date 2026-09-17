# -*- coding: utf-8 -*-
"""论文训练曲线的**重建**（reconstruction），不是实验数据。

⚠️ 用途与边界（务必如实说明，不能当成自己的实验结果）
------------------------------------------------------------------
本模块依据 arXiv:2503.09516v5 的 Figure 2/3/5/6/7 的
  * 坐标轴范围（从 PDF 文本层读出，如 "0 100 200 300 400 500 Step" / "0.0 0.1 ... 0.4"）
  * 正文与附录对曲线形态的文字描述（如 "GRPO 收敛更快但多步后 reward collapse"、
    "response length 呈 下降-上升-稳定 趋势"、 "topk=5 前 200 步最快但后续不稳"）
  * 以及 Table 2/3/4/6/7/8 给出的确定终值
生成形状与量级一致的示意曲线。

它唯一的用途是：
  1. 在没有 GPU 的情况下验证"论文曲线 -> 对比图"的绘图管线；
  2. 在复现时作为"应当看到的形状"的参考基线（shape prior），
     真机跑完把 wandb/tensorboard 导出的真实 CSV 覆盖到 results/runs/ 即可自动替换。

所有输出都带 provenance 字段 = "reconstructed_from_paper"，
出图时会在图上打水印，绝不会与真实实验数据混淆。
"""
from __future__ import annotations

import csv
import math
import os
import random
from typing import Dict, List

PROVENANCE = "reconstructed_from_paper"


def _noise(rng: random.Random, scale: float) -> float:
    return rng.gauss(0, scale)


def _smooth_noise(rng: random.Random, n: int, scale: float) -> List[float]:
    """一阶平滑的噪声，比白噪声更像真实 tensorboard 曲线。"""
    out, prev = [], 0.0
    for _ in range(n):
        prev = 0.75 * prev + 0.25 * rng.gauss(0, scale)
        out.append(prev)
    return out


def _logistic(start: float, end: float, k: float, x0: float, x: int) -> float:
    return start + (end - start) / (1 + math.exp(-k * (x - x0)))


def _series(xs: List[int], fn, rng, noise_scale, lo, hi) -> List[float]:
    vals = []
    for x in xs:
        v = fn(x) + _noise(rng, noise_scale)
        vals.append(max(lo, min(hi, v)))
    return vals


# ---------------------------------------------------------------------------
# Figure 2(a) / 5: PPO vs GRPO  (axis: step 0-500, reward 0.0-0.4)
# 描述：GRPO 收敛更快；PPO 更稳；GRPO 后期 collapse；最终两者相当。
# ---------------------------------------------------------------------------
def ppo_vs_grpo(steps: int = 500, seed: int = 1) -> Dict[str, List[float]]:
    rng = random.Random(seed)
    xs = list(range(steps + 1))

    def ppo(x):
        # critic 需要 warm-up，起步慢；之后稳定爬升到 ~0.43
        return _logistic(0.03, 0.43, 0.011, 210, x) + 0.012 * math.sin(x / 40.0)

    def grpo(x):
        # 快速爬到 ~0.40；约 300 步后 collapse 到 ~0.28
        base = _logistic(0.03, 0.41, 0.030, 70, x)
        collapse = 0.13 / (1 + math.exp(-0.035 * (x - 330)))
        return base - collapse

    return {
        "step": xs,
        "PPO": _series(xs, ppo, rng, 0.010, 0.0, 0.45),
        "GRPO": _series(xs, grpo, rng, 0.014, 0.0, 0.45),
    }


# ---------------------------------------------------------------------------
# Figure 2(b) / 4: Base vs Instruct (axis: step 0-200, reward 0.05-0.40)
# 描述：instruct 起点更高、收敛更快；最终两者非常接近。
# ---------------------------------------------------------------------------
def base_vs_instruct(steps: int = 200, seed: int = 2) -> Dict[str, List[float]]:
    rng = random.Random(seed)
    xs = list(range(steps + 1))

    def base(x):
        return _logistic(0.06, 0.385, 0.020, 95, x)

    def instr(x):
        return _logistic(0.14, 0.395, 0.038, 45, x)

    return {
        "step": xs,
        "Base": _series(xs, base, rng, 0.010, 0.02, 0.45),
        "Instruct": _series(xs, instr, rng, 0.010, 0.02, 0.45),
    }


# ---------------------------------------------------------------------------
# Figure 2(c): response length (900-1150) + reward (0.1-0.5), step 0-200
# 描述：前 100 步长度骤降、reward 略升；之后长度与 reward 同步显著上升。
# ---------------------------------------------------------------------------
def response_length(steps: int = 200, seed: int = 3) -> Dict[str, List[float]]:
    rng = random.Random(seed)
    xs = list(range(steps + 1))

    def length(x):
        drop = 210 / (1 + math.exp(-0.08 * (x - 35)))          # 早期骤降
        rise = 430 / (1 + math.exp(-0.030 * (x - 150)))        # 后期上升
        return 1100 - drop + rise

    def reward(x):
        return 0.14 + 0.10 / (1 + math.exp(-0.05 * (x - 40))) + 0.24 / (1 + math.exp(-0.028 * (x - 150)))

    return {
        "step": xs,
        "Response Length": [length(x) + _noise(rng, 8) for x in xs],
        "Train Reward": _series(xs, reward, rng, 0.008, 0.05, 0.55),
    }


# ---------------------------------------------------------------------------
# Figure 2(d): # valid search (1.4-2.0) + reward, step 0-200
# 描述：随训练推进，模型学会更多地调用搜索。
# ---------------------------------------------------------------------------
def valid_search(steps: int = 200, seed: int = 4) -> Dict[str, List[float]]:
    rng = random.Random(seed)
    xs = list(range(steps + 1))

    def nsearch(x):
        return 1.42 + 0.56 / (1 + math.exp(-0.026 * (x - 105)))

    return {
        "step": xs,
        "# Valid Search": [nsearch(x) + _noise(rng, 0.015) for x in xs],
        "Train Reward": _series(xs, lambda x: 0.13 + 0.34 / (1 + math.exp(-0.024 * (x - 110))),
                                rng, 0.008, 0.05, 0.55),
    }


# ---------------------------------------------------------------------------
# Figure 3 / Table 4: retrieved token loss mask
# 3b: step 0-400, reward 0.0-0.4 ; 7b: step 0-200, reward 0.10-0.50
# 描述：w. mask 始终优于 w.o. mask，且更稳定。
# ---------------------------------------------------------------------------
def loss_mask(model: str = "7b", steps: int = None, seed: int = 5) -> Dict[str, List[float]]:
    steps = steps or (200 if model == "7b" else 400)
    rng = random.Random(seed + (0 if model == "7b" else 10))
    xs = list(range(steps + 1))
    lo, hi = (0.08, 0.52) if model == "7b" else (0.0, 0.42)
    end_w, end_wo = (0.45, 0.34) if model == "7b" else (0.34, 0.26)
    x0 = 80 if model == "7b" else 170

    def with_mask(x):
        return _logistic(0.10 if model == "7b" else 0.02, end_w, 0.018, x0, x)

    def without_mask(x):
        v = _logistic(0.10 if model == "7b" else 0.02, end_wo, 0.022, x0 * 1.1, x)
        v += 0.02 * math.sin(x / 30.0)     # 更抖
        return v

    return {
        "step": xs,
        "w. mask": _series(xs, with_mask, rng, 0.008, lo, hi),
        "w.o. mask": _series(xs, without_mask, rng, 0.014, lo, hi),
    }


# ---------------------------------------------------------------------------
# Figure 6 / Table 7: top-k (step 0-500, reward 0.1-0.5)
# 描述：topk=5 前 200 步最快最高，之后下降且不稳；topk=1/3 稳步提升，topk=3 最终最高。
# ---------------------------------------------------------------------------
def topk(steps: int = 500, seed: int = 6) -> Dict[str, List[float]]:
    rng = random.Random(seed)
    xs = list(range(steps + 1))

    def f1(x):
        return _logistic(0.10, 0.375, 0.014, 210, x)

    def f3(x):
        return _logistic(0.10, 0.432, 0.013, 230, x)

    def f5(x):
        fast = _logistic(0.10, 0.46, 0.028, 90, x)
        decay = 0.075 / (1 + math.exp(-0.018 * (x - 260)))
        return fast - decay

    return {
        "step": xs,
        "topk=1": _series(xs, f1, rng, 0.009, 0.05, 0.55),
        "topk=3": _series(xs, f3, rng, 0.009, 0.05, 0.55),
        "topk=5": _series(xs, f5, rng, 0.014, 0.05, 0.55),
    }


# ---------------------------------------------------------------------------
# Figure 7 / Table 8: GRPO group size (step 0-500, reward 0.0-0.5)
# 描述：group size 越大收敛越快，但越容易 collapse；size=1 更稳、泛化更好。
# ---------------------------------------------------------------------------
def group_size(steps: int = 500, seed: int = 7) -> Dict[str, List[float]]:
    rng = random.Random(seed)
    xs = list(range(steps + 1))

    def s1(x):
        return _logistic(0.02, 0.405, 0.0095, 240, x)

    def s3(x):
        v = _logistic(0.02, 0.385, 0.017, 130, x)
        return v - 0.035 / (1 + math.exp(-0.020 * (x - 330)))

    def s5(x):
        v = _logistic(0.02, 0.375, 0.026, 85, x)
        return v - 0.055 / (1 + math.exp(-0.018 * (x - 300)))

    return {
        "step": xs,
        "size=1": _series(xs, s1, rng, 0.009, 0.0, 0.55),
        "size=3": _series(xs, s3, rng, 0.012, 0.0, 0.55),
        "size=5": _series(xs, s5, rng, 0.016, 0.0, 0.55),
    }


CURVES = {
    "ppo_vs_grpo": ppo_vs_grpo,
    "base_vs_instruct": base_vs_instruct,
    "response_length": response_length,
    "valid_search": valid_search,
    "loss_mask_7b": lambda: loss_mask("7b"),
    "loss_mask_3b": lambda: loss_mask("3b"),
    "topk": topk,
    "group_size": group_size,
}


def dump_csv(outdir: str) -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    written = []
    for name, fn in CURVES.items():
        data = fn()
        keys = [k for k in data if k != "step"]
        path = os.path.join(outdir, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["step"] + keys + ["provenance"])
            for i, s in enumerate(data["step"]):
                w.writerow([s] + [f"{data[k][i]:.6f}" for k in keys] + [PROVENANCE])
        written.append(path)
    return written


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for p in dump_csv(os.path.join(here, "..", "results", "reconstructed_paper_curves")):
        print("wrote", os.path.normpath(p))
