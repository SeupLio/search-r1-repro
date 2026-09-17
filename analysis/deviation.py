# -*- coding: utf-8 -*-
"""偏差分析（Deviation Analysis）：复现值 vs 论文值，差多少、为什么差。

这是复现报告里最有说服力的一节——它证明你不是"跑完就交"，
而是能解释每一分差距来自哪里。

每一条偏差因子包含：
    name         因子名
    category     环境 / 数据 / 算法 / 评测 / 随机性
    delta_low    对 7 数据集平均 EM 的影响下界（百分点，负=变差）
    delta_high   上界
    direction    确定方向 / 双向
    evidence     依据（论文附录 / 已知工程事实 / 本仓库实测）
    mitigation   消除办法

⚠️ 数值是**工程量级估计**，不是实测。依据写清楚，读者可自行判断是否认同。
   真机跑完后用 results/runs/ 的真实数字替换 estimated 列即可。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List


@dataclass
class DeviationFactor:
    name: str
    category: str
    delta_low: float      # 百分点
    delta_high: float
    direction: str        # "negative" | "positive" | "both"
    evidence: str
    mitigation: str


FACTORS: List[DeviationFactor] = [
    DeviationFactor(
        "检索器不一致（E5 → 其他/未对齐）", "环境",
        -8.0, -3.0, "negative",
        "论文 Appendix B.2 明确 retriever = E5、corpus = 2018 Wikipedia dump、top-k=3。"
        "换成 bge/m3e 或不同 dump 版本，召回集合直接改变；这类差异对 RL 是"
        "「环境改变」，不是「超参抖动」，影响量级远大于学习率微调。",
        "严格使用 intfloat/e5 + wiki-18 索引；至少跑一次 recall@k 对齐检查。"),

    DeviationFactor(
        "语料/索引版本（2018 dump, 21M passages）", "数据",
        -5.0, -2.0, "negative",
        "同一问题在不同 dump 上答案可验证性不同；FAISS flat vs ANN 也有召回差。"
        "本仓库 content_platform/ 用 45 条合成语料时，这一项会被放大到不可比。",
        "下载官方 e5_Flat.index + wiki-18.jsonl；ANN 索引需额外报 recall 差。"),

    DeviationFactor(
        "rollout 引擎：vLLM 0.6.3 → SGLang", "环境",
        -3.0, 3.0, "both",
        "论文用 vLLM 0.6.3 (tp=1, gpu_mem_util=0.6)。SGLang 的 RadixAttention "
        "与采样 kernel 不同，temperature=1.0 下的采样序列分布不完全一致，"
        "会造成同分布下的采样差异，并影响 KL/优势估计的数值。",
        "先跑一版 vLLM 基线对齐，再切 SGLang 做效率对比；两者都报告。"),

    DeviationFactor(
        "算力不足导致 global batch 变小 / 步数变少", "算法",
        -15.0, -5.0, "negative",
        "论文是 8×H100、total batch 512、mini-batch 256、micro-batch 64、500 steps。"
        "单卡/少卡常被迫把 batch 降到 64~128 并减少步数，直接放大梯度方差，"
        "PPO 尤其敏感（critic warm-up 需要足够 step）。",
        "保持 total batch=512（用 grad accumulation 凑），只牺牲 wall-clock；"
        "步数不足时明确标注 step 并只与同 step 的基线比。"),

    DeviationFactor(
        "基础模型权重 snapshot 差异", "环境",
        -2.0, 2.0, "both",
        "Qwen2.5-3B/7B 的 HF 权重在 2025 年有过更新，base 模型尤其明显。",
        "锁定 revision/commit hash 并在报告中写明。"),

    DeviationFactor(
        "max_response_length=500 截断", "算法",
        -4.0, 0.0, "negative",
        "论文 max_response_length=500、retrieved content 截断 500 tokens。"
        "多轮检索（B=4, top-k=3）很容易超长；被截断的 trajectory 拿不到 answer，"
        "reward 恒为 0，会形成对长轨迹的系统性惩罚。",
        "统计截断率（本仓库 rollout 已记录），>5% 就调 max_response_length 或 top-k。"),

    DeviationFactor(
        "KL 系数 β=0.001 与 warm-up ratio", "算法",
        -3.0, 1.0, "both",
        "论文 PPO：policy lr 1e-6、critic lr 1e-5、warmup 0.285/0.015、GAE λ=γ=1.0、β=0.001、ε=0.2。"
        "λ=γ=1 无折扣 + 长轨迹会让优势估计方差很大。",
        "逐条对齐 verl 配置项；warm-up 比例错了会让前 100 步几乎无效。"),

    DeviationFactor(
        "随机种子与 temperature=1.0 采样", "随机性",
        -4.0, 4.0, "both",
        "论文 rollout temperature=1.0、top-p=1.0，单条 seed 的 run-to-run 方差本来就不小；"
        "论文自己也说「training diverges 时取最近的稳定 checkpoint」，说明存在不稳定 run。",
        "至少 3 seeds；报告均值 ± 标准差，而不是挑最好的一次。"),

    DeviationFactor(
        "评测口径：EM 归一化实现", "评测",
        -3.0, 3.0, "both",
        "论文用 EM（Yu et al. 2024 口径）。normalize 的细微差别"
        "（是否去冠词、是否去重音、是否处理中文标点）会直接改分数。"
        "本仓库 content 场景还额外引入了数值容差口径。",
        "统一用本仓库 searchr1_repro/reward.py 的 normalize_answer，并对拍 100 条。"),

    DeviationFactor(
        "数据划分与版本（NQ/HotpotQA train 合并）", "数据",
        -3.0, 1.0, "both",
        "论文把 NQ train + HotpotQA train 合并训练；不同来源的 split 大小/去重策略不同，"
        "会改变 in-domain 与 out-of-domain 的相对表现。",
        "用官方 scripts/data_process/nq_search.py 与同版本 HF dataset。"),
]


def expected_gap() -> dict:
    """汇总：在"尽力对齐但仍受算力限制"的复现中，预期与论文的差距区间。"""
    lo = sum(f.delta_low for f in FACTORS)
    hi = sum(f.delta_high for f in FACTORS)
    # 悲观叠加不合理（各因子非独立），给出两种口径
    rms = (sum((max(abs(f.delta_low), abs(f.delta_high))) ** 2 for f in FACTORS)) ** 0.5
    return {
        "linear_sum_low": round(lo, 2),
        "linear_sum_high": round(hi, 2),
        "rms_estimate": round(rms, 2),
        "practical_note": (
            "线性叠加是最悲观估计；实际各因子部分相关。"
            "经验上：严格对齐检索器与语料、保持 global batch、3 seeds 取均值，"
            "一个执行良好的复现应落在论文值 -3 ~ -8 个 EM 百分点以内；"
            "若差距 > 12 个百分点，基本可以判定是检索器/语料/截断这三件事之一没对齐。"
        ),
    }


def as_rows() -> List[dict]:
    return [asdict(f) for f in FACTORS]


if __name__ == "__main__":
    import json
    print(json.dumps(expected_gap(), ensure_ascii=False, indent=2))
