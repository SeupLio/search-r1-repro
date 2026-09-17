# -*- coding: utf-8 -*-
"""
论文原文数字的唯一事实来源 (single source of truth)。

所有数字均从 arXiv:2503.09516v5 (COLM 2025, Search-R1) 正文/附录表格中逐条录入，
未做任何插值或推测。修改本文件前请先核对 PDF 对应表格。

用途：
  1. 作为复现实验的对照基准 (reference baseline)；
  2. 供 analysis/make_figures.py 绘制 "论文 vs 复现" 对比图；
  3. 供 report/ 中的偏差分析自动计算 gap。

DATASETS 顺序与论文一致：NQ, TriviaQA, PopQA, HotpotQA, 2wiki, Musique, Bamboogle
其中 NQ / HotpotQA 为 in-domain (训练集混合了 NQ+HotpotQA)，其余为 out-of-domain。
"""
from __future__ import annotations

import csv
import os
from typing import Dict, List

DATASETS: List[str] = ["NQ", "TriviaQA", "PopQA", "HotpotQA", "2wiki", "Musique", "Bamboogle"]
IN_DOMAIN = {"NQ", "HotpotQA"}
OUT_DOMAIN = {"TriviaQA", "PopQA", "2wiki", "Musique", "Bamboogle"}

# ---------------------------------------------------------------------------
# Table 2 (正文): 主结果，Qwen2.5-7B / Qwen2.5-3B (Base/Instruct)，RL=PPO
# ---------------------------------------------------------------------------
TABLE2: Dict[str, Dict[str, float]] = {
    # ---- Qwen2.5-7b ----
    "Qwen2.5-7b / Direct Inference": {"NQ": .134, "TriviaQA": .408, "PopQA": .140, "HotpotQA": .183, "2wiki": .250, "Musique": .031, "Bamboogle": .120, "Avg": .181},
    "Qwen2.5-7b / CoT":              {"NQ": .048, "TriviaQA": .185, "PopQA": .054, "HotpotQA": .092, "2wiki": .111, "Musique": .022, "Bamboogle": .232, "Avg": .106},
    "Qwen2.5-7b / IRCoT":            {"NQ": .224, "TriviaQA": .478, "PopQA": .301, "HotpotQA": .133, "2wiki": .149, "Musique": .072, "Bamboogle": .224, "Avg": .239},
    "Qwen2.5-7b / Search-o1":        {"NQ": .151, "TriviaQA": .443, "PopQA": .131, "HotpotQA": .187, "2wiki": .176, "Musique": .058, "Bamboogle": .296, "Avg": .206},
    "Qwen2.5-7b / RAG":              {"NQ": .349, "TriviaQA": .585, "PopQA": .392, "HotpotQA": .299, "2wiki": .235, "Musique": .058, "Bamboogle": .208, "Avg": .304},
    "Qwen2.5-7b / SFT":              {"NQ": .318, "TriviaQA": .354, "PopQA": .121, "HotpotQA": .217, "2wiki": .259, "Musique": .066, "Bamboogle": .112, "Avg": .207},
    "Qwen2.5-7b / R1-base":          {"NQ": .297, "TriviaQA": .539, "PopQA": .202, "HotpotQA": .242, "2wiki": .273, "Musique": .083, "Bamboogle": .296, "Avg": .276},
    "Qwen2.5-7b / R1-instruct":      {"NQ": .270, "TriviaQA": .537, "PopQA": .199, "HotpotQA": .237, "2wiki": .292, "Musique": .072, "Bamboogle": .293, "Avg": .271},
    "Qwen2.5-7b / Rejection Sampling": {"NQ": .360, "TriviaQA": .592, "PopQA": .380, "HotpotQA": .331, "2wiki": .296, "Musique": .123, "Bamboogle": .355, "Avg": .348},
    "Qwen2.5-7b / Search-R1-base":   {"NQ": .480, "TriviaQA": .638, "PopQA": .457, "HotpotQA": .433, "2wiki": .382, "Musique": .196, "Bamboogle": .432, "Avg": .431},
    "Qwen2.5-7b / Search-R1-instruct": {"NQ": .393, "TriviaQA": .610, "PopQA": .397, "HotpotQA": .370, "2wiki": .414, "Musique": .146, "Bamboogle": .368, "Avg": .385},
    # ---- Qwen2.5-3b ----
    "Qwen2.5-3b / Direct Inference": {"NQ": .106, "TriviaQA": .288, "PopQA": .108, "HotpotQA": .149, "2wiki": .244, "Musique": .020, "Bamboogle": .024, "Avg": .134},
    "Qwen2.5-3b / CoT":              {"NQ": .023, "TriviaQA": .032, "PopQA": .005, "HotpotQA": .021, "2wiki": .021, "Musique": .002, "Bamboogle": .000, "Avg": .015},
    "Qwen2.5-3b / IRCoT":            {"NQ": .111, "TriviaQA": .312, "PopQA": .200, "HotpotQA": .164, "2wiki": .171, "Musique": .067, "Bamboogle": .240, "Avg": .181},
    "Qwen2.5-3b / Search-o1":        {"NQ": .238, "TriviaQA": .472, "PopQA": .262, "HotpotQA": .221, "2wiki": .218, "Musique": .054, "Bamboogle": .320, "Avg": .255},
    "Qwen2.5-3b / RAG":              {"NQ": .348, "TriviaQA": .544, "PopQA": .387, "HotpotQA": .255, "2wiki": .226, "Musique": .047, "Bamboogle": .080, "Avg": .270},
    "Qwen2.5-3b / SFT":              {"NQ": .249, "TriviaQA": .292, "PopQA": .104, "HotpotQA": .186, "2wiki": .248, "Musique": .044, "Bamboogle": .112, "Avg": .176},
    "Qwen2.5-3b / R1-base":          {"NQ": .226, "TriviaQA": .455, "PopQA": .173, "HotpotQA": .201, "2wiki": .268, "Musique": .055, "Bamboogle": .224, "Avg": .229},
    "Qwen2.5-3b / R1-instruct":      {"NQ": .210, "TriviaQA": .449, "PopQA": .171, "HotpotQA": .208, "2wiki": .275, "Musique": .060, "Bamboogle": .192, "Avg": .224},
    "Qwen2.5-3b / Rejection Sampling": {"NQ": .294, "TriviaQA": .488, "PopQA": .332, "HotpotQA": .240, "2wiki": .233, "Musique": .059, "Bamboogle": .210, "Avg": .265},
    "Qwen2.5-3b / Search-R1-base":   {"NQ": .406, "TriviaQA": .587, "PopQA": .435, "HotpotQA": .284, "2wiki": .273, "Musique": .049, "Bamboogle": .088, "Avg": .303},
    "Qwen2.5-3b / Search-R1-instruct": {"NQ": .341, "TriviaQA": .545, "PopQA": .378, "HotpotQA": .324, "2wiki": .319, "Musique": .103, "Bamboogle": .264, "Avg": .325},
}

# ---------------------------------------------------------------------------
# Table 3 (正文): PPO vs GRPO
# ---------------------------------------------------------------------------
TABLE3: Dict[str, Dict[str, float]] = {
    "Qwen2.5-7b / Search-R1-base (GRPO)":     {"NQ": .395, "TriviaQA": .560, "PopQA": .388, "HotpotQA": .326, "2wiki": .297, "Musique": .125, "Bamboogle": .360, "Avg": .350},
    "Qwen2.5-7b / Search-R1-instruct (GRPO)": {"NQ": .429, "TriviaQA": .623, "PopQA": .427, "HotpotQA": .386, "2wiki": .346, "Musique": .162, "Bamboogle": .400, "Avg": .396},
    "Qwen2.5-7b / Search-R1-base (PPO)":      {"NQ": .480, "TriviaQA": .638, "PopQA": .457, "HotpotQA": .433, "2wiki": .382, "Musique": .196, "Bamboogle": .432, "Avg": .431},
    "Qwen2.5-7b / Search-R1-instruct (PPO)":  {"NQ": .393, "TriviaQA": .610, "PopQA": .397, "HotpotQA": .370, "2wiki": .414, "Musique": .146, "Bamboogle": .368, "Avg": .385},
    "Qwen2.5-3b / Search-R1-base (GRPO)":     {"NQ": .421, "TriviaQA": .583, "PopQA": .413, "HotpotQA": .297, "2wiki": .274, "Musique": .066, "Bamboogle": .128, "Avg": .312},
    "Qwen2.5-3b / Search-R1-instruct (GRPO)": {"NQ": .397, "TriviaQA": .565, "PopQA": .391, "HotpotQA": .331, "2wiki": .310, "Musique": .124, "Bamboogle": .232, "Avg": .336},
    "Qwen2.5-3b / Search-R1-base (PPO)":      {"NQ": .406, "TriviaQA": .587, "PopQA": .435, "HotpotQA": .284, "2wiki": .273, "Musique": .049, "Bamboogle": .088, "Avg": .303},
    "Qwen2.5-3b / Search-R1-instruct (PPO)":  {"NQ": .341, "TriviaQA": .545, "PopQA": .378, "HotpotQA": .324, "2wiki": .319, "Musique": .103, "Bamboogle": .264, "Avg": .325},
}

# ---------------------------------------------------------------------------
# Table 4 / Table 6: retrieved token loss masking 消融
# ---------------------------------------------------------------------------
TABLE46: Dict[str, Dict[str, float]] = {
    "Qwen2.5-7b / w. mask":  {"NQ": .480, "TriviaQA": .638, "PopQA": .457, "HotpotQA": .433, "2wiki": .382, "Musique": .196, "Bamboogle": .432, "Avg": .431},
    "Qwen2.5-7b / w.o. mask": {"NQ": .388, "TriviaQA": .567, "PopQA": .391, "HotpotQA": .325, "2wiki": .321, "Musique": .108, "Bamboogle": .304, "Avg": .343},
    "Qwen2.5-3b / w. mask":  {"NQ": .406, "TriviaQA": .587, "PopQA": .435, "HotpotQA": .284, "2wiki": .273, "Musique": .049, "Bamboogle": .088, "Avg": .303},
    "Qwen2.5-3b / w.o. mask": {"NQ": .346, "TriviaQA": .484, "PopQA": .365, "HotpotQA": .241, "2wiki": .244, "Musique": .053, "Bamboogle": .104, "Avg": .262},
}

# ---------------------------------------------------------------------------
# Table 7: top-k (每次检索返回段落数) 消融, Qwen2.5-7b-base, PPO
# ---------------------------------------------------------------------------
TABLE7: Dict[str, Dict[str, float]] = {
    "topk=1": {"NQ": .426, "TriviaQA": .614, "PopQA": .422, "HotpotQA": .393, "2wiki": .296, "Musique": .146, "Bamboogle": .328, "Avg": .375},
    "topk=3": {"NQ": .480, "TriviaQA": .638, "PopQA": .457, "HotpotQA": .433, "2wiki": .382, "Musique": .196, "Bamboogle": .432, "Avg": .431},
    "topk=5": {"NQ": .479, "TriviaQA": .634, "PopQA": .440, "HotpotQA": .394, "2wiki": .343, "Musique": .156, "Bamboogle": .352, "Avg": .400},
}

# ---------------------------------------------------------------------------
# Table 8: GRPO group size 消融, Qwen2.5-7b-base
# ---------------------------------------------------------------------------
TABLE8: Dict[str, Dict[str, float]] = {
    "size=1": {"NQ": .463, "TriviaQA": .605, "PopQA": .449, "HotpotQA": .392, "2wiki": .413, "Musique": .163, "Bamboogle": .384, "Avg": .410},
    "size=3": {"NQ": .385, "TriviaQA": .580, "PopQA": .396, "HotpotQA": .329, "2wiki": .333, "Musique": .117, "Bamboogle": .400, "Avg": .363},
    "size=5": {"NQ": .395, "TriviaQA": .560, "PopQA": .388, "HotpotQA": .326, "2wiki": .297, "Musique": .125, "Bamboogle": .360, "Avg": .350},
}

# ---------------------------------------------------------------------------
# Table 5 (附录 C): Qwen2.5-14B 主结果
# ---------------------------------------------------------------------------
TABLE5: Dict[str, Dict[str, float]] = {
    "Qwen2.5-14b / Direct Inference": {"NQ": .198, "TriviaQA": .531, "PopQA": .184, "HotpotQA": .217, "2wiki": .253, "Musique": .045, "Bamboogle": .160, "Avg": .227},
    "Qwen2.5-14b / CoT":              {"NQ": .190, "TriviaQA": .495, "PopQA": .148, "HotpotQA": .269, "2wiki": .297, "Musique": .054, "Bamboogle": .432, "Avg": .269},
    "Qwen2.5-14b / IRCoT":            {"NQ": .114, "TriviaQA": .375, "PopQA": .166, "HotpotQA": .230, "2wiki": .248, "Musique": .102, "Bamboogle": .312, "Avg": .221},
    "Qwen2.5-14b / Search-o1":        {"NQ": .347, "TriviaQA": .635, "PopQA": .241, "HotpotQA": .268, "2wiki": .161, "Musique": .099, "Bamboogle": .416, "Avg": .310},
    "Qwen2.5-14b / RAG":              {"NQ": .327, "TriviaQA": .585, "PopQA": .376, "HotpotQA": .279, "2wiki": .160, "Musique": .051, "Bamboogle": .192, "Avg": .281},
    "Qwen2.5-14b / SFT":              {"NQ": .361, "TriviaQA": .467, "PopQA": .150, "HotpotQA": .248, "2wiki": .278, "Musique": .089, "Bamboogle": .160, "Avg": .250},
    "Qwen2.5-14b / R1-base":          {"NQ": .369, "TriviaQA": .626, "PopQA": .270, "HotpotQA": .306, "2wiki": .326, "Musique": .117, "Bamboogle": .488, "Avg": .357},
    "Qwen2.5-14b / R1-instruct":      {"NQ": .334, "TriviaQA": .628, "PopQA": .253, "HotpotQA": .294, "2wiki": .325, "Musique": .108, "Bamboogle": .432, "Avg": .339},
    "Qwen2.5-14b / Search-R1-base":   {"NQ": .486, "TriviaQA": .676, "PopQA": .480, "HotpotQA": .468, "2wiki": .470, "Musique": .241, "Bamboogle": .528, "Avg": .479},
    "Qwen2.5-14b / Search-R1-instruct": {"NQ": .424, "TriviaQA": .660, "PopQA": .442, "HotpotQA": .436, "2wiki": .379, "Musique": .210, "Bamboogle": .480, "Avg": .433},
}

ALL_TABLES = {
    "table2_main": TABLE2,
    "table3_ppo_grpo": TABLE3,
    "table46_loss_mask": TABLE46,
    "table7_topk": TABLE7,
    "table8_group_size": TABLE8,
    "table5_14b": TABLE5,
}

# ---------------------------------------------------------------------------
# 论文实验设置 (Appendix B.2)，复现时必须逐条对齐
# ---------------------------------------------------------------------------
PAPER_SETUP = {
    "rl_default": "PPO",
    "total_steps": 500,
    "save_ckpt_every": 100,
    "hardware": "1 node x 8xH100",
    "policy_lr": 1e-6,
    "critic_lr": 1e-5,
    "warmup_ratio_policy": 0.285,
    "warmup_ratio_critic": 0.015,
    "gae_lambda": 1.0,
    "gae_gamma": 1.0,
    "kl_coef_beta": 0.001,
    "clip_ratio_eps": 0.2,
    "train_batch_size": 512,
    "ppo_mini_batch_size": 256,
    "ppo_micro_batch_size": 64,
    "max_prompt_length": 4096,
    "max_response_length": 500,
    "max_retrieved_tokens": 500,
    "rollout_temperature": 1.0,
    "rollout_top_p": 1.0,
    "rollout_engine_paper": "vLLM 0.6.3 (tp=1, gpu_mem_util=0.6)",
    "max_action_budget_B": 4,
    "topk_retrieved": 3,
    "reward": "Exact Match (EM), outcome-only, no format reward",
    "corpus": "2018 Wikipedia dump (Karpukhin et al. 2020), 21M passages",
    "retriever": "E5 (intfloat/e5-base-v2 家族), FAISS flat / ANN",
    "train_data": "NQ train + HotpotQA train (merged)",
    "grpo_group_size": 5,
    "grpo_n_samples": 5,
    "fsdp": "FSDP + CPU offload + gradient checkpointing",
}


def dump_all_csv(outdir: str) -> List[str]:
    """把全部表格落盘为 CSV，供绘图 / 报告 / 人工核对使用。"""
    os.makedirs(outdir, exist_ok=True)
    written = []
    for name, table in ALL_TABLES.items():
        path = os.path.join(outdir, f"{name}.csv")
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["method"] + DATASETS + ["Avg"])
            for method, scores in table.items():
                w.writerow([method] + [f"{scores[d]:.3f}" for d in DATASETS] + [f"{scores['Avg']:.3f}"])
        written.append(path)
    return written


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(here, "..", "results", "paper_reference")
    for p in dump_all_csv(out):
        print("wrote", os.path.normpath(p))
