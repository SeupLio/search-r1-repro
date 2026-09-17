# -*- coding: utf-8 -*-
"""生成"失败案例库"：对每个失败类型挑出可读的典型样本，连同完整轨迹一起落盘。

为什么要这个
------------------------------------------------------------------
论文只给 EM 总分。但在真实项目里，你要向人解释"为什么这个 Agent 不行"时，
总分是没用的，需要的是：
    "这一类失败有 N 条，典型长这样，根因是 X，修法是 Y。"

本脚本模拟一个"训练到中途、各种毛病都还在"的 agent checkpoint：
  * 大部分题能正常「先检索、再作答」（ideal）；
  * 但仍有相当比例出现 不检索 / 检索了不用 / 检索停不下来 / query 乱写 / 格式崩掉。

跑完整个评测集后按失败类型分组，每条给出：
  question / gold / prediction / 完整 trajectory / 命中文档 / 归因理由。

产物：
  results/failure_attribution/cases_all.jsonl       全部案例
  results/failure_attribution/cases_by_type.md      按类型分组的可读报告（每类 3 例）
  results/failure_attribution/taxonomy_summary.json 类型分布
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from content_platform.seed_data import parse_seed_items
from eval.failure_attribution import (FAILURE_DESC, FAILURE_TYPES, attribute,
                                      summarize)
from eval.run_dryrun import (ParametricMemory, _info_docs, build_subqueries,
                             naive_reader)
from searchr1_repro.retriever import ZeroDependencyRetriever, load_corpus
from searchr1_repro.reward import compute_reward
from searchr1_repro.rollout import RolloutConfig, run_rollout, stable_hash

# 模拟"训练到中途"的行为分布：主体已正常，各类毛病仍在
MID_TRAINING_MIX = {
    "ideal": 0.55,        # 先检索再作答（已学会主流程）
    "no_search": 0.10,    # 直接凭记忆答
    "unfaithful": 0.10,   # 检索了但不看结果
    "over_search": 0.08,  # 检索停不下来
    "bad_query": 0.09,    # query 乱写
    "format_fail": 0.08,  # 格式崩掉
}


class MidTrainingPolicy:
    """中训练阶段的混合行为策略（确定性）。"""

    def __init__(self, case: dict, subqueries: List[str], mix: dict,
                 memory: ParametricMemory, seed: int = 0):
        self.case = case
        self.subqueries = subqueries
        self.memory = memory
        r = ((stable_hash(case["id"]) ^ seed) & 0xFFFFFF) / 0xFFFFFF
        acc, self.behavior = 0.0, "ideal"
        for k, v in sorted(mix.items()):
            acc += v
            if r <= acc:
                self.behavior = k
                break

    def __call__(self, state, cfg):
        n = state.n_info
        q = self.case["question"]
        b = self.behavior

        if b == "format_fail":
            return "<think> I think I know this already. </think> The answer should be obvious."

        if b == "no_search":
            return f"<think> I recall this from memory. </think><answer> {self.memory(self.case)} </answer>"

        if b == "over_search":
            return (f"<think> I still need more information, let me search again. </think>"
                    f"<search> {q} (attempt {n + 1}) </search>")

        if b == "bad_query" and n == 0:
            return (f"<think> Let me search for something. </think>"
                    f"<search> 今天天气怎么样 </search>")
        if b == "bad_query":
            return f"<think> I have enough now. </think><answer> {naive_reader(q, _info_docs(state))} </answer>"

        if b == "unfaithful":
            if n == 0:
                return f"<think> Let me look this up. </think><search> {q} </search>"
            # 检索到了，但仍然按参数记忆作答
            return f"<think> I remember the answer. </think><answer> {self.memory(self.case)} </answer>"

        # ---- ideal：多轮检索 + 阅读作答 ----
        if n == 0:
            return f"<think> I need external information for this. </think><search> {q} </search>"
        if n < len(self.subqueries):
            return (f"<think> I still lack part of the information. </think>"
                    f"<search> {self.subqueries[n]} </search>")
        ans = naive_reader(q, _info_docs(state))
        return f"<think> Now I can answer based on the retrieved evidence. </think><answer> {ans} </answer>"


def run(seed: int, corpus_path: str, eval_path: str, outdir: str, mix: dict):
    corpus = load_corpus(corpus_path)
    cases = [json.loads(l) for l in open(eval_path, encoding="utf-8") if l.strip()]
    retriever = ZeroDependencyRetriever(corpus)
    items = parse_seed_items()
    items_by_id = {it["id"]: it for it in items}
    memory = ParametricMemory(items, seed=seed)

    cfg = RolloutConfig.from_paper()
    cfg.domain = "content"

    records = []
    for case in cases:
        policy = MidTrainingPolicy(case, build_subqueries(case, items_by_id),
                                   mix, memory, seed=seed)
        traj = run_rollout(policy, retriever, case["question"], cfg)

        pred = traj.answer or ""
        reward = compute_reward(pred, case["gold"], case.get("gold_is_numeric", False))
        hit_docs = []
        for qy in re.findall(r"<search>(.*?)</search>", traj.text, re.S):
            hit_docs += [d for d, _ in retriever.search_raw(qy.strip(), cfg.topk)]
        docs_ctx = re.findall(r"<information>(.*?)</information>", traj.text, re.S)

        a = attribute(case=case, pred=pred, trajectory_text=traj.text,
                      n_search=traj.n_search, n_valid_search=traj.n_valid_search,
                      has_answer_tag="</answer>" in traj.text,
                      has_search_tag="</search>" in traj.text,
                      hit_docs=hit_docs, retrieved_text=" ".join(docs_ctx),
                      strict_reward=reward, numeric=case.get("gold_is_numeric", False))
        d = a.to_dict()
        d.update({"difficulty": case["difficulty"], "task_type": case["task_type"],
                  "platform": case["platform"], "behavior": policy.behavior,
                  "trajectory": traj.text, "queries": traj.queries})
        records.append(d)

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "cases_all.jsonl"), "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    counts = Counter(r["failure_type"] for r in records)
    n = len(records)
    with open(os.path.join(outdir, "taxonomy_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"n": n,
                   "counts": {t: counts.get(t, 0) for t in FAILURE_TYPES},
                   "distribution": {t: round(counts.get(t, 0) / n, 4) for t in FAILURE_TYPES},
                   "by_behavior": dict(Counter(r["behavior"] for r in records)),
                   "by_difficulty": dict(Counter(r["difficulty"] for r in records))},
                  f, ensure_ascii=False, indent=2)

    by_type = defaultdict(list)
    for r in records:
        by_type[r["failure_type"]].append(r)

    lines = ["# 失败案例库（自动生成）", "",
             f"样本总数 **{n}**。模拟一个「训练到中途」的 agent：主流程已学会，"
             "但各类工程/能力缺陷仍在。", "",
             "| 失败类型 | 含义 | 数量 | 占比 |",
             "|---|---|---|---|"]
    for t in FAILURE_TYPES:
        c = counts.get(t, 0)
        lines.append(f"| `{t}` | {FAILURE_DESC[t]} | {c} | {c/n:.1%} |")
    lines.append("")

    for t in FAILURE_TYPES:
        if t == "correct" or not by_type.get(t):
            continue
        lines += [f"## `{t}` — {FAILURE_DESC[t]}", "",
                  f"共 {len(by_type[t])} 例，下面取 3 例。", ""]
        for r in by_type[t][:3]:
            lines += [
                f"**Q**: {r['question']}",
                "",
                f"- gold: `{r['gold']}`",
                f"- pred: `{r['pred']}`",
                f"- 检索次数 {r['n_search']}｜命中文档 {r['hit_docs']}｜支撑文档 {r['supporting_docs']}",
                f"- 归因理由: {r['detail']}",
                f"- 触发行为: `{r['behavior']}`",
                "",
                "<details><summary>完整轨迹</summary>",
                "",
                "```",
                r["trajectory"][:1800],
                "```",
                "",
                "</details>",
                "",
            ]
    with open(os.path.join(outdir, "cases_by_type.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("failure type counts:")
    for t, c in counts.most_common():
        print(f"  {t:20s} {c:4d}  ({c/n:.1%})  {FAILURE_DESC[t]}")
    print(f"\nwrote {outdir}/cases_all.jsonl, cases_by_type.md, taxonomy_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/content_corpus.jsonl")
    ap.add_argument("--eval", default="data/content_eval.jsonl")
    ap.add_argument("--outdir", default="results/failure_attribution")
    ap.add_argument("--seed", type=int, default=20250916)
    args = ap.parse_args()
    run(args.seed, args.corpus, args.eval, args.outdir, MID_TRAINING_MIX)


if __name__ == "__main__":
    main()
