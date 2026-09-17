# -*- coding: utf-8 -*-
"""在自建评测集 ContentSearch-Bench 上评测一个已经训好的 Search-R1 checkpoint。

两种模式：
  --ckpt  + SGLang 服务  -> 真实推理（需要 GPU）
  --offline               -> 只做语料/检索/评测口径自检（无需 GPU）

输出与论文表格同结构，且额外给出"检索行为指标"：
    EM、平均检索次数、检索命中率(recall@k)、拒答率、平均响应长度
这些在论文里没有，但对落地更重要——EM 一样的两个模型，检索次数差一倍就是成本差一倍。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searchr1_repro.retriever import HttpRetriever, ZeroDependencyRetriever, load_corpus  # noqa: E402
from searchr1_repro.reward import compute_reward  # noqa: E402
from searchr1_repro.rollout import RolloutConfig, run_rollout  # noqa: E402


def make_sglang_policy(model: str, port: int, cfg: RolloutConfig):
    """用 SGLang 的 OpenAI 兼容接口做多轮 rollout。

    真机路径：每个 action 调一次 /v1/completions，遇到 stop token 停。
    stop 用 </search> / </answer>，与论文 Algorithm 1 一致。
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise SystemExit("pip install openai")
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="EMPTY")

    def policy(state, c):
        out = client.completions.create(
            model=model if model.startswith("/") else model,
            prompt=state.full,
            max_tokens=cfg.max_response_length,
            temperature=cfg.rollout_temperature if hasattr(cfg, "rollout_temperature") else 1.0,
            stop=["</search>", "</answer>"],
        )
        return out.choices[0].text
    return policy


def offline_selfcheck(eval_path: str, corpus_path: str) -> dict:
    """无 GPU 自检：语料可索引、题目可检索、答案口径可计算。"""
    corpus = load_corpus(corpus_path)
    retriever = ZeroDependencyRetriever(corpus)
    cases = [json.loads(l) for l in open(eval_path, encoding="utf-8") if l.strip()]

    recall_hit, numeric_cnt, unans_cnt = 0, 0, 0
    for c in cases:
        hits = [d for d, _ in retriever.search_raw(c["question"], 3)]
        sup = set(c.get("supporting_docs") or [])
        if sup and (set(hits) & sup):
            recall_hit += 1
        if c.get("gold_is_numeric"):
            numeric_cnt += 1
        if not c.get("answerable", True):
            unans_cnt += 1

    ans = [c for c in cases if c.get("answerable", True)]
    return {
        "n_corpus": len(corpus),
        "n_cases": len(cases),
        "top3_recall_hit_rate": round(recall_hit / max(len(ans), 1), 4),
        "numeric_questions": numeric_cnt,
        "unanswerable_questions": unans_cnt,
        "note": "用问题原文直接检索的召回率上界；真实 rollout 会改写 query，通常更低。",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--sglang-port", type=int, default=30000)
    ap.add_argument("--eval", default="data/content_eval.jsonl")
    ap.add_argument("--corpus", default="data/content_corpus.jsonl")
    ap.add_argument("--retrieval-url", default="http://127.0.0.1:8010/retrieve")
    ap.add_argument("--out", default="results/eval/content_bench.json")
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--dump-preds", default="results/preds/content.jsonl")
    args = ap.parse_args()

    if args.offline or not args.ckpt:
        res = offline_selfcheck(args.eval, args.corpus)
        print(json.dumps(res, ensure_ascii=False, indent=2))
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        return

    cfg = RolloutConfig.from_paper()
    cfg.domain = "content"
    policy = make_sglang_policy(args.ckpt, args.sglang_port, cfg)
    retriever = HttpRetriever(args.retrieval_url, timeout=30)
    cases = [json.loads(l) for l in open(args.eval, encoding="utf-8") if l.strip()]

    per_task, per_diff = defaultdict(lambda: [0.0, 0]), defaultdict(lambda: [0.0, 0])
    total, nsearch, nlength, refused = 0.0, 0, 0, 0
    preds = []
    for c in cases:
        traj = run_rollout(policy, retriever, c["question"], cfg)
        pred = traj.answer or ""
        r = compute_reward(pred, c["gold"], c.get("gold_is_numeric", False))
        total += r
        nsearch += traj.n_search
        nlength += traj.response_length
        if "信息不足" in pred:
            refused += 1
        per_task[c["task_type"]][0] += r
        per_task[c["task_type"]][1] += 1
        per_diff[c["difficulty"]][0] += r
        per_diff[c["difficulty"]][1] += 1
        preds.append({"id": c["id"], "prediction": pred, "gold": c["gold"],
                      "gold_is_numeric": c.get("gold_is_numeric", False),
                      "task_type": c["task_type"], "platform": c["platform"],
                      "n_search": traj.n_search, "trajectory": traj.text})

    n = max(len(cases), 1)
    report = {
        "ckpt": args.ckpt,
        "n": len(cases),
        "EM": round(total / n, 4),
        "avg_search_calls": round(nsearch / n, 3),
        "avg_response_chars": round(nlength / n, 1),
        "refusal_rate": round(refused / n, 4),
        "by_task": {k: round(v[0] / v[1], 4) for k, v in sorted(per_task.items())},
        "by_difficulty": {k: round(v[0] / v[1], 4) for k, v in sorted(per_diff.items())},
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.dump_preds) or ".", exist_ok=True)
    json.dump(report, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    with open(args.dump_preds, "w", encoding="utf-8") as f:
        for p in preds:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
