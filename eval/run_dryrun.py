# -*- coding: utf-8 -*-
"""端到端 dry-run：在无 GPU / 无模型权重的前提下，真实跑通
"语料 -> 索引 -> 检索服务 -> 多轮 rollout -> outcome reward -> 失败归因 -> 落盘" 全链路。

为什么要做 dry-run
------------------------------------------------------------------
真正的 RL 训练需要 8×H100。但整条链路里真正容易写错的
（模板拼接、多轮状态机、retrieved token 位置、reward 口径、归因判定、出图）
跟有没有 GPU 无关。dry-run 用一个确定性的"桩策略"替代 LLM 采样，
把这些逻辑全部真实执行一遍并产出可检查的产物，
从而保证真机上只需要换掉 policy，其余代码是已经被验证过的。

三个待比较的系统（与论文 Table 2 的口径对应）：
    Direct     不检索，纯参数记忆                      ~ Direct Inference / R1
    RAG        单轮检索（question 作 query）+ 阅读器    ~ RAG
    SearchR1   多轮检索（B=4）+ 阅读器                  ~ Search-R1

⚠️ 桩策略不是 LLM。它的绝对分数没有论文意义，
   有意义的是"三者在同样检索器/语料/奖励口径下的相对差异"，
   以及归因分布与出图管线是否可用。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from content_platform.seed_data import (AUTHOR_NAME, PLATFORM_CN, VIEW_NAME,
                                        parse_seed_items)
from eval.failure_attribution import Attribution, attribute, summarize
from searchr1_repro.retriever import ZeroDependencyRetriever, load_corpus
from searchr1_repro.reward import compute_reward, normalize_answer
from searchr1_repro.rollout import RolloutConfig, RolloutState, run_rollout
from searchr1_repro.template import build_prompt

# 参数记忆命中率：模拟不同规模 LLM 对该类知识的掌握程度（用于 Direct 基线）
MEMORY_PRIOR = {"easy": 0.22, "medium": 0.08, "hard": 0.03, "unknown": 0.0}


# ---------------------------------------------------------------------------
# 1) 模拟"参数记忆"
# ---------------------------------------------------------------------------
class ParametricMemory:
    """模拟 LLM 不看检索时"凭记忆"给出的答案。

    - 命中率按难度递减；
    - unanswerable 题永远编造一个答案（这正是幻觉的来源，命中率 0 且不说"信息不足"）。
    """

    def __init__(self, items: List[dict], seed: int = 7):
        self.rng = random.Random(seed)
        self.items = items
        self._distractors = defaultdict(list)
        for it in items:
            self._distractors["author"].append(it["author"])
            self._distractors["title"].append(it["title"])
            self._distractors["date"].append(it["publish_date"])
            self._distractors["num"].append(str(it["views"]))

    def __call__(self, case: dict) -> str:
        diff = case["difficulty"]
        p = MEMORY_PRIOR.get(diff, 0.1)
        if not case.get("answerable", True):
            p = 0.0
        if self.rng.random() < p:
            return case["gold"][0]
        return self._confabulate(case)

    def _confabulate(self, case: dict) -> str:
        task = case.get("task_type", "")
        if "author" in task or "谁" in case["question"]:
            return self.rng.choice(self._distractors["author"])
        if task.startswith("aggregate_count"):
            return str(self.rng.choice([2, 3, 5, 8, 12]))
        if "numeric" in task or "avg" in task or "count" in task:
            return str(self.rng.randint(1000, 900000))
        if "date" in task:
            return self.rng.choice(self._distractors["date"])
        if "cross" in task:
            return self.rng.choice(list(PLATFORM_CN.values()))
        return self.rng.choice(self._distractors["title"])


# ---------------------------------------------------------------------------
# 2) 简易阅读器（RAG / Search-R1 共用的"从检索文本里抽答案"模块）
# ---------------------------------------------------------------------------
_FIELD_PATTERNS = [
    (("UP主", "博主", "作者", "谁"), [r"(?:UP主|博主|作者)：([^；\n]+)"]),
    (("播放量", "曝光量"), [r"(?:播放量|曝光量)：(\d+)"]),
    (("点赞",), [r"点赞数：(\d+)"]),
    (("评论",), [r"评论数：(\d+)"]),
    (("哪一天", "发布日期", "发布"), [r"发布日期：([\d\-/]+)"]),
    (("话题",), [r"话题：([^；\n]+)"]),
]


DOC_RE = re.compile(r'Doc \d+\(Title: "([^"]*)"\)')


def parse_docs(block: str) -> List[tuple]:
    """把检索服务返回的文本解析成 [(title, body), ...]。

    检索服务格式（与 Search-R1 原版一致）：
        Doc 1(Title: "xxx") 平台：...；话题：...；UP主：... Doc 2(Title: "yyy") ...
    语料原始格式（兜底）：
        "title"\n平台：...
    """
    docs: List[tuple] = []
    for m in DOC_RE.finditer(block):
        start = m.end()
        nxt = DOC_RE.search(block, start)
        end = nxt.start() if nxt else len(block)
        docs.append((m.group(1), block[start:end]))
    if not docs:
        for m in re.finditer(r'"([^"]+)"\s*\n(.*?)(?=\n"|$)', block, re.S):
            docs.append((m.group(1), m.group(2)))
    return docs


def _dedup(docs: Sequence[tuple]) -> List[tuple]:
    """多轮检索会重复召回同一条内容，必须去重后再聚合。

    去重键用 "title || body 前 40 字"，避免 title 解析失败时把所有文档折叠成一条。
    """
    seen, out = set(), []
    for t, b in docs:
        key = (t or "").strip() or b[:40]
        if not key or key in seen:
            continue
        seen.add(key)
        out.append((t, b))
    return out


def _relevance(question: str, doc: tuple) -> float:
    def toks(s):
        s = re.sub(r"\s+", "", s.lower())
        return set(s[i:i + 2] for i in range(len(s) - 1)) or {s}
    qt = toks(question)
    dt = toks(doc[0] + doc[1])
    return len(qt & dt) / max(len(qt), 1)


def _field(body: str, pat: str) -> str:
    m = re.search(pat, body)
    return m.group(1).strip() if m else ""


def _constrain(docs: Sequence[tuple], question: str) -> List[tuple]:
    """按问题里显式出现的话题 / 平台约束过滤文档集合。

    RAG（单轮）与 Search-R1（多轮）共用同一个阅读器；
    差别在于多轮能召回更全的候选，过滤后才有得可选。
    """
    out = docs
    topics = re.findall(r"「([^」]+)」", question)
    if topics:
        t = topics[0]
        filtered = [(ti, b) for ti, b in out if f"话题：{t}" in b]
        if filtered:
            out = filtered
    plats = [cn for cn in PLATFORM_CN.values() if cn in question]
    if len(plats) == 1:
        filtered = [(ti, b) for ti, b in out if f"平台：{plats[0]}" in b]
        if filtered:
            out = filtered
    return out


def naive_reader(question: str, blocks: Sequence[str], abstain: bool = False) -> str:
    """规则型阅读器：能覆盖单跳字段抽取与简单聚合，覆盖不了复杂多跳推理。

    刻意保持"朴素"——它就是 RAG 类方法的典型上限：
    检索到了就能答，检索不全或多跳就崩。
    多轮检索（Search-R1）的优势正来自"去重后的文档并集更大"。
    """
    if not blocks:
        return ""
    q = question
    docs = _dedup([d for b in blocks for d in parse_docs(b)])
    if not docs:
        return ""

    # 约束过滤：问题里带了话题/平台时，聚合只能在满足约束的文档子集上做。
    # 这一步正是"推理"的体现——RAG 版没有它，多检索反而会引入噪声。
    agg_docs = _constrain(docs, q)

    # 单跳：问题里带《标题》时直接锁定该条内容
    titled = re.findall(r"《([^》]+)》", q)
    target = None
    if titled:
        for t, b in docs:
            if titled[0] in t:
                target = (t, b)
                break

    # 拒答机制：问的是某条具体内容，但检索结果里根本没有它 -> 应当说"信息不足"。
    # 这是奖励设计直接换来的能力：不加拒答奖励，模型永远不会主动说不知道。
    if abstain and titled and target is None:
        return "信息不足"

    # 聚合：条数
    if "多少条" in q:
        return str(len(agg_docs))
    # 聚合：平均点赞
    if "平均点赞" in q:
        vals = [int(m) for _, b in agg_docs for m in re.findall(r"点赞数：(\d+)", b)]
        return str(int(sum(vals) / len(vals))) if vals else ""
    # 跨平台聚合
    if ("平均" in q and any(p in q for p in PLATFORM_CN.values())) or "点赞总数" in q:
        best, bestv = "", -1.0
        for cn in PLATFORM_CN.values():
            sub = [b for _, b in agg_docs if f"平台：{cn}" in b]
            if not sub:
                continue
            if "点赞总数" in q:
                v = sum(int(m) for b in sub for m in re.findall(r"点赞数：(\d+)", b))
            else:
                nums = [int(m) for b in sub for m in re.findall(r"(?:播放量|曝光量)：(\d+)", b)]
                v = (sum(nums) / len(nums)) if nums else 0.0
            if v > bestv:
                best, bestv = cn, v
        if best:
            return best
    # 最高 / 比较
    if "最高" in q or "更高" in q:
        best, bestv = "", -1
        for t, b in agg_docs:
            m = re.search(r"(?:播放量|曝光量)：(\d+)", b)
            v = int(m.group(1)) if m else -1
            if v > bestv:
                best, bestv = t, v
        if best:
            return best
    # 单跳字段
    if target:
        for keys, pats in _FIELD_PATTERNS:
            if any(k in q for k in keys):
                for pat in pats:
                    v = _field(target[1], pat)
                    if v:
                        return v
        return target[0]

    ranked = sorted(docs, key=lambda d: -_relevance(q, d))
    for keys, pats in _FIELD_PATTERNS:
        if any(k in q for k in keys):
            for _, b in ranked:
                for pat in pats:
                    v = _field(b, pat)
                    if v:
                        return v
    # 兜底
    return " ".join((ranked[0][0] + " " + ranked[0][1]).replace('"', " ").split()[:12])


# ---------------------------------------------------------------------------
# 3) 三个系统的策略实现（统一 generate(context, cfg) -> chunk 接口）
# ---------------------------------------------------------------------------
def _info_docs(state: "RolloutState") -> List[str]:
    """只在 generated 文本上抽 <information> 块（模板里也有同名字面 token）。"""
    return re.findall(r"<information>(.*?)</information>", state.text, re.S)


def make_direct_policy(memory: ParametricMemory, case: dict):
    def policy(state: "RolloutState", cfg: RolloutConfig) -> str:
        return f"<think> I recall this from my own knowledge. </think><answer> {memory(case)} </answer>"
    return policy


def make_rag_policy(reader_ctx: dict):
    """单轮检索：第一次动作发 <search>，拿到 information 后直接作答。"""
    def policy(state: "RolloutState", cfg: RolloutConfig) -> str:
        q = reader_ctx["question"]
        if state.n_info == 0:
            return f"<think> I should look this up. </think><search> {q} </search>"
        docs = _info_docs(state)
        reader_ctx["docs"] = docs
        ans = naive_reader(q, docs)
        return f"<think> The retrieved passages contain the answer. </think><answer> {ans} </answer>"
    return policy


def make_searchr1_policy(reader_ctx: dict, subqueries: List[str], abstain: bool = False):
    """多轮检索：B=4 预算内依次尝试 question / 子问题，最后作答。

    子问题由"伪推理"产生（真实场景由 LLM 生成），
    这里用手写模板模拟"学会了分解子问题"后的行为。
    """
    def policy(state: "RolloutState", cfg: RolloutConfig) -> str:
        q = reader_ctx["question"]
        n_info = state.n_info
        if n_info == 0:
            return f"<think> I need to find out the answer. Let me search first. </think><search> {q} </search>"
        docs = _info_docs(state)
        reader_ctx["docs"] = docs
        # 预算约束：最多 len(subqueries)+1 次检索，最后一次必须是 answer（B=4）
        if n_info < len(subqueries):
            sq = subqueries[n_info]
            think = ("Let me verify the key fact with one more targeted search."
                     if n_info == len(subqueries) - 1
                     else "I still lack part of the information. Let me search for a sub-question.")
            return f"<think> {think} </think><search> {sq} </search>"
        ans = naive_reader(q, docs, abstain=abstain)
        return f"<think> Now I have enough evidence to answer. </think><answer> {ans} </answer>"
    return policy


def build_subqueries(case: dict, items_by_id: Dict[str, dict]) -> List[str]:
    """为多轮策略构造"子问题"（模拟 LLM 的问题分解）。"""
    q = case["question"]
    subs = []
    topic = case.get("topic")
    plat = case.get("platform")
    if topic and plat in PLATFORM_CN:
        subs.append(f"{PLATFORM_CN[plat]} {topic}")
    if topic:
        subs.append(f"{topic} 内容列表")
    for d in (case.get("supporting_docs") or [])[:1]:
        it = items_by_id.get(d)
        if it:
            subs.append(it["title"])
    return subs[:3] if subs else [q]


# ---------------------------------------------------------------------------
# 4) 主流程
# ---------------------------------------------------------------------------
def run(system: str, cases: List[dict], retriever: ZeroDependencyRetriever,
        items_by_id: Dict[str, dict], memory: ParametricMemory,
        cfg: RolloutConfig, seed: int) -> List[Attribution]:
    rng = random.Random(seed)
    results: List[Attribution] = []
    for case in cases:
        ctx = {"question": case["question"], "docs": []}
        if system == "Direct":
            policy = make_direct_policy(memory, case)
        elif system == "RAG":
            policy = make_rag_policy(ctx)
        elif system == "Search-R1+Abstain":
            policy = make_searchr1_policy(ctx, build_subqueries(case, items_by_id), abstain=True)
        else:
            policy = make_searchr1_policy(ctx, build_subqueries(case, items_by_id))

        traj = run_rollout(policy, retriever, case["question"], cfg)
        docs_ctx = re.findall(r"<information>(.*?)</information>", traj.text, re.S)

        # 记录检索是否命中支撑文档
        hit_docs: List[str] = []
        for qy in re.findall(r"<search>(.*?)</search>", traj.text, re.S):
            for doc_id, _score in retriever.search_raw(qy.strip(), cfg.topk):
                hit_docs.append(doc_id)

        pred = traj.answer or ""
        reward = compute_reward(pred, case["gold"], case.get("gold_is_numeric", False))
        attr = attribute(
            case=case, pred=pred, trajectory_text=traj.text,
            n_search=traj.n_search, n_valid_search=traj.n_valid_search,
            has_answer_tag="</answer>" in traj.text,
            has_search_tag="</search>" in traj.text,
            hit_docs=hit_docs, retrieved_text=" ".join(docs_ctx),
            strict_reward=reward, numeric=case.get("gold_is_numeric", False),
        )
        attr.n_search = traj.n_search
        results.append(attr)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/content_corpus.jsonl")
    ap.add_argument("--eval", default="data/content_eval.jsonl")
    ap.add_argument("--outdir", default="results/dryrun")
    ap.add_argument("--seed", type=int, default=20250916)
    ap.add_argument("--systems", default="Direct,RAG,Search-R1,Search-R1+Abstain")
    args = ap.parse_args()

    corpus = load_corpus(args.corpus)
    cases = [json.loads(l) for l in open(args.eval, encoding="utf-8") if l.strip()]
    print(f"corpus={len(corpus)} docs, eval={len(cases)} cases")

    retriever = ZeroDependencyRetriever(corpus)
    items = parse_seed_items()
    items_by_id = {it["id"]: it for it in items}
    memory = ParametricMemory(items, seed=args.seed)
    cfg = RolloutConfig.from_paper()
    cfg.domain = "content"

    os.makedirs(args.outdir, exist_ok=True)
    summary: Dict[str, dict] = {}

    for sysname in [s.strip() for s in args.systems.split(",")]:
        attrs = run(sysname, cases, retriever, items_by_id, memory, cfg, args.seed)
        dist, counts = summarize(attrs)
        em = sum(a.reward for a in attrs) / len(attrs)

        # 分难度
        by_diff = defaultdict(lambda: [0, 0])
        for c, a in zip(cases, attrs):
            by_diff[c["difficulty"]][0] += a.reward
            by_diff[c["difficulty"]][1] += 1
        # 分题型
        by_task = defaultdict(lambda: [0, 0])
        for c, a in zip(cases, attrs):
            by_task[c["task_type"]][0] += a.reward
            by_task[c["task_type"]][1] += 1
        # 分平台
        by_plat = defaultdict(lambda: [0, 0])
        for c, a in zip(cases, attrs):
            by_plat[c["platform"]][0] += a.reward
            by_plat[c["platform"]][1] += 1

        avg_search = sum(a.n_search for a in attrs) / len(attrs)

        summary[sysname] = {
            "EM": round(em, 4),
            "avg_valid_search": round(avg_search, 3),
            "failure_distribution": {k: round(v, 4) for k, v in dist.items()},
            "failure_counts": counts,
            "by_difficulty": {k: round(v[0] / v[1], 4) for k, v in sorted(by_diff.items())},
            "by_task": {k: round(v[0] / v[1], 4) for k, v in sorted(by_task.items())},
            "by_platform": {k: round(v[0] / v[1], 4) for k, v in sorted(by_plat.items())},
        }

        with open(os.path.join(args.outdir, f"{sysname}_cases.jsonl"), "w", encoding="utf-8") as f:
            for a in attrs:
                f.write(json.dumps(a.to_dict(), ensure_ascii=False) + "\n")
        print(f"[{sysname:9s}] EM={em:.4f}  avg_search={avg_search:.2f}  "
              f"fail_top={sorted(counts.items(), key=lambda x: -x[1])[:3]}")

    with open(os.path.join(args.outdir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("wrote", os.path.join(args.outdir, "summary.json"))


if __name__ == "__main__":
    main()
