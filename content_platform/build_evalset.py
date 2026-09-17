# -*- coding: utf-8 -*-
"""自建内容平台评测集生成器（ContentSearch-Bench）。

设计原则（写报告 / 面试时可作为亮点讲）：
  1. **答案可验证**：所有问题都由结构化语料"构造"而非人工编造，gold 由代码从同一份
     语料算出，因此不存在"我自己也不确定答案对不对"的问题。
  2. **覆盖检索必要性梯度**：含 unanswerable 子集，用于检测模型的幻觉/过度自信，
     这是 wiki-QA 类 benchmark（NQ/HotpotQA）天然缺失的能力维度。
  3. **覆盖内容平台特有题型**：数值聚合（播放量最高/平均点赞）、跨平台对比、
     时间约束（2025年X月发布）、长尾实体（具体作品标题）。
  4. **难度分层**：easy(单跳) / medium(多跳·聚合) / hard(跨平台·多约束)。

用法：
    python -m content_platform.build_evalset --out data/content_eval.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List

from .seed_data import AUTHOR_NAME, PLATFORM_CN, VIEW_NAME, parse_seed_items

UNANSWERABLE_ANSWER = "信息不足"
PLATFORMS = ["bilibili", "xiaohongshu", "douyin"]


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------
def _by(items: List[dict], key: str) -> Dict[str, List[dict]]:
    d = defaultdict(list)
    for it in items:
        d[it[key]].append(it)
    return d


def _fmt_num(n) -> List[str]:
    """数值答案给出多种等价写法，降低 EM 对格式的敏感度。"""
    n = int(round(float(n)))
    outs = [str(n)]
    if n >= 10000:
        outs.append(f"{n/10000:.1f}万")
        outs.append(f"{n/10000:.2f}万")
    if n >= 100000000:
        outs.append(f"{n/100000000:.2f}亿")
    return sorted(set(outs))


def _mk(qid, question, gold, numeric, platform, topic, task, difficulty,
        docs, answerable=True, retrieval_required=True) -> dict:
    return {
        "id": qid,
        "question": question,
        "gold": gold if isinstance(gold, list) else [gold],
        "gold_is_numeric": bool(numeric),
        "platform": platform,
        "topic": topic,
        "task_type": task,
        "difficulty": difficulty,
        "supporting_docs": docs,
        "answerable": bool(answerable),
        "retrieval_required": bool(retrieval_required),
    }


# ---------------------------------------------------------------------------
# 各题型生成器
# ---------------------------------------------------------------------------
def gen_single_hop(items: List[dict], rng: random.Random, per_item: int = 2) -> List[dict]:
    """每个内容条目随机抽 per_item 种单跳题型，避免 easy 题占比过高。

    4 种候选：作者 / 播放(曝光)量 / 发布日期 / 点赞数。
    """
    out = []
    for it in items:
        p, plat_cn = it["platform"], it["platform_cn"]
        t = it["title"]

        def author():
            return _mk(f"{it['id']}_author",
                       f"{plat_cn}上标题为《{t}》的内容，{AUTHOR_NAME[p]}是谁？",
                       [it["author"]], False, p, it["topic"],
                       "single_hop_author", "easy", [it["id"]])

        def views():
            return _mk(f"{it['id']}_views",
                       f"《{t}》在{plat_cn}的{VIEW_NAME[p]}是多少？",
                       _fmt_num(it["views"]), True, p, it["topic"],
                       "single_hop_numeric", "easy", [it["id"]])

        def date():
            return _mk(f"{it['id']}_date",
                       f"《{t}》发布于哪一天？",
                       [it["publish_date"], it["publish_date"].replace("-", "/"),
                        f'{it["publish_date"][:4]}年{int(it["publish_date"][5:7])}月'
                        f'{int(it["publish_date"][8:10])}日'],
                       False, p, it["topic"], "single_hop_date", "easy", [it["id"]])

        def likes():
            return _mk(f"{it['id']}_likes",
                       f"《{t}》的点赞数是多少？",
                       _fmt_num(it["likes"]), True, p, it["topic"],
                       "single_hop_numeric", "easy", [it["id"]])

        cands = [author, views, date, likes]
        for fn in rng.sample(cands, min(per_item, len(cands))):
            out.append(fn())
    return out


def gen_topic_aggregate(items: List[dict]) -> List[dict]:
    out = []
    grouped = defaultdict(list)
    for it in items:
        grouped[(it["platform"], it["topic"])].append(it)

    for (p, topic), group in grouped.items():
        plat_cn = PLATFORM_CN[p]
        if len(group) < 2:
            continue
        docs = [g["id"] for g in group]
        # 5) 话题内播放量最高的标题
        top = max(group, key=lambda x: x["views"])
        out.append(_mk(f"{p}_{topic}_maxview",
                       f"{plat_cn}上「{topic}」话题里{VIEW_NAME[p]}最高的内容标题是什么？",
                       [top["title"]], False, p, topic, "aggregate_max", "medium", docs))
        # 6) 话题内点赞最高的作者
        top_like = max(group, key=lambda x: x["likes"])
        out.append(_mk(f"{p}_{topic}_maxlike",
                       f"{plat_cn}「{topic}」话题下点赞数最高的内容，{AUTHOR_NAME[p]}是谁？",
                       [top_like["author"]], False, p, topic, "aggregate_max", "medium", docs))
        # 7) 话题内平均点赞（数值）
        avg_like = sum(g["likes"] for g in group) / len(group)
        out.append(_mk(f"{p}_{topic}_avglike",
                       f"{plat_cn}「{topic}」话题下这些内容的平均点赞数是多少？",
                       _fmt_num(avg_like), True, p, topic, "aggregate_avg", "medium", docs))
        # 8) 话题内条数
        out.append(_mk(f"{p}_{topic}_count",
                       f"{plat_cn}「{topic}」话题下一共有多少条内容？",
                       _fmt_num(len(group)), True, p, topic, "aggregate_count", "medium", docs))
    return out


def gen_pair_compare(items: List[dict], rng: random.Random, n_per_topic: int = 2) -> List[dict]:
    out = []
    grouped = defaultdict(list)
    for it in items:
        grouped[(it["platform"], it["topic"])].append(it)
    for (p, topic), group in grouped.items():
        if len(group) < 2:
            continue
        plat_cn = PLATFORM_CN[p]
        pairs = rng.sample(group, min(len(group), 2 * n_per_topic))
        for i in range(0, len(pairs) - 1, 2):
            a, b = pairs[i], pairs[i + 1]
            win = a if a["views"] >= b["views"] else b
            out.append(_mk(
                f"{a['id']}_{b['id']}_cmp",
                f"《{a['title']}》和《{b['title']}》都在{plat_cn}发布，"
                f"哪一条的{VIEW_NAME[p]}更高？请回答完整标题。",
                [win["title"]], False, p, topic, "multi_hop_compare", "medium",
                [a["id"], b["id"]]))
    return out


def gen_time_filtered(items: List[dict]) -> List[dict]:
    out = []
    grouped = defaultdict(list)
    for it in items:
        y, m, _ = it["publish_date"].split("-")
        grouped[(it["platform"], it["topic"], f"{y}-{m}")].append(it)
    for (p, topic, ym), group in grouped.items():
        if len(group) < 2:
            continue
        plat_cn = PLATFORM_CN[p]
        y, m = ym.split("-")
        top = max(group, key=lambda x: x["views"])
        out.append(_mk(
            f"{p}_{topic}_{ym}_top",
            f"{plat_cn}上{y}年{int(m)}月发布的「{topic}」内容中，{VIEW_NAME[p]}最高的是哪一条？请回答完整标题。",
            [top["title"]], False, p, topic, "time_filtered", "hard",
            [g["id"] for g in group]))
    return out


def gen_cross_platform(items: List[dict]) -> List[dict]:
    out = []
    by_topic = _by(items, "topic")
    for topic, group in by_topic.items():
        by_p = _by(group, "platform")
        if len(by_p) < 2:
            continue
        stats = {p: sum(x["views"] for x in g) / len(g) for p, g in by_p.items()}
        winner = max(stats, key=stats.get)
        out.append(_mk(
            f"cross_{topic}_avgview",
            f"「{topic}」这个话题，在"
            + "、".join(PLATFORM_CN[p] for p in sorted(by_p))
            + f"上都有内容发布。哪个平台的平均{VIEW_NAME['bilibili']}更高？",
            [PLATFORM_CN[winner], winner], False, "cross", topic,
            "cross_platform_compare", "hard", [x["id"] for x in group]))
        # 跨平台：点赞总数
        tot = {p: sum(x["likes"] for x in g) for p, g in by_p.items()}
        win_like = max(tot, key=tot.get)
        out.append(_mk(
            f"cross_{topic}_sumlike",
            f"「{topic}」话题下，哪个平台这些内容的点赞总数最高？",
            [PLATFORM_CN[win_like], win_like], False, "cross", topic,
            "cross_platform_compare", "hard", [x["id"] for x in group]))
    return out


def gen_unanswerable(rng: random.Random, n: int = 18) -> List[dict]:
    """语料中不存在的实体 / 不存在的字段 —— 用于检测幻觉。

    gold 固定为 "信息不足"；模型若给出具体答案，即判为 hallucination/overconfidence。
    """
    fake_titles = [
        "2026年旗舰机横评：十台真机一年的续航真相",
        "一千五价位显示器怎么选？二十台实测色彩",
        "把机械键盘扔掉之后，我理解了什么叫手感",
        "三个月徒手训练，我的体脂率从30%到8%",
        "一个人在关西走了一个月，总共花了三百块",
        "在家复刻兰州牛肉面，汤头熬了六十个钟头",
        "四十平出租屋改造，总共花了六十八万",
        "干皮换季烂脸自救，这套精简护肤我用了两年",
        "155小个子显高穿搭，增高比比例重要",
        "一天走完十座小城，路线我排好了",
    ]
    fake_authors = ["不存在的UP主甲", "虚空博主乙", "查无此人丙"]
    out = []
    for i, t in enumerate(rng.sample(fake_titles, min(n - 6, len(fake_titles)))):
        p = rng.choice(PLATFORMS)
        out.append(_mk(f"unans_title_{i}",
                       f"{PLATFORM_CN[p]}上标题为《{t}》的内容，{AUTHOR_NAME[p]}是谁？",
                       [UNANSWERABLE_ANSWER], False, p, "unknown", "unanswerable", "hard",
                       [], answerable=False))
    for i, a in enumerate(fake_authors):
        p = rng.choice(PLATFORMS)
        out.append(_mk(f"unans_author_{i}",
                       f"{PLATFORM_CN[p]}上{AUTHOR_NAME[p]}「{a}」发布的内容里，"
                       f"{VIEW_NAME[p]}最高的是哪一条？请回答完整标题。",
                       [UNANSWERABLE_ANSWER], False, p, "unknown", "unanswerable", "hard",
                       [], answerable=False))
    for i in range(3):
        p = rng.choice(PLATFORMS)
        out.append(_mk(f"unans_field_{i}",
                       f"{PLATFORM_CN[p]}上内容的「观众完播率」字段，平均值是多少？",
                       [UNANSWERABLE_ANSWER], False, p, "unknown", "unanswerable", "hard",
                       [], answerable=False))
    return out


def build_evalset(items: List[dict], seed: int = 20250916) -> List[dict]:
    rng = random.Random(seed)
    rows: List[dict] = []
    rows += gen_single_hop(items, rng)
    rows += gen_topic_aggregate(items)
    rows += gen_pair_compare(items, rng)
    rows += gen_time_filtered(items)
    rows += gen_cross_platform(items)
    rows += gen_unanswerable(rng)
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["index"] = i
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/content_eval.jsonl")
    ap.add_argument("--seed", type=int, default=20250916)
    ap.add_argument("--from-jsonl", default=None, help="真实抓取的条目 jsonl")
    args = ap.parse_args()

    if args.from_jsonl:
        items = [json.loads(l) for l in open(args.from_jsonl, encoding="utf-8") if l.strip()]
    else:
        items = parse_seed_items()
        print("[warn] 基于合成种子语料生成评测集（demo）。")

    rows = build_evalset(items, seed=args.seed)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    from collections import Counter
    print(f"wrote {len(rows)} eval items -> {args.out}")
    print("  difficulty:", dict(Counter(r["difficulty"] for r in rows)))
    print("  task_type :", dict(Counter(r["task_type"] for r in rows)))
    print("  platform  :", dict(Counter(r["platform"] for r in rows)))
    print("  answerable:", dict(Counter(r["answerable"] for r in rows)))


if __name__ == "__main__":
    main()
