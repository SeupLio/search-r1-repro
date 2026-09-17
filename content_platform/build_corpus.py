# -*- coding: utf-8 -*-
"""把内容平台条目转换成 Search-R1 语料格式（与 wiki-18.jsonl 完全同构）。

Search-R1 语料约定（README: "Use your own dataset" / "Corpora"）：
    每行一个 JSON：{"id": str, "contents": str}
    contents = '"' + title + '"\n' + text
这样 build_index.sh 与检索服务可以零改动复用。
"""
from __future__ import annotations

import argparse
import json
import os

from .seed_data import (AUTHOR_NAME, PLATFORM_CN, VIEW_NAME, parse_seed_items)


def item_to_contents(it: dict) -> str:
    """把一条内容渲染成检索用的纯文本。

    刻意做成"元数据 + 摘要"的形式，因为内容平台的检索单元本来就是元数据密集的短文本，
    这也决定了迁移后 top-k / max_retrieved_tokens 需要重新调参（见报告偏差分析）。
    """
    p = it["platform"]
    head = f'{it["title"]}'
    parts = [
        f'平台：{it["platform_cn"]}',
        f'话题：{it["topic"]}',
        f'{AUTHOR_NAME[p]}：{it["author"]}',
        f'发布日期：{it["publish_date"]}',
        f'{VIEW_NAME[p]}：{it["views"]}',
        f'点赞数：{it["likes"]}',
    ]
    if p == "bilibili":
        parts += [f'投币数：{it["collects"]}', f'收藏数：{it["comments"]}']
    elif p == "xiaohongshu":
        parts += [f'收藏数：{it["collects"]}']
    else:
        parts += [f'分享数：{it["shares"]}']
    parts += [f'评论数：{it["comments"]}']
    if it.get("duration_sec"):
        parts.append(f'时长：{it["duration_sec"]}秒')
    parts.append(f'简介：{it["note"]}')
    return '"' + head + '"\n' + "；".join(parts)


def build(items: list[dict]) -> list[dict]:
    return [{"id": it["id"], "contents": item_to_contents(it),
             "platform": it["platform"], "topic": it["topic"]} for it in items]


TOPICS = ["数码测评", "健身", "美食探店", "旅行攻略", "职场成长", "家居收纳",
          "读书笔记", "美妆护肤", "宠物日常", "摄影技巧", "投资理财", "手工DIY"]
TITLE_TPL = [
    "{topic}入门：新手最容易踩的{k}个坑",
    "我用{topic}改变了 everyday：{k}条真实经验",
    "{topic}进阶指南：从零到进阶的完整路径（第{k}期）",
    "关于{topic}，我有话要说——第{k}期",
    "{topic}实测对比：{k}款谁更值得",
]
NOTE_TPL = [
    "围绕{topic}做了长期记录，重点讲清楚方法和取舍。",
    "结合实际体验讲{topic}，避免空谈理论。",
    "一期关于{topic}的复盘，含踩坑与修正。",
    "把{topic}的关键点拆开讲，适合新手。",
]
AUTHORS = ["小林同学", "阿哲", "Mia", "老陈", "青野", "大鹏", "一只鹿", "北岸", "南屿", "拾柒"]


def make_distractors(n: int, seed: int = 1234) -> list[dict]:
    """生成 n 条合成"干扰文档"。

    为什么需要它：原始种子语料只有 45 条，任何检索器在上面都能拿到接近满分的
    recall@k，测不出真实检索差距。加入大量同分布干扰文档后，
    recall 才会随语料规模退化，从而真实反映「检索器质量」这一偏差源。
    """
    import random
    from datetime import date, timedelta
    rng = random.Random(seed)
    out = []
    start = date(2024, 1, 1)
    for i in range(n):
        topic = rng.choice(TOPICS)
        platform = rng.choice(["bilibili", "xiaohongshu", "douyin"])
        k = rng.randint(3, 12)
        title = rng.choice(TITLE_TPL).format(topic=topic, k=k)
        d = start + timedelta(days=rng.randint(0, 700))
        views = rng.randint(5_000, 4_000_000)
        likes = int(views * rng.uniform(0.01, 0.08))
        out.append({
            "id": f"synth_{i:06d}",
            "platform": platform,
            "platform_cn": PLATFORM_CN[platform],
            "topic": topic,
            "title": title,
            "author": f"{rng.choice(AUTHORS)}{rng.randint(1, 999)}",
            "publish_date": d.isoformat(),
            "views": views,
            "likes": likes,
            "collects": int(likes * rng.uniform(0.1, 0.5)),
            "comments": int(likes * rng.uniform(0.05, 0.3)),
            "shares": int(likes * rng.uniform(0.05, 0.4)),
            "duration_sec": rng.choice([0, 180, 420, 900, 1500]),
            "note": rng.choice(NOTE_TPL).format(topic=topic),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/content_corpus.jsonl")
    ap.add_argument("--from-jsonl", default=None,
                    help="用真实抓取得到的 jsonl（schema.py 的 ContentItem）替代合成种子")
    ap.add_argument("--distractors", type=int, default=0,
                    help="额外追加 N 条合成干扰文档，用于检索规模实验")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.from_jsonl:
        items = [json.loads(l) for l in open(args.from_jsonl, encoding="utf-8") if l.strip()]
    else:
        items = parse_seed_items()
        print("[warn] 使用合成种子语料（demo）。真实数据请用 --from-jsonl 传入抓取结果。")
    if args.distractors:
        items = items + make_distractors(args.distractors, seed=args.seed)
        print(f"[info] 追加 {args.distractors} 条合成干扰文档")

    rows = build(items)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} docs -> {args.out}")


if __name__ == "__main__":
    main()
