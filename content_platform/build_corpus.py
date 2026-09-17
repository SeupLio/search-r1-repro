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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/content_corpus.jsonl")
    ap.add_argument("--from-jsonl", default=None,
                    help="用真实抓取得到的 jsonl（schema.py 的 ContentItem）替代合成种子")
    args = ap.parse_args()

    if args.from_jsonl:
        items = [json.loads(l) for l in open(args.from_jsonl, encoding="utf-8") if l.strip()]
    else:
        items = parse_seed_items()
        print("[warn] 使用合成种子语料（demo）。真实数据请用 --from-jsonl 传入抓取结果。")

    rows = build(items)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} docs -> {args.out}")


if __name__ == "__main__":
    main()
