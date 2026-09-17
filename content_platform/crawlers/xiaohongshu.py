# -*- coding: utf-8 -*-
"""小红书数据获取 —— 不提供逆向抓取，提供两条合规路径。

为什么不直接写爬虫
------------------------------------------------------------------
小红书核心接口（`/api/sns/web/v1/search/notes` 等）全部需要 `x-s` / `x-t` 签名，
签名算法由前端 JS 混淆生成且**持续变更**。写出来有两个问题：
  1. 属于逆向工程，维护成本极高，且大概率违反平台服务条款；
  2. 对一个"算法实习作品集"来说，这既不加分，还会显得工程判断力有问题。

工程上正确的做法是按优先级选：
  A. 官方开放平台 / 商业化数据接口（有授权，字段最稳）
  B. 创作者中心的"数据导出"（自己账号的后台数据，完全合规）
  C. 第三方合规数据服务商（如新榜、蝉妈妈等的 API）
  D. 人工抽样导出（评测集只需要几百条，人工成本可接受）

本文件实现 A/B/D 三条路径的统一入口：只要你能拿到 jsonl/csv，就能进 pipeline。

用法：
    # 路径 B/D：把人工导出的 csv 转成标准 schema
    python -m content_platform.crawlers.xiaohongshu --import-file raw.xlsx --out data/xhs.jsonl

    # 路径 A：填好 API 凭据后直接拉
    python -m content_platform.crawlers.xiaohongshu --api-key xxx --keyword "护肤" --out data/xhs.jsonl
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from typing import Iterable, List

# 中文表头 -> 统一 schema 的字段映射（人工导出常见列名）
HEADER_MAP = {
    "标题": "title", "笔记标题": "title", "title": "title",
    "作者": "author", "博主": "author", "昵称": "author", "author": "author",
    "发布时间": "publish_date", "发布日期": "publish_date", "date": "publish_date",
    "点赞": "likes", "点赞数": "likes", "likes": "likes",
    "收藏": "collects", "收藏数": "collects", "collects": "collects",
    "评论": "comments", "评论数": "comments", "comments": "comments",
    "曝光": "views", "曝光量": "views", "views": "views",
    "话题": "topic", "分类": "topic", "topic": "topic",
    "正文": "note", "简介": "note", "note": "note",
    "链接": "url", "url": "url",
}


def _coerce_int(v) -> int:
    try:
        s = str(v).strip().lower().replace(",", "")
        mult = 1
        for u, m in (("万", 10000), ("w", 10000), ("亿", 100000000)):
            if s.endswith(u):
                mult, s = m, s[:-len(u)]
                break
        return int(float(s) * mult)
    except Exception:                   # noqa: BLE001
        return 0


def import_file(path: str) -> List[dict]:
    """把 csv/xlsx/jsonl 转成标准 schema。"""
    from ..schema import ContentItem
    rows: List[dict] = []

    if path.endswith(".jsonl"):
        raw = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    elif path.endswith((".csv", ".tsv")):
        delim = "\t" if path.endswith(".tsv") else ","
        with open(path, encoding="utf-8-sig") as f:
            raw = list(csv.DictReader(f, delimiter=delim))
    elif path.endswith((".xlsx", ".xls")):
        import pandas as pd
        raw = pd.read_excel(path).to_dict("records")
    else:
        raise SystemExit(f"不支持的格式: {path}")

    for i, r in enumerate(raw):
        m = {}
        for k, v in r.items():
            key = HEADER_MAP.get(str(k).strip(), str(k).strip())
            m[key] = v
        item = ContentItem(
            id=m.get("id") or f"xiaohongshu_{i:05d}",
            platform="xiaohongshu",
            title=str(m.get("title", "")).strip(),
            author=str(m.get("author", "")).strip(),
            publish_date=str(m.get("publish_date", "")).strip()[:10],
            topic=str(m.get("topic", "unknown")).strip() or "unknown",
            views=_coerce_int(m.get("views", 0)),
            likes=_coerce_int(m.get("likes", 0)),
            collects=_coerce_int(m.get("collects", 0)),
            comments=_coerce_int(m.get("comments", 0)),
            note=str(m.get("note", ""))[:300],
            url=str(m.get("url", "")),
        )
        if item.title:
            rows.append(item.to_dict())
    return rows


def crawl_api(api_key: str, keyword: str, pages: int = 1, sleep: float = 1.0) -> List[dict]:
    """官方/授权 API 路径的骨架。

    真实 endpoint 与鉴权方式取决于你买的哪家服务，这里只给出统一出口，
    避免把某一家供应商的参数写死进仓库。
    """
    if not api_key:
        raise SystemExit("需要 --api-key（请从你采购的数据服务商处获取）")
    raise NotImplementedError(
        "请在 data_providers/ 下按你的服务商实现 crawl_api，"
        "输出必须符合 content_platform/schema.py 的 ContentItem。"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--import-file", default=None, help="人工导出/官方导出的 csv/xlsx/jsonl")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--keyword", default=None)
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--out", default="data/xiaohongshu_corpus.jsonl")
    args = ap.parse_args()

    if args.import_file:
        rows = import_file(args.import_file)
    elif args.api_key:
        rows = crawl_api(args.api_key, args.keyword or "", args.pages)
    else:
        raise SystemExit("请指定 --import-file 或 --api-key")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} items -> {args.out}")


if __name__ == "__main__":
    main()
