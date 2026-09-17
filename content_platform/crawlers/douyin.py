# -*- coding: utf-8 -*-
"""抖音数据获取 —— 同小红书，走合规路径。

抖音接口需要 `a_bogus` / `X-Bogus` 签名（由混淆 JS 动态生成），
实现它属于逆向。本项目采取与小红书一致的策略：
  优先官方开放平台 / 授权数据服务商 / 创作者后台导出。

一个额外坑：抖音的"播放量"在创作者后台才可见，公开页面只有点赞/评论/分享。
因此用公开渠道构建的语料里 views 常常缺失 ——
这一点必须在评测集里显式处理（见 `schema.AVAILABLE_FIELDS`），
否则会出现"题目本身不可答、却算模型错"的假失败。

用法：
    python -m content_platform.crawlers.douyin --import-file raw.csv --out data/dy.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from .xiaohongshu import _coerce_int, import_file as _generic_import


def import_file(path: str) -> List[dict]:
    rows = _generic_import(path)
    for r in rows:
        r["platform"] = "douyin"
        r["id"] = r["id"].replace("xiaohongshu_", "douyin_")
    return rows


def crawl_api(api_key: str, keyword: str, pages: int = 1) -> List[dict]:
    if not api_key:
        raise SystemExit("需要 --api-key（官方开放平台或授权服务商）")
    raise NotImplementedError(
        "请在 data_providers/ 下按你的服务商实现，输出符合 ContentItem schema。"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--import-file", default=None)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--keyword", default=None)
    ap.add_argument("--out", default="data/douyin_corpus.jsonl")
    args = ap.parse_args()

    if args.import_file:
        rows = import_file(args.import_file)
    elif args.api_key:
        rows = crawl_api(args.api_key, args.keyword or "")
    else:
        raise SystemExit("请指定 --import-file 或 --api-key")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} items -> {args.out}")


if __name__ == "__main__":
    main()
