# -*- coding: utf-8 -*-
"""B站内容抓取（走公开 Web 接口，无需登录即可拿到主要字段）。

为什么只有 B站 能直接抓：
  * B站的搜索/详情接口在未登录状态下即可返回结构化 JSON（首次请求拿 buvid3 cookie 即可）；
  * 小红书、抖音的核心接口需要 `x-s` / `a_bogus` 等签名参数，实现签名属于逆向范畴，
    且高频抓取违反平台规则——本项目不提供，改用"官方开放平台 + 人工导出"两条路径
    （见 xiaohongshu.py / douyin.py）。

合规提醒（请务必遵守）：
  * 只抓公开可访问的数据，频率控制在 1 QPS 以内；
  * 不抓取用户隐私字段（手机号、IP、私信）；
  * 遵守平台 robots.txt 与服务条款，商业用途需走官方开放平台授权；
  * 抓到的数据仅用于本地研究/评测，不要二次分发。

用法：
    python -m content_platform.crawlers.bilibili --keyword "手机测评" --pages 3 \
        --out data/content_corpus.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
import urllib.request

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

SEARCH_API = "https://api.bilibili.com/x/web-interface/wbi/search/type"
VIEW_API = "https://api.bilibili.com/x/web-interface/view"


def _get(url: str, params: dict, cookie: str = "", retry: int = 2) -> dict:
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{qs}", headers={
        "User-Agent": UA,
        "Referer": "https://www.bilibili.com/",
        **({"Cookie": cookie} if cookie else {}),
    })
    last = None
    for _ in range(retry + 1):
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                data = json.loads(r.read().decode("utf-8"))
            if data.get("code") != 0:
                raise RuntimeError(f"api code={data.get('code')} msg={data.get('message')}")
            return data
        except Exception as e:          # noqa: BLE001
            last = e
            time.sleep(1.5)
    raise RuntimeError(f"bilibili api failed: {last}")


def get_buvid() -> str:
    """首次访问首页拿 buvid3，缺失会导致搜索接口返回 -412（风控）。"""
    req = urllib.request.Request("https://www.bilibili.com/", headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            cookies = r.headers.get_all("Set-Cookie") or []
        for c in cookies:
            if "buvid3=" in c:
                return c.split(";")[0]
    except Exception:                   # noqa: BLE001
        pass
    return ""


def strip_html(s: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", s or "")


def crawl(keyword: str, pages: int = 1, cookie: str = "", sleep: float = 1.2) -> list[dict]:
    from ..schema import ContentItem

    items: list[ContentItem] = []
    for page in range(1, pages + 1):
        data = _get(SEARCH_API, {
            "search_type": "video", "keyword": keyword, "page": page,
            "page_size": 30, "order": "totalrank",
        }, cookie=cookie)
        for v in data.get("data", {}).get("result", []):
            try:
                bvid = v.get("bvid") or v.get("aid")
                if not bvid:
                    continue
                pub = v.get("pubdate") or 0
                items.append(ContentItem(
                    id=f"bilibili_{bvid}",
                    platform="bilibili",
                    title=strip_html(v.get("title", "")),
                    author=v.get("author", ""),
                    publish_date=(time.strftime("%Y-%m-%d", time.localtime(pub)) if pub else ""),
                    topic=v.get("typename", "unknown"),
                    views=int(v.get("play") or 0),
                    likes=int(v.get("like") or 0),
                    collects=int(v.get("favorites") or 0),
                    comments=int(v.get("review") or 0),
                    duration_sec=_parse_duration(v.get("duration", "")),
                    note=strip_html(v.get("description", ""))[:200],
                    url=f"https://www.bilibili.com/video/{bvid}",
                ))
            except Exception:           # noqa: BLE001
                continue
        print(f"[bilibili] keyword='{keyword}' page={page} got={len(items)}")
        time.sleep(sleep)               # 限速，务必别去掉
    return [i.to_dict() for i in items]


def _parse_duration(s) -> int:
    try:
        parts = [int(x) for x in str(s).split(":")]
        if len(parts) == 2:
            return parts[0] * 60 + parts[1]
        if len(parts) == 3:
            return parts[0] * 3600 + parts[1] * 60 + parts[2]
    except Exception:                   # noqa: BLE001
        pass
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keyword", required=True)
    ap.add_argument("--pages", type=int, default=1)
    ap.add_argument("--out", default="data/bilibili_corpus.jsonl")
    ap.add_argument("--append", action="store_true", help="追加到已有文件")
    args = ap.parse_args()

    cookie = get_buvid()
    print("[bilibili] buvid cookie:", "ok" if cookie else "missing (可能触发 -412)")
    rows = crawl(args.keyword, args.pages, cookie)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    mode = "a" if args.append else "w"
    with open(args.out, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} items -> {args.out}")


if __name__ == "__main__":
    main()
