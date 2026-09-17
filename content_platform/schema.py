# -*- coding: utf-8 -*-
"""内容平台统一 schema。

三个平台的原始字段差异很大（B站有投币/收藏，小红书有收藏/笔记类型，抖音有分享），
统一成一套 schema 之后，检索、评测、奖励代码才能跨平台复用。

ContentItem:
    id            str   全局唯一，建议 "<platform>_<平台侧ID>"
    platform      str   bilibili | xiaohongshu | douyin
    title         str
    author        str
    publish_date  str   YYYY-MM-DD
    topic         str   话题/分区（人工或规则标注，评测集依赖它）
    views         int   播放量(B站/抖音) / 曝光量(小红书)
    likes         int
    collects      int   收藏(B站:投币数位, 小红书:收藏数)
    comments      int
    shares        int   抖音分享数；其余平台可为 0
    duration_sec  int   视频时长；图文笔记为 0
    note          str   简介/正文摘要
    url           str   可选

⚠️ 字段口径必须写清楚，否则跨平台聚合会得出错误结论：
   - B站的"收藏"与"投币"是两个不同字段，本 schema 的 collects 在 B站语义为"投币数"，
     真实分析时应显式区分；
   - 小红书没有公开"播放量"，只能拿到"曝光量"或干脆拿不到，用 0 表示缺失，
     评测时凡是依赖缺失字段的题目都应判为不可答。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class ContentItem:
    id: str
    platform: str
    title: str
    author: str
    publish_date: str
    topic: str = "unknown"
    views: int = 0
    likes: int = 0
    collects: int = 0
    comments: int = 0
    shares: int = 0
    duration_sec: int = 0
    note: str = ""
    url: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ContentItem":
        allowed = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in allowed})


PLATFORM_CN = {"bilibili": "哔哩哔哩", "xiaohongshu": "小红书", "douyin": "抖音"}
VIEW_NAME = {"bilibili": "播放量", "xiaohongshu": "曝光量", "douyin": "播放量"}
AUTHOR_NAME = {"bilibili": "UP主", "xiaohongshu": "博主", "douyin": "作者"}

# 各平台可稳定获取的字段（用于判断某道题在数据上是否可答）
AVAILABLE_FIELDS = {
    "bilibili":    {"views", "likes", "collects", "comments", "duration_sec", "publish_date", "author"},
    "xiaohongshu": {"likes", "collects", "comments", "publish_date", "author"},
    "douyin":      {"views", "likes", "comments", "shares", "duration_sec", "publish_date", "author"},
}
