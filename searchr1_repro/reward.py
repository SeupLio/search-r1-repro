# -*- coding: utf-8 -*-
"""Search-R1 的 outcome reward：论文式 (4) r = EM(a_pred, a_gold)。

论文明确：
  * 只使用 outcome reward，不使用 format reward（"we do not incorporate format rewards"）；
  * 不训练 neural reward model；
  * 指标为 Exact Match。

这里实现论文同款 EM（Yu et al. 2024 风格的 normalize -> 去冠词 -> 去标点 -> 空白折叠），
并提供内容平台场景需要的扩展（数值容差、集合匹配、时效惩罚），
扩展项默认关闭，保证与论文口径一致。
"""
from __future__ import annotations

import re
import string
import unicodedata
from typing import Iterable, List, Sequence

ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)
PUNCT = set(string.punctuation) | {"，", "。", "、", "；", "：", "？", "！", "（", "）",
                                   "《", "》", "“", "”", "‘", "’", "－", "—", "～"}


def normalize_answer(s: str) -> str:
    """论文式 EM 归一化：小写、去标点、去冠词、折叠空白。"""
    if s is None:
        return ""

    def _strip_accents(x: str) -> str:
        x = unicodedata.normalize("NFD", x)
        return "".join(c for c in x if unicodedata.category(c) != "Mn")

    s = str(s).strip().lower()
    s = _strip_accents(s)
    s = "".join(ch for ch in s if ch not in PUNCT)
    s = ARTICLES.sub(" ", s)
    s = " ".join(s.split())
    return s


def exact_match(pred: str, gold: str) -> float:
    """论文 reward：EM，返回 0.0 / 1.0。"""
    return float(normalize_answer(pred) == normalize_answer(gold))


def em_any(pred: str, golds: Sequence[str]) -> float:
    """gold 可能有多个等价别名时取最大（数据集侧的 answer alias）。"""
    if not golds:
        return 0.0
    return max(exact_match(pred, g) for g in golds)


def contains_match(pred: str, gold: str) -> float:
    """宽松口径：gold 出现在 pred 中。仅用于诊断，不作为论文 reward。"""
    p, g = normalize_answer(pred), normalize_answer(gold)
    if not p or not g:
        return 0.0
    return float(g in p or p in g)


# ---------------------------------------------------------------------------
# 内容平台场景的扩展奖励（默认关闭，需显式打开）
# ---------------------------------------------------------------------------
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_UNITS = {"万": 1e4, "亿": 1e8, "千": 1e3, "w": 1e4, "k": 1e3, "m": 1e6}


def _to_number(s: str):
    """把 '12.3万' '1,234' '3.2w' 解析成 float。"""
    if s is None:
        return None
    s = str(s).strip().lower().replace(",", "")
    mult = 1.0
    for u, m in _UNITS.items():
        if s.endswith(u):
            mult, s = m, s[: -len(u)]
            break
    m = _NUM.search(s)
    if not m:
        return None
    try:
        return float(m.group()) * mult
    except ValueError:
        return None


def numeric_tolerance_match(pred: str, gold: str, rel_tol: float = 0.02) -> float:
    """数值型答案（播放量、点赞数、涨粉数）允许 2% 相对误差。

    内容平台问答大量答案是数值，直接 EM 会严重低估（"1234.5万" vs "1.23千万"）。
    """
    a, b = _to_number(pred), _to_number(gold)
    if a is None or b is None:
        return exact_match(pred, gold)
    if b == 0:
        return float(a == 0)
    return float(abs(a - b) / abs(b) <= rel_tol)


def content_reward(pred: str,
                   golds: Sequence[str],
                   gold_is_numeric: bool = False,
                   use_numeric_tol: bool = True,
                   rel_tol: float = 0.02) -> float:
    """内容平台评测用的 outcome reward。

    论文口径（EM）是默认行为；只有 gold_is_numeric=True 时才启用数值容差。
    这样既保证和论文可比，又能反映内容平台数据的特性。
    """
    if gold_is_numeric and use_numeric_tol:
        return max(numeric_tolerance_match(pred, g, rel_tol) for g in golds)
    return em_any(pred, golds)


def compute_reward(predicted: str,
                   gold: Sequence[str] | str,
                   gold_is_numeric: bool = False) -> float:
    """统一入口，供 verl 的 reward function / 离线评测共用。"""
    if isinstance(gold, str):
        gold = [gold]
    return content_reward(predicted, gold, gold_is_numeric=gold_is_numeric)


def batch_em(preds: Iterable[str], golds: Iterable[Sequence[str]]) -> List[float]:
    return [em_any(p, g) for p, g in zip(preds, golds)]
