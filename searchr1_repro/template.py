# -*- coding: utf-8 -*-
"""Search-R1 训练/推理模板（论文 Table 1，逐字对齐）。

论文原文 (Table 1):
    Answer the given question. You must conduct reasoning inside <think> and </think>
    first every time you get new information. After reasoning, if you find you lack some
    knowledge, you can call a search engine by <search> query </search>, and it will
    return the top searched results between <information> and </information>. You
    can search as many times as you want. If you find no further external knowledge
    needed, you can directly provide the answer inside <answer> and </answer> without
    detailed illustrations. For example, <answer> xxx </answer>. Question: {question}.
"""

SEARCH_TAG = ("<search>", "</search>")
INFORMATION_TAG = ("<information>", "</information>")
THINK_TAG = ("<think>", "</think>")
ANSWER_TAG = ("<answer>", "</answer>")

_BASE = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack some "
    "knowledge, you can call a search engine by <search> query </search>, and it will "
    "return the top searched results between <information> and </information>. You "
    "can search as many times as you want. If you find no further external knowledge "
    "needed, you can directly provide the answer inside <answer> and </answer> without "
    "detailed illustrations. For example, <answer> xxx </answer>. "
    "Question: {question}."
)

# 迁移到内容平台时的模板：只把 "search engine" 的语义换成"内容平台检索"，
# 结构（think / search / information / answer）保持不变，以保证 RL 行为可比。
_CONTENT = (
    "Answer the given question about content from Bilibili / Xiaohongshu / Douyin. "
    "You must conduct reasoning inside <think> and </think> first every time you get new "
    "information. After reasoning, if you find you lack some knowledge, you can call a "
    "content-platform search engine by <search> query </search>, and it will return the "
    "top searched results between <information> and </information>. You can search as many "
    "times as you want. If you find no further external knowledge needed, you can directly "
    "provide the answer inside <answer> and </answer> without detailed illustrations. "
    "For example, <answer> xxx </answer>. Question: {question}."
)


def build_prompt(question: str, domain: str = "wiki") -> str:
    """构造 RL rollout 的初始 user prompt。

    domain="wiki"    -> 论文原版（NQ/HotpotQA 等）
    domain="content" -> 内容平台迁移版
    """
    tmpl = _CONTENT if domain == "content" else _BASE
    return tmpl.format(question=question.strip())


def extract_between(text: str, open_tag: str, close_tag: str):
    """取最后一对 tag 之间的内容；不存在返回 None。"""
    s = text.rfind(open_tag)
    if s < 0:
        return None
    s += len(open_tag)
    e = text.find(close_tag, s)
    if e < 0:
        return None
    return text[s:e].strip()


def extract_search_query(text: str):
    return extract_between(text, *SEARCH_TAG)


def extract_answer(text: str):
    return extract_between(text, *ANSWER_TAG)
