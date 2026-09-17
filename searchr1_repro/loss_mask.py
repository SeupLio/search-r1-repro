# -*- coding: utf-8 -*-
"""Retrieved Token Loss Masking —— Search-R1 稳定训练的关键机制。

论文 Section 3.1 / 5.4 / Appendix D：
    检索回来的 <information>...</information> 内容**不是模型生成的**，
    如果把它一起算进 policy loss，等于让模型去"预测"外部文档的 token，
    会引入与任务无关的优化目标，削弱训练稳定性。
    论文 Table 4/6 显示：加 mask 比不加 mask 平均 EM 高 约 +8.8 个百分点
    （7B: 0.431 vs 0.343；3B: 0.303 vs 0.262）。

这里的实现要点：
  * token 级布尔 mask，1 = 参与 loss，0 = 屏蔽；
  * <information> 与 </information> 标签本身也要屏蔽（它们同样是注入的）；
  * 该 mask 同时作用于 policy loss 与 KL loss（论文原文：
    "The retrieved token masking is also applied when calculating the KL divergence loss"）。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

INFORMATION_OPEN = "<information>"
INFORMATION_CLOSE = "</information>"


def find_spans(text: str,
               open_tag: str = INFORMATION_OPEN,
               close_tag: str = INFORMATION_CLOSE) -> List[Tuple[int, int]]:
    """返回 [start, end) 字符区间列表，含标签本身。"""
    spans = []
    i = 0
    while True:
        s = text.find(open_tag, i)
        if s < 0:
            break
        e = text.find(close_tag, s)
        if e < 0:
            spans.append((s, len(text)))
            break
        e += len(close_tag)
        spans.append((s, e))
        i = e
    return spans


def char_mask(text: str) -> List[int]:
    """字符级 mask（用于快速验证 / dry-run）。"""
    mask = [1] * len(text)
    for s, e in find_spans(text):
        for i in range(s, e):
            mask[i] = 0
    return mask


def token_mask_from_offsets(offsets: Sequence[Tuple[int, int]],
                            text: str,
                            prompt_len: int = 0,
                            response_start: int = 0) -> List[int]:
    """token 级 mask。

    offsets: tokenizer(return_offsets_mapping=True) 得到的 (char_start, char_end) 列表，
             字符坐标相对于**完整序列**（prompt + response）。
    prompt_len: prompt 部分 token 数（这些位置一律置 0，由 verl 自己处理 response mask）。
    response_start: response 在 full_text 中的字符偏移。
    """
    spans = find_spans(text)
    mask = [0] * len(offsets)
    for ti, (cs, ce) in enumerate(offsets):
        if ti < prompt_len:
            mask[ti] = 0
            continue
        # 相对 response 的字符坐标
        rs, re_ = cs - response_start, ce - response_start
        if re_ <= 0:
            mask[ti] = 0
            continue
        masked = False
        for s, e in spans:
            # token 与任一 span 有交集即屏蔽（宁可多屏蔽，不要漏）
            if rs < e and re_ > s:
                masked = True
                break
        mask[ti] = 0 if masked else 1
    return mask


def apply_mask_to_loss_mask(loss_mask: List[int], text: str,
                            offsets=None, prompt_len: int = 0,
                            response_start: int = 0) -> List[int]:
    """把 retrieved-token mask 叠加到 verl 原本的 loss_mask 上（按位与）。"""
    if offsets is None:
        cm = char_mask(text)
        return [a * b for a, b in zip(loss_mask, cm)]
    tm = token_mask_from_offsets(offsets, text, prompt_len, response_start)
    n = min(len(loss_mask), len(tm))
    return [loss_mask[i] * tm[i] for i in range(n)]


def mask_ratio(text: str) -> float:
    """被屏蔽字符占比，用于监控检索内容是否挤占了过多序列长度。"""
    if not text:
        return 0.0
    m = char_mask(text)
    return 1.0 - sum(m) / len(m)


if __name__ == "__main__":
    demo = ("<think> I need to look this up. </think>"
            "<search> who is X </search>"
            "<information>Doc 1(Title: \"X\") X is a person. </information>"
            "<think> Now I know. </think><answer> X </answer>")
    print("mask ratio:", round(mask_ratio(demo), 3))
    print("".join(str(c) for c in char_mask(demo)))
