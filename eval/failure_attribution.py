# -*- coding: utf-8 -*-
"""失败案例归因（Failure Case Attribution）。

Search-R1 原文只报告了 EM 总分，没有拆解"错的题为什么错"。
对一个要落地的 Agent 系统来说，总分没用，归因才有用——因为它决定了你下一步
该调检索、调奖励、调模板，还是调模型规模。

归因体系（互斥、按判定优先级从高到低）：
    format_error      输出里既没有合法 <search> 也没有合法 <answer>
    no_search         一次检索都没调就作答（依赖参数记忆）
    over_search       检索次数用尽（B=4）仍未给出答案
    retrieval_miss    检索了，但 top-k 从未召回任何支撑文档  -> 该换/调检索器
    query_poor        检索了，但 query 与问题几乎无词汇重叠    -> query 生成还没学会
    unfaithful        检索命中了支撑文档，但答案在检索文本里找不到 -> 幻觉/不忠实
    aggregation_error 检索命中且答案与检索文本相关，但仍算错（聚合/数值/多跳推理错）
    gold_strict       宽松口径下算对、严格 EM 下算错（评测口径偏差，不是模型错）
    correct           正确

判定顺序刻意设计成"先把工程问题（格式/预算）摘干净，再看检索，再看忠实性，
最后才归给推理能力"，否则所有问题都会被笼统归成"模型不行"。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from searchr1_repro.reward import contains_match, em_any, exact_match
from searchr1_repro.template import ANSWER_TAG

FAILURE_TYPES = [
    "correct",
    "format_error",
    "no_search",
    "over_search",
    "retrieval_miss",
    "query_poor",
    "unfaithful",
    "aggregation_error",
    "gold_strict",
]

FAILURE_DESC = {
    "correct": "答对",
    "format_error": "输出无合法 <search>/<answer> 标签，格式未学会",
    "no_search": "未调用检索即作答，依赖参数记忆",
    "over_search": "检索预算（B=4）耗尽仍未收敛到答案",
    "retrieval_miss": "检索了但 top-k 未召回支撑文档，检索器/索引问题",
    "query_poor": "检索 query 与问题几乎无重叠，query 生成能力不足",
    "unfaithful": "检索命中支撑文档但答案无法由检索内容推出，幻觉",
    "aggregation_error": "检索命中且答案相关，但聚合/数值/多跳推理错误",
    "gold_strict": "语义正确但严格 EM 判错，属评测口径问题",
}


@dataclass
class Attribution:
    case_id: str
    question: str
    failure_type: str
    pred: str
    gold: List[str]
    reward: float
    n_search: int
    hit_docs: List[str] = field(default_factory=list)
    supporting_docs: List[str] = field(default_factory=list)
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.case_id, "question": self.question,
            "failure_type": self.failure_type, "failure_desc": FAILURE_DESC[self.failure_type],
            "pred": self.pred, "gold": self.gold, "reward": self.reward,
            "n_search": self.n_search, "hit_docs": self.hit_docs,
            "supporting_docs": self.supporting_docs, "detail": self.detail,
        }


_TOKEN = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9]+", re.I)


def _terms(text: str) -> Set[str]:
    return set(t.lower() for t in _TOKEN.findall(str(text)) if len(t) > 1)


def query_overlap(question: str, query: str) -> float:
    """query 与问题的词汇重叠率（Jaccard），用于识别 query_poor。"""
    a, b = _terms(question), _terms(query)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def answer_grounded_in(pred: str, retrieved_text: str) -> bool:
    """答案是否"有据可依"：pred 的主要实词是否出现在检索到的文本里。

    这是 faithfulness 的一个廉价但可用的代理指标（不依赖 NLI 模型）。
    """
    pt = _terms(pred)
    if not pt:
        return False
    rt = _terms(retrieved_text)
    hit = len(pt & rt) / len(pt)
    return hit >= 0.6


def attribute(case: dict,
              pred: Optional[str],
              trajectory_text: str,
              n_search: int,
              n_valid_search: int,
              has_answer_tag: bool,
              has_search_tag: bool,
              hit_docs: Sequence[str],
              retrieved_text: str,
              strict_reward: float,
              numeric: bool = False) -> Attribution:
    """对单条样本做归因。"""
    gold: List[str] = case["gold"]
    supporting = set(case.get("supporting_docs", []) or [])
    hit = set(hit_docs or [])
    answerable = case.get("answerable", True)
    q = case["question"]
    pred = pred or ""

    # --- 1. 格式 ---
    if not has_answer_tag and not has_search_tag:
        return Attribution(case["id"], q, "format_error", pred, gold, strict_reward,
                           n_search, sorted(hit), sorted(supporting),
                           "输出中没有可解析的 <search> 或 <answer> 标签")

    # --- 2. 未检索 ---
    if n_valid_search == 0 and has_answer_tag:
        # 不可答题若直接答"信息不足"应算对；其余算 no_search
        if not answerable and _says_insufficient(pred):
            return Attribution(case["id"], q, "correct", pred, gold, strict_reward,
                               n_search, sorted(hit), sorted(supporting), "正确拒答")
        return Attribution(case["id"], q, "no_search", pred, gold, strict_reward,
                           n_search, sorted(hit), sorted(supporting),
                           "未调用检索即给出答案")

    # --- 3. 预算耗尽 ---
    if not has_answer_tag:
        return Attribution(case["id"], q, "over_search", pred, gold, strict_reward,
                           n_search, sorted(hit), sorted(supporting),
                           f"检索 {n_search} 次后仍未产出 <answer>")

    # --- 4. 检索未召回 ---
    if supporting and not (hit & supporting):
        return Attribution(case["id"], q, "retrieval_miss", pred, gold, strict_reward,
                           n_search, sorted(hit), sorted(supporting),
                           "top-k 结果中没有任何支撑文档")

    # --- 5. query 质量 ---
    queries = re.findall(r"<search>(.*?)</search>", trajectory_text, re.S)
    if queries:
        ov = max(query_overlap(q, qq) for qq in queries)
        if ov < 0.05:
            return Attribution(case["id"], q, "query_poor", pred, gold, strict_reward,
                               n_search, sorted(hit), sorted(supporting),
                               f"最佳 query 与问题重叠率仅 {ov:.2f}")

    # --- 6. 正确性判定 ---
    if strict_reward >= 1.0:
        return Attribution(case["id"], q, "correct", pred, gold, strict_reward,
                           n_search, sorted(hit), sorted(supporting), "")

    loose = max(contains_match(pred, g) for g in gold) if gold else 0.0
    if loose >= 1.0:
        return Attribution(case["id"], q, "gold_strict", pred, gold, strict_reward,
                           n_search, sorted(hit), sorted(supporting),
                           "宽松口径（包含匹配）正确，严格 EM 判错")

    # --- 7. 不忠实 ---
    if supporting and (hit & supporting) and not answer_grounded_in(pred, retrieved_text):
        return Attribution(case["id"], q, "unfaithful", pred, gold, strict_reward,
                           n_search, sorted(hit), sorted(supporting),
                           "支撑文档已召回，但答案实词无法在检索文本中找到")

    # --- 8. 其余归为推理/聚合错误 ---
    return Attribution(case["id"], q, "aggregation_error", pred, gold, strict_reward,
                       n_search, sorted(hit), sorted(supporting),
                       "检索命中，但聚合/数值/多跳推理出错")


_INSUFFICIENT = ("信息不足", "无法确定", "不知道", "没有找到", "无法回答", "不确定", "无相关信息")


def _says_insufficient(pred: str) -> bool:
    p = str(pred)
    return any(k in p for k in _INSUFFICIENT)


def summarize(attrs: Sequence[Attribution]) -> Dict[str, float]:
    """输出归因分布（占比）。"""
    n = max(len(attrs), 1)
    counts = {k: 0 for k in FAILURE_TYPES}
    for a in attrs:
        counts[a.failure_type] = counts.get(a.failure_type, 0) + 1
    return {k: counts.get(k, 0) / n for k in FAILURE_TYPES}, counts
