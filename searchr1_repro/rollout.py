# -*- coding: utf-8 -*-
"""Search-R1 多轮 rollout（论文 Algorithm 1）。

核心循环：
    while b < B:
        生成直到 </search> / </answer> / <eos>
        if 出现 </search>:  抽出 query -> 检索 -> 追加 <information>d</information>
        elif 出现 </answer>: 结束
        else: 追加 "My action is not correct. Let me rethink."
        b += 1

⚠️ 关键坑（本项目踩过，值得写进报告）
------------------------------------------------------------------
论文 Table 1 的模板里**字面包含** `<information>` 和 `</information>` 两个 token
（"it will return the top searched results between <information> and </information>"）。
因此：
  * 不能用 `context.count("<information>")` 判断"已经检索了几轮"——初始就是 1；
  * 不能用 `re.findall("<information>(.*?)</information>", context)` 抽取检索内容——
    第一个"文档"会是从模板字面量开始的一段废话。
必须只在 **generated 文本** 上做统计。这也是本项目显式区分
`RolloutState.prompt` 与 `RolloutState.text` 的原因。

本文件同时提供 StubPolicy —— 一个不依赖 GPU 的确定性"桩策略"，
用于在无算力环境下端到端验证整条链路。
真机训练时把 StubPolicy 换成 verl 的 SGLang/vLLM rollout 即可。
"""
from __future__ import annotations

import random
import zlib
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from .template import build_prompt, extract_answer, extract_search_query


def stable_hash(s: str) -> int:
    """跨进程稳定的 32 位哈希。

    ⚠️ 不要用内置 hash()：Python 3 对 str 的哈希每进程随机加盐
    （PYTHONHASHSEED），会让"确定性策略"每次跑出不同结果，
    复现报告里的分布数字就对不上了。这是本仓库踩过的一个真实坑。
    """
    return zlib.crc32(str(s).encode("utf-8")) & 0xFFFFFFFF


@dataclass
class RolloutConfig:
    max_action_budget: int = 4          # 论文 B=4
    topk: int = 3                       # 论文 top-k=3
    max_response_length: int = 500      # 论文 max_response_length=500
    max_retrieved_tokens: int = 500     # 论文 retrieved content 截断到 500 tokens
    domain: str = "wiki"

    @classmethod
    def from_paper(cls) -> "RolloutConfig":
        return cls()


@dataclass
class RolloutState:
    """传给 policy 的状态。text 只含"已生成"的部分，不含 prompt。"""
    prompt: str
    text: str = ""
    n_action: int = 0
    n_search: int = 0

    @property
    def full(self) -> str:
        return self.prompt + self.text

    @property
    def n_info(self) -> int:
        """已插入的 <information> 块数（只在生成文本上统计，避开模板字面量）。"""
        return self.text.count("<information>")


@dataclass
class Trajectory:
    question: str
    prompt: str
    segments: List[dict] = field(default_factory=list)
    n_search: int = 0
    n_valid_search: int = 0
    finished: bool = False
    text: str = ""
    queries: List[str] = field(default_factory=list)

    @property
    def answer(self) -> Optional[str]:
        return extract_answer(self.text)

    @property
    def response_length(self) -> int:
        """用字符数近似 token 数（dry-run 足够；真机请用 tokenizer）。"""
        return len(self.text)


PolicyFn = Callable[[RolloutState, RolloutConfig], str]

RETHINK = " My action is not correct. Let me rethink."


def run_rollout(policy: PolicyFn,
                retriever,
                question: str,
                cfg: Optional[RolloutConfig] = None) -> Trajectory:
    """执行 Algorithm 1。retriever 需实现 .search(query, topk) -> str。"""
    cfg = cfg or RolloutConfig.from_paper()
    prompt = build_prompt(question, cfg.domain)
    traj = Trajectory(question=question, prompt=prompt)
    state = RolloutState(prompt=prompt)

    safety = 0
    while state.n_action < cfg.max_action_budget and safety < cfg.max_action_budget * 4:
        safety += 1
        chunk = policy(state, cfg)
        traj.text += chunk
        traj.segments.append({"type": "gen", "text": chunk})
        state.text = traj.text

        if "</search>" in chunk:
            q = extract_search_query(chunk)
            traj.n_search += 1
            state.n_search = traj.n_search
            if not q:
                traj.text += RETHINK
                traj.segments.append({"type": "rethink", "text": RETHINK})
            else:
                docs = retriever.search(q, cfg.topk)
                limit = cfg.max_retrieved_tokens * 4   # 粗近似：1 token ~ 4 chars
                if len(docs) > limit:
                    docs = docs[:limit]
                block = f"<information>{docs}</information>"
                traj.text += block
                traj.segments.append({"type": "information", "text": block})
                traj.n_valid_search += 1
                traj.queries.append(q)
            state.text = traj.text
            state.n_action += 1
            continue

        if "</answer>" in chunk:
            traj.finished = True
            break

        traj.text += RETHINK
        traj.segments.append({"type": "rethink", "text": RETHINK})
        state.text = traj.text
        state.n_action += 1

    return traj


# ---------------------------------------------------------------------------
# 桩策略：无 GPU 时的可复现代理
# ---------------------------------------------------------------------------
BEHAVIORS = ["ideal", "no_search", "over_search", "bad_query", "unfaithful", "format_fail"]


class StubPolicy:
    """确定性桩策略，模拟 RL 训练不同阶段的典型行为。

    ideal        先检索再依据检索结果作答（模拟训练收敛后的 Search-R1）
    no_search    不检索直接作答（模拟 R1-base / RL 极早期）
    over_search  一直检索、耗尽预算仍不作答
    bad_query    检索了但 query 与问题无关
    unfaithful   检索到正确信息但仍按参数记忆作答（faithfulness 失败）
    format_fail  输出不含任何合法 tag（模拟 base 模型早期格式错误）
    """

    def __init__(self,
                 behavior: str = "ideal",
                 seed: int = 0,
                 behavior_mix: Optional[Dict[str, float]] = None,
                 parametric_answer_fn: Optional[Callable[[str], str]] = None,
                 query_fn: Optional[Callable[[str, int], str]] = None):
        assert behavior in BEHAVIORS or behavior_mix, f"unknown behavior {behavior}"
        self.behavior = behavior
        self.behavior_mix = behavior_mix
        self.rng = random.Random(seed)
        self.parametric_answer_fn = parametric_answer_fn or (lambda q: "")
        self.query_fn = query_fn or (lambda q, i: q)

    def _pick(self, question: str) -> str:
        if not self.behavior_mix:
            return self.behavior
        # 用 question 的稳定 hash 做确定性分配，保证同一题跨进程、跨机器行为一致
        r = (stable_hash(question) & 0xFFFFFF) / 0xFFFFFF
        acc = 0.0
        for k, v in sorted(self.behavior_mix.items()):
            acc += v
            if r <= acc:
                return k
        return self.behavior

    def __call__(self, state: RolloutState, cfg: RolloutConfig) -> str:
        n_info = state.n_info
        question = self._recover_question(state.prompt)
        behavior = self._pick(question)

        if behavior == "format_fail":
            return f"<think> {self._think(question, n_info)} </think> I think the answer is obvious."

        if behavior == "no_search":
            return f"<think> I recall this from my own knowledge. </think><answer> {self.parametric_answer_fn(question)} </answer>"

        if behavior == "over_search":
            q = self.query_fn(question, n_info)
            return f"<think> {self._think(question, n_info)} </think><search> {q} (part {n_info + 1}) </search>"

        if behavior == "bad_query":
            if n_info == 0:
                return f"<think> {self._think(question, 0)} </think><search> unrelated topic about weather </search>"
            return f"<think> {self._think(question, n_info)} </think><answer> {self._from_context(state.text)} </answer>"

        if behavior == "unfaithful":
            if n_info == 0:
                return f"<think> {self._think(question, 0)} </think><search> {self.query_fn(question, 0)} </search>"
            return f"<think> {self._think(question, n_info)} </think><answer> {self.parametric_answer_fn(question)} </answer>"

        # ---- ideal ----
        if n_info == 0:
            return f"<think> {self._think(question, 0)} </think><search> {self.query_fn(question, 0)} </search>"
        if n_info == 1:
            return f"<think> Let me verify the key fact with one more targeted search. </think><search> {self.query_fn(question, 1)} </search>"
        return f"<think> {self._think(question, n_info)} </think><answer> {self._from_context(state.text)} </answer>"

    # ---------- helpers ----------
    @staticmethod
    def _recover_question(prompt: str) -> str:
        marker = "Question: "
        i = prompt.rfind(marker)
        if i < 0:
            return ""
        return prompt[i + len(marker):].rstrip(".").strip()

    @staticmethod
    def _think(question: str, step: int) -> str:
        if step == 0:
            return f"I need to find out the answer to: {question[:60]}".strip()
        return "Based on the retrieved information, I can now synthesize the answer."

    @staticmethod
    def _from_context(text: str) -> str:
        seg = text.split("<information>")[-1]
        seg = seg.split("</information>")[0]
        if not seg.strip():
            return ""
        words = [w for w in seg.replace('"', " ").split() if w]
        return " ".join(words[:12])
