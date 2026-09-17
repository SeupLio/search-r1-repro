# -*- coding: utf-8 -*-
"""真机 GRPO 训练器（Search-R1 的核心 RL 闭环，单卡 6GB 可跑）。

复现的是论文 §3.3 + §3.4 的四件事：
  1. 多轮 rollout：<think> -> <search> -> 环境注入 <information> -> ... -> <answer>
  2. retrieved token masking：环境注入的 <information> 段不计入 loss。
     不屏蔽的话模型会去学"复述检索结果"，而不是"用检索结果推理"，
     这是 Search-R1 相比朴素 RAG 的关键之一。
  3. outcome reward：只用最终答案的 EM，不训 reward model、不加格式奖励
  4. GRPO 分组相对优势：同一 question 采样 G 条轨迹，组内均值/标准差归一化

运行（需 CUDA 环境，本机 RTX 3060 6GB）：
    <py312> -m searchr1_repro.grpo_train --seed 0 --steps 150

显存：6GB 放不下 0.5B 全参 + AdamW 的 fp32 优化器状态（约 6GB），
故默认 --mode lora。这是与论文全参微调的一处显式偏差，已记入偏差表。
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from searchr1_repro.retriever import ZeroDependencyRetriever, load_corpus  # noqa: E402
from searchr1_repro.reward import em_any  # noqa: E402
from searchr1_repro.template import (  # noqa: E402
    INFORMATION_TAG, SEARCH_TAG, build_prompt, extract_answer, extract_search_query,
)


LOGPATH = os.path.join(ROOT, "results", "training", "train_debug.log")


def _log(msg: str) -> None:
    """训练过程日志（落盘，便于排查 Windows 下控制台吞输出/硬崩溃）。"""
    try:
        os.makedirs(os.path.dirname(LOGPATH), exist_ok=True)
        with open(LOGPATH, "a", encoding="utf-8") as f:
            f.write(str(msg) + "\n")
    except Exception:
        pass


@dataclass
class Seq:
    ids: List[int] = field(default_factory=list)
    mask: List[int] = field(default_factory=list)   # 1 = policy 生成，计入 loss
    prompt_len: int = 0
    finished: bool = False
    reward: float = 0.0
    adv: float = 0.0
    n_search: int = 0
    answer: str = ""
    group: int = 0
    gold: Sequence[str] = ()


class SearchEnv:
    def __init__(self, retriever, topk: int = 3):
        self.retriever = retriever
        self.topk = topk

    def retrieve_text(self, query: str) -> str:
        docs = self.retriever.search(query, self.topk)
        return f"{INFORMATION_TAG[0]}{docs}{INFORMATION_TAG[1]}"


class GRPOTrainer:
    def __init__(self, model, tok, env, args, device="cuda"):
        self.model, self.tok, self.env, self.args = model, tok, env, args
        self.device = device
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # ---------------- 生成 ----------------
    @torch.no_grad()
    def _gen_batch(self, seqs: List[Seq], max_new: int) -> None:
        if not seqs:
            return
        maxlen = max(len(s.ids) for s in seqs)
        ii, am = [], []
        for s in seqs:
            pad = [self.pad_id] * (maxlen - len(s.ids))
            ii.append(pad + s.ids)
            am.append([0] * len(pad) + [1] * len(s.ids))
        ii_t = torch.tensor(ii, device=self.device)
        am_t = torch.tensor(am, device=self.device)
        out = self.model.generate(
            input_ids=ii_t, attention_mask=am_t, max_new_tokens=max_new,
            do_sample=True, temperature=self.args.temperature, top_p=self.args.top_p,
            pad_token_id=self.pad_id, eos_token_id=self.tok.eos_token_id,
        )
        for s, row in zip(seqs, out[:, maxlen:]):
            gen = [int(t) for t in row if int(t) != self.pad_id]
            if self.tok.eos_token_id in gen:
                gen = gen[:gen.index(self.tok.eos_token_id) + 1]
            s.ids.extend(gen)
            s.mask.extend([1] * len(gen))

    def _append(self, s: Seq, text: str, is_policy: bool) -> None:
        ids = self.tok(text, add_special_tokens=False)["input_ids"]
        s.ids.extend(ids)
        s.mask.extend([int(is_policy)] * len(ids))

    def _advance(self, s: Seq) -> None:
        """推进一条序列：有 <answer> 就结束，有新的 <search> 就注入检索结果。

        注意：必须用「已注入次数」判定新查询，否则每轮都会把同一个 <search>
        反复注入（extract_search_query 取的是最后一对 tag）。
        """
        text = self.tok.decode(s.ids[s.prompt_len:], skip_special_tokens=True)
        ans = extract_answer(text)
        if ans is not None:
            s.finished, s.answer = True, ans
            return
        n_req = text.count(SEARCH_TAG[0])
        if n_req > s.n_search and s.n_search < self.args.max_turns - 1:
            q = extract_search_query(text)
            if q:
                self._append(s, self.env.retrieve_text(q), False)
                s.n_search += 1

    def rollout(self, prompts, golds) -> List[Seq]:
        seqs = []
        for gi, (p, g) in enumerate(zip(prompts, golds)):
            for _ in range(self.args.group_size):
                ids = self.tok(p, add_special_tokens=False)["input_ids"]
                seqs.append(Seq(ids=list(ids), mask=[0] * len(ids),
                                prompt_len=len(ids), group=gi, gold=g))
        for _ in range(self.args.max_turns):
            active = [s for s in seqs if not s.finished]
            if not active:
                break
            self._gen_batch(active, self.args.gen_tokens)
            for s in active:
                self._advance(s)
        for s in seqs:
            if not s.finished:
                s.answer = extract_answer(
                    self.tok.decode(s.ids[s.prompt_len:], skip_special_tokens=True)) or ""
            s.reward = float(em_any(s.answer, s.gold))
        return seqs

    # ---------------- 训练 ----------------
    def step(self, prompts, golds, opt) -> Dict[str, float]:
        _log("[step] rollout start")
        seqs = self.rollout(prompts, golds)
        _log(f"[step] rollout done n={len(seqs)}")

        groups: Dict[int, List[Seq]] = {}
        for s in seqs:
            groups.setdefault(s.group, []).append(s)
        useful = 0
        for gs in groups.values():
            rs = torch.tensor([s.reward for s in gs], device=self.device)
            sd = rs.std(unbiased=False)
            if float(sd) > 1e-6:
                useful += 1
            adv = (rs - rs.mean()) / (sd + 1e-4)
            for s, a in zip(gs, adv):
                s.adv = float(a)

        maxlen = min(max(len(s.ids) for s in seqs), self.args.max_len)
        B = len(seqs)
        input_ids = torch.full((B, maxlen), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, maxlen), dtype=torch.long)
        mask = torch.zeros((B, maxlen), dtype=torch.float)
        advs = torch.zeros((B, maxlen), dtype=torch.float)
        for i, s in enumerate(seqs):
            ids, m = s.ids[:maxlen], s.mask[:maxlen]
            input_ids[i, :len(ids)] = torch.tensor(ids)
            attn[i, :len(ids)] = 1
            mask[i, :len(m)] = torch.tensor(m, dtype=torch.float)
            advs[i, :len(m)] = torch.tensor(m, dtype=torch.float) * s.adv
        input_ids, attn = input_ids.to(self.device), attn.to(self.device)
        mask, advs = mask.to(self.device), advs.to(self.device)

        # 显存关键：logits 的形状是 (B, L, V)，V=151936。
        # 一次性 log_softmax 成 fp32 需要 ~5GB，6GB 单卡必然 OOM。
        # 因此按 batch 分块前向 + 分块算 logprob，梯度在块间累加。
        denom = mask[:, 1:].sum().clamp(min=1.0)
        chunk = max(1, self.args.loss_chunk)
        loss = input_ids.new_zeros((), dtype=torch.float32)
        _log(f"[step] forward B={B} maxlen={maxlen} chunk={chunk}")
        for i in range(0, B, chunk):
            j = min(i + chunk, B)
            lg = self.model(input_ids=input_ids[i:j],
                            attention_mask=attn[i:j]).logits  # (c, L, V)
            lp = F.log_softmax(lg[:, :-1].float(), dim=-1)
            tlp = lp.gather(-1, input_ids[i:j, 1:].unsqueeze(-1)).squeeze(-1)
            m_ = mask[i:j, 1:]
            a_ = advs[i:j, 1:]
            loss = loss - (a_ * tlp * m_).sum() / denom
            del lg, lp, tlp

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], self.args.clip)
        opt.step()

        return {"loss": float(loss),
                "reward": sum(s.reward for s in seqs) / len(seqs),
                "n_search": sum(s.n_search for s in seqs) / len(seqs),
                "useful_groups": useful, "n_groups": len(groups),
                "resp_len": sum(len(s.ids) - s.prompt_len for s in seqs) / len(seqs)}

    # ---------------- 贪心评测 ----------------
    @torch.no_grad()
    def evaluate(self, questions, n: int = 60) -> Dict[str, float]:
        _log(f"[evaluate] enter n={n}")
        self.model.eval()
        ems, nsr, lens = [], [], []
        for qi, q in enumerate(questions[:n]):
            alloc = torch.cuda.memory_allocated() / 1e9
            reserv = torch.cuda.memory_reserved() / 1e9
            _log(f"[evaluate] q{qi} start alloc={alloc:.2f}GB reserved={reserv:.2f}GB")
            ids = self.tok(build_prompt(q["question"], domain=self.args.domain),
                           add_special_tokens=False)["input_ids"]
            s = Seq(ids=list(ids), mask=[0] * len(ids), prompt_len=len(ids))
            for ti in range(self.args.max_turns):
                _log(f"[evaluate] q{qi} turn{ti} pre-tensor len={len(s.ids)}")
                ii = torch.tensor([s.ids], device=self.device)
                _log(f"[evaluate] q{qi} turn{ti} pre-generate")
                out = self.model.generate(input_ids=ii, max_new_tokens=self.args.gen_tokens,
                                          do_sample=False, pad_token_id=self.pad_id,
                                          eos_token_id=self.tok.eos_token_id)
                _log(f"[evaluate] q{qi} turn{ti} post-generate {tuple(out.shape)}")
                new = [int(t) for t in out[0][len(s.ids):] if int(t) != self.pad_id]
                s.ids.extend(new)
                s.mask.extend([1] * len(new))
                text = self.tok.decode(s.ids[s.prompt_len:], skip_special_tokens=True)
                if extract_answer(text) is not None:
                    break
                n_req = text.count(SEARCH_TAG[0])
                if n_req > s.n_search and s.n_search < self.args.max_turns - 1:
                    sq = extract_search_query(text)
                    if sq:
                        self._append(s, self.env.retrieve_text(sq), False)
                        s.n_search += 1
            ans = extract_answer(self.tok.decode(s.ids[s.prompt_len:],
                                                 skip_special_tokens=True)) or ""
            ems.append(float(em_any(ans, q["gold"])))
            nsr.append(s.n_search)
            lens.append(len(s.ids) - s.prompt_len)
            _log(f"[evaluate] q{qi} done em={ems[-1]} search={s.n_search} len={len(s.ids)}")
        self.model.train()
        return {"eval_em": sum(ems) / len(ems), "eval_n_search": sum(nsr) / len(nsr),
                "eval_resp_len": sum(lens) / len(lens)}


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen2.5-0.5B-Instruct"))
    ap.add_argument("--corpus", default=os.path.join(ROOT, "data", "content_corpus.jsonl"))
    ap.add_argument("--eval", default=os.path.join(ROOT, "data", "content_eval.jsonl"))
    ap.add_argument("--domain", default="content", choices=["content", "wiki"])
    ap.add_argument("--mode", default="lora", choices=["lora", "full"])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--prompts-per-step", type=int, default=8)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--gen-tokens", type=int, default=72)
    ap.add_argument("--max-turns", type=int, default=3)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-len", type=int, default=512,
                    help="训练序列截断长度（越长越吃显存：B*L*151936）")
    ap.add_argument("--loss-chunk", type=int, default=2,
                    help="损失计算的 batch 分块大小，用于规避 logits OOM")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--eval-n", type=int, default=60)
    ap.add_argument("--skip-baseline", action="store_true",
                    help="跳过训练前的基线评测（节省前向次数）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    # 必须保留 KV cache：generate 若关掉 use_cache，会对每个新 token 做一次完整前向，
    # 显存峰值随序列长度二次增长，连续多轮调用会在 Windows/WDDM 上触发驱动级硬崩。
    model.config.use_cache = True
    model.to("cuda")

    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0, bias="none",
                         task_type="CAUSAL_LM",
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                         "gate_proj", "up_proj", "down_proj"])
        model = get_peft_model(model, cfg)
        model.enable_input_require_grads()
        model.print_trainable_parameters()
    # 说明：0.5B + LoRA + 短序列的显存占用约 2GB，6GB 单卡完全不需要梯度检查点；
    # 而且 gradient checkpointing 与 generate 复用同一份模型时，在 Windows/WDDM 上
    # 反复调用会触发驱动级硬崩（无 Python traceback）。故此处刻意不启用。
    model.train()

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    retriever = ZeroDependencyRetriever(load_corpus(args.corpus))
    env = SearchEnv(retriever, topk=args.topk)

    evalset = [json.loads(l) for l in open(args.eval, encoding="utf-8") if l.strip()]
    evalset = [e for e in evalset if e.get("answerable")]
    train_q, eval_q = evalset[:160], evalset[160:]
    print(f"train n={len(train_q)}  held-out eval n={len(eval_q)}", flush=True)

    trainer = GRPOTrainer(model, tok, env, args)
    rng = random.Random(args.seed)

    out = args.out or os.path.join(ROOT, "results", "training", f"grpo_seed{args.seed}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    def save(hist, note=""):
        """每步增量落盘：本机 GPU 在累计数百次前向后会驱动级硬崩，
        增量保存可保证崩溃前的真实训练数据不丢失。"""
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"args": vars(args), "gpu": torch.cuda.get_device_name(0),
                       "baseline": hist[0] if hist else None,
                       "final": hist[-1] if hist else None,
                       "history": hist, "note": note},
                      f, ensure_ascii=False, indent=2)

    hist = []
    if not args.skip_baseline:
        base = trainer.evaluate(eval_q, args.eval_n)
        base["step"] = 0
        print(f"baseline(step0): {base}", flush=True)
        hist.append(base)
        save(hist, "after baseline eval")

    t0 = time.time()
    for step in range(1, args.steps + 1):
        batch = rng.sample(train_q, min(args.prompts_per_step, len(train_q)))
        m = trainer.step([build_prompt(q["question"], domain=args.domain) for q in batch],
                         [q["gold"] for q in batch], opt)
        hist.append({"step": step, "train_reward": m["reward"], "loss": m["loss"],
                     "n_search": m["n_search"], "useful_groups": m["useful_groups"],
                     "n_groups": m["n_groups"], "eval_em": None})
        save(hist, f"after step {step}")
        if step % 5 == 0 or step == 1:
            print(f"step {step:4d} loss {m['loss']:+.4f} reward {m['reward']:.3f} "
                  f"search {m['n_search']:.2f} useful {m['useful_groups']}/{m['n_groups']} "
                  f"len {m['resp_len']:.0f} ({time.time()-t0:.0f}s)", flush=True)
        if step % args.eval_every == 0:
            ev = trainer.evaluate(eval_q, args.eval_n)
            ev["step"] = step
            ev["train_reward"] = m["reward"]
            hist.append(ev)
            save(hist, f"after eval at step {step}")
            print(f"  [eval@{step}] EM {ev['eval_em']:.3f} search {ev['eval_n_search']:.2f}",
                  flush=True)

    save(hist, "completed")
    print(f"\n写入 {out}")
    if hist:
        ems = [h["eval_em"] for h in hist if h.get("eval_em") is not None]
        if len(ems) >= 2:
            print(f"EM: {ems[0]:.3f} -> {ems[-1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
