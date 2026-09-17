# -*- coding: utf-8 -*-
"""在文本游戏环境里复现 ReAct / Reflexion，并与 Act-only 基线对比。

对应简历手册里"必读论文"的 ReAct（Yao et al. 2022）与 Reflexion（Shinn et al. 2023），
以及高频面试题「ReAct 为什么比 CoT / 纯 Act 强」——这里给出可跑的实测对比。

三种策略：
  act       : 只输出动作（无显式推理）
  react     : Thought + Action 交替（ReAct）
  reflexion : ReAct 失败后生成一段自我反思，带着反思重试一次（Reflexion）

运行（需 CUDA 环境）：
    <py312> -m game_env.baselines --methods act react reflexion --tasks 8

注意：本机 GPU 在累计数百次前向后会驱动级硬崩，故脚本按 (方法, 任务) 增量落盘，
崩溃前的数据依然有效。
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from game_env.world import World, make_tasks  # noqa: E402

MODEL_DIR = os.path.join(ROOT, "models", "Qwen2.5-0.5B-Instruct")
OUT = os.path.join(ROOT, "results", "game", "react_baselines.json")

ACTS = ("go", "take", "drop", "search", "unlock", "talk", "use", "look", "inventory")

SYS = {
    "act": "你在一个文字冒险世界里。只输出一个动作，不要解释。",
    "react": ("你在一个文字冒险世界里。每一步先写一行 Thought: 说明你的判断，"
              "再写一行 Action: 给出动作。"),
    "reflexion": ("你在一个文字冒险世界里。每一步先写一行 Thought: 说明你的判断，"
                  "再写一行 Action: 给出动作。"),
}


def build_prompt(method: str, task_desc: str, obs: str, memory: str = "") -> str:
    head = SYS[method] + "\n"
    if memory:
        head += f"上一次尝试的反思：{memory}\n"
    head += f"任务：{task_desc}\n当前观测：\n{obs}\n"
    head += ("可用动作：go <方向/房间> | take <物品> | drop <物品> | search | "
             "unlock <方向> with <钥匙> | talk <人名> | use <物品> | look | inventory\n")
    if method == "act":
        head += "Action:"
    else:
        head += "Thought:"
    return head


def parse_action(text: str) -> str:
    """从模型输出里抽出动作行。"""
    for line in text.splitlines():
        s = line.strip()
        low = s.lower()
        if low.startswith("action:"):
            s = s.split(":", 1)[1].strip()
        if any(s.startswith(a) for a in ACTS):
            return s
    return ""


class Runner:
    def __init__(self, model, tok, max_new=40, device="cuda"):
        self.model, self.tok, self.max_new = model, tok, max_new
        self.device = device

    @torch.no_grad()
    def gen(self, prompt: str) -> str:
        """手写贪心采样（不用 generate：本机 generate 调用约 6 次后驱动级硬崩）。"""
        ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        cur = list(ids)
        for _ in range(self.max_new):
            ii = torch.tensor([cur], device=self.device)
            lg = self.model(input_ids=ii).logits[:, -1, :]
            nxt = int(torch.argmax(lg, dim=-1))
            cur.append(nxt)
            if nxt == self.tok.eos_token_id:
                break
        return self.tok.decode(cur[len(ids):], skip_special_tokens=True)


def run_episode(runner, method: str, task: dict, max_steps: int, reflection: bool = False):
    w = World(seed=task["seed"])
    memory = ""
    for _ in range(2):  # Reflexion 最多 2 次尝试
        for _s in range(max_steps):
            prompt = build_prompt(method, w.quest.desc, w.observe(), memory)
            out = runner.gen(prompt)
            act = parse_action(out)
            if not act:
                w.step("look")
                continue
            w.step(act)
            if w.success():
                return True, w.step_count, memory, out
        if not reflection:
            break
        # 失败 -> 生成反思，带着反思重试
        refl_prompt = (
            f"你在游戏里失败了。任务：{w.quest.desc}\n"
            f"你的历史动作：{' | '.join(a for a, _ in w.history)}\n"
            f"请用一句话总结失败原因，并给出下次应该先做什么：")
        memory = runner.gen(refl_prompt).strip()[:200]
        w = World(seed=task["seed"])  # 重置世界重试
    return False, w.step_count, memory, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", default=["act", "react"])
    ap.add_argument("--tasks", type=int, default=4)
    ap.add_argument("--max-steps", type=int, default=6)
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float16)
    model.config.use_cache = True
    model.to("cuda").eval()
    runner = Runner(model, tok, max_new=args.max_new)

    tasks = make_tasks(args.tasks, seed=7)
    results = {"args": vars(args), "gpu": torch.cuda.get_device_name(0), "runs": []}

    def save():
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

    for method in args.methods:
        ok = 0
        for t in tasks:
            succ, steps, mem, _ = run_episode(
                runner, method, t, args.max_steps,
                reflection=(method == "reflexion"))
            ok += int(succ)
            results["runs"].append({"method": method, "task": t["id"], "quest": t["quest"],
                                    "success": bool(succ), "steps": steps,
                                    "reflection": mem[:120]})
            save()
            print(f"{method:10s} {t['id']} succ={succ} steps={steps}", flush=True)
        results.setdefault("summary", {})[method] = {
            "success_rate": ok / len(tasks), "n": len(tasks)}
        save()
        print(f"== {method}: success {ok}/{len(tasks)}", flush=True)

    print(f"\n写入 {args.out}")
    print(json.dumps(results.get("summary", {}), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
