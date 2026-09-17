# -*- coding: utf-8 -*-
"""E5 稠密检索 vs 稀疏 TF-IDF 检索的 recall@k 自检。

为什么做这个：检索器质量是本次复现与论文偏差的**最大单一来源**。
Search-R1 原版用 E5-base + wiki-18 (21M passages) + FAISS；
本仓库 dry-run 用的是纯 Python 的 char n-gram TF-IDF。
两者差多少必须量化，否则报告里所有的 EM 数字都无法解释。

本脚本做的是**受控实验**：
  - 标准答案固定指向同一批"黄金文档"（原始 45 条种子内容）
  - 只改变语料规模（往里塞合成干扰文档 45 → 500 → 5000）
  - 观测 recall@1/3/5 随规模退化的曲线
这样两次运行之间的差异**只**来自检索器本身，不会混淆其他因素。

必须在带 torch 的环境运行（本仓库用 Python 3.12 + CUDA 的 venv）：
    <py312> analysis/recall_selfcheck.py --scales 45 500 5000

输出：results/retrieval/recall_selfcheck.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from searchr1_repro.retriever import ZeroDependencyRetriever, load_corpus  # noqa: E402

CORPUS_DIR = os.path.join(ROOT, "data")
EVAL_PATH = os.path.join(CORPUS_DIR, "content_eval.jsonl")
OUT_PATH = os.path.join(ROOT, "results", "retrieval", "recall_selfcheck.json")
E5_DIR = os.path.join(ROOT, "models", "multilingual-e5-base")

TOPK = (1, 3, 5)


# --------------------------------------------------------------------------
# E5 稠密检索
# --------------------------------------------------------------------------
class E5Retriever:
    """multilingual-e5-base 稠密检索器（transformers 直加载，无需 sentence-transformers）。

    E5 系列强制要求前缀：文档 "passage: "，查询 "query: "，否则效果显著下降。
    """

    def __init__(self, corpus: Sequence[dict], model_dir: str = E5_DIR,
                 batch_size: int = 64, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir).to(self.device).eval()
        self.ids: List[str] = []
        texts: List[str] = []
        for row in corpus:
            contents = row.get("contents", "")
            self.ids.append(str(row.get("id")))
            texts.append("passage: " + contents.replace("\n", " "))
        self.emb = self._encode(texts, batch_size)

    @staticmethod
    def _mean_pool(last_hidden, mask):
        mask = mask.unsqueeze(-1).float()
        return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def _encode(self, texts: Sequence[str], batch_size: int) -> torch.Tensor:
        outs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = self.tok(batch, max_length=256, padding=True,
                           truncation=True, return_tensors="pt").to(self.device)
            with torch.no_grad():
                h = self.model(**enc).last_hidden_state
            e = self._mean_pool(h, enc["attention_mask"])
            e = torch.nn.functional.normalize(e, p=2, dim=1)
            outs.append(e.cpu())
            if i % (batch_size * 20) == 0:
                print(f"    encoded {i}/{len(texts)}", flush=True)
        return torch.cat(outs, dim=0)

    def search_ids(self, query: str, topk: int = 5) -> List[str]:
        q = self._encode(["query: " + query], 1)          # (1, d)
        scores = (self.emb @ q.T).squeeze(1)              # (N,)
        k = min(topk, len(self.ids))
        idx = torch.topk(scores, k).indices.tolist()
        return [self.ids[i] for i in idx]


# --------------------------------------------------------------------------
# 评测
# --------------------------------------------------------------------------
def recall_at_k(retriever_ids_fn, questions: List[dict], ks: Sequence[int]) -> Dict[str, float]:
    hits = {k: 0 for k in ks}
    total = 0
    for q in questions:
        gold = set(q.get("supporting_docs") or [])
        if not gold:
            continue
        total += 1
        got = retriever_ids_fn(q["question"], max(ks))
        for k in ks:
            if set(got[:k]) & gold:
                hits[k] += 1
    return {f"recall@{k}": (hits[k] / total if total else 0.0) for k in ks} | {"n_eval": total}


def load_eval(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # 只保留可答题：不可答题的 supporting_docs 为空，无法算 recall
    return [r for r in rows if r.get("answerable") and (r.get("supporting_docs") or [])]


def corpus_path(scale: int) -> str:
    return os.path.join(CORPUS_DIR, f"corpus_{scale}.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", type=int, nargs="+", default=[45, 500, 5000])
    ap.add_argument("--eval", default=EVAL_PATH)
    ap.add_argument("--out", default=OUT_PATH)
    ap.add_argument("--no-dense", action="store_true", help="跳过 E5（仅测稀疏）")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device = {device}")
    if device == "cuda":
        print(f"GPU   = {torch.cuda.get_device_name(0)}")

    questions = load_eval(args.eval)
    print(f"评测题数（可答且有 gold doc）: {len(questions)}")

    results = {"device": device,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
               "n_questions": len(questions),
               "scales": {}}

    for scale in args.scales:
        path = corpus_path(scale) if scale != 45 else os.path.join(CORPUS_DIR, "content_corpus.jsonl")
        if not os.path.exists(path):
            print(f"[skip] 语料不存在: {path}")
            continue
        corpus = load_corpus(path)
        print(f"\n=== corpus size = {len(corpus)} ({os.path.basename(path)}) ===")

        print("  稀疏 char-ngram TF-IDF ...", flush=True)
        sparse = ZeroDependencyRetriever(corpus)
        r_sparse = recall_at_k(lambda q, k: [i for i, _ in sparse.search_raw(q, k)], questions, TOPK)
        print(f"    {r_sparse}")

        entry = {"corpus_size": len(corpus), "sparse_tfidf": r_sparse}
        results["scales"][str(scale)] = entry

        if not args.no_dense:
            print("  E5 dense ...", flush=True)
            dense = E5Retriever(corpus)
            r_dense = recall_at_k(lambda q, k: dense.search_ids(q, k), questions, TOPK)
            print(f"    {r_dense}")
            entry["e5_dense"] = r_dense
            for k in TOPK:
                key = f"recall@{k}"
                entry[f"delta_{key}"] = r_dense.get(key, 0) - r_sparse.get(key, 0)
            del dense
            if device == "cuda":
                torch.cuda.empty_cache()

        del sparse

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n写入 {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
