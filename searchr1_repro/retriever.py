# -*- coding: utf-8 -*-
"""检索器抽象。

线上/真机复现时使用 search_r1 原版的 HTTP 检索服务（E5 + FAISS，见 scripts/02_*.sh）；
本地 dry-run / 单测 / 内容平台小规模语料时使用 ZeroDependencyRetriever，
它是纯标准库实现的 char n-gram TF-IDF + cosine，
对中英文都可用，无需 faiss / torch / 模型下载。

两者通过同一个 .search(query, topk) 接口暴露，rollout 代码无需改动。
"""
from __future__ import annotations

import json
import math
import re
import urllib.request
from typing import Dict, Iterable, List, Sequence, Tuple


class Retriever:
    """检索器基类：search(query, topk) -> List[str]（已格式化为 Doc i(Title: "...") ...）"""

    def search(self, query: str, topk: int = 3) -> List[str]:
        raise NotImplementedError

    # ---- 与 Search-R1 原版一致的返回文本格式 ----
    @staticmethod
    def format_docs(docs: Sequence[Tuple[str, str]]) -> str:
        parts = []
        for i, (title, text) in enumerate(docs, start=1):
            # 语料里 contents = '"' + title + '"\n' + text，切分后要去掉残留引号，
            # 否则会出现 Doc 1(Title: ""xxx"")，下游按 title 去重时会全部折叠成一条。
            title = (title or "").strip().strip('"').strip()
            parts.append(f'Doc {i}(Title: "{title}") {text}')
        return " ".join(parts)


class HttpRetriever(Retriever):
    """对接 search_r1/search/retriever_server.py 启动的服务（默认 127.0.0.1:8000）。"""

    def __init__(self, endpoint: str = "http://127.0.0.1:8000/retrieve", timeout: int = 30):
        self.endpoint = endpoint
        self.timeout = timeout

    def search(self, query: str, topk: int = 3) -> List[str]:
        payload = json.dumps({"queries": [query], "topk": topk, "return_scores": False}).encode()
        req = urllib.request.Request(
            self.endpoint, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            res = json.loads(resp.read().decode())
        # 原版返回 {"result": [[{document: {...}, score: ...}, ...]]}
        out = []
        for item in res["result"][0]:
            doc = item["document"]
            contents = doc.get("contents", "")
            title, _, text = contents.partition("\n")
            out.append((title, text or contents))
        return self.format_docs(out)


class ZeroDependencyRetriever(Retriever):
    """纯标准库 char n-gram TF-IDF 检索器（dry-run / 小规模语料用）。

    注意：它不等价于 E5，只是让整条 pipeline（rollout -> reward -> 归因 -> 出图）
    在没有 GPU / 模型权重时也能真实跑通。真机复现请务必切到 HttpRetriever。
    """

    def __init__(self, corpus: Sequence[dict], ngram: int = 2, id_field: str = "id",
                 contents_field: str = "contents"):
        self.ngram = ngram
        self.docs: List[Tuple[str, str]] = []      # (title, text)
        self.ids: List[str] = []
        self.tf: List[Dict[str, int]] = []
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.norm: List[float] = []

        for i, row in enumerate(corpus):
            doc_id = str(row.get(id_field, i))
            contents = row.get(contents_field, "")
            title, _, text = contents.partition("\n")
            self.ids.append(doc_id)
            self.docs.append((title, text or contents))
            toks = self._tokens(title + " " + (text or contents))
            tf: Dict[str, int] = {}
            for t in toks:
                tf[t] = tf.get(t, 0) + 1
            self.tf.append(tf)
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1

        n = max(len(self.tf), 1)
        for t, d in self.df.items():
            self.idf[t] = math.log((n - d + 0.5) / (d + 0.5) + 1.0)
        for tf in self.tf:
            vec = {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in tf.items()}
            self.norm.append(math.sqrt(sum(v * v for v in vec.values())) or 1.0)
            tf.clear()
            tf.update(vec)   # 就地换成 tf-idf 向量

    def _tokens(self, text: str) -> List[str]:
        text = re.sub(r"\s+", "", str(text).lower())
        if not text:
            return []
        if len(text) < self.ngram:
            return [text]
        return [text[i:i + self.ngram] for i in range(len(text) - self.ngram + 1)]

    def _vectorize(self, query: str) -> Dict[str, float]:
        toks = self._tokens(query)
        tf: Dict[str, int] = {}
        for t in toks:
            tf[t] = tf.get(t, 0) + 1
        vec = {t: (1 + math.log(c)) * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, topk: int = 3) -> List[str]:
        q = self._vectorize(query)
        scores = []
        for i, tfidf in enumerate(self.tf):
            s = 0.0
            if len(q) < len(tfidf):
                for t, v in q.items():
                    if t in tfidf:
                        s += v * tfidf[t]
            else:
                for t, v in tfidf.items():
                    if t in q:
                        s += v * q[t]
            scores.append((s / self.norm[i], i))
        scores.sort(reverse=True)
        picked = [(self.docs[i][0], self.docs[i][1]) for _, i in scores[:topk]]
        return self.format_docs(picked)

    def search_raw(self, query: str, topk: int = 3) -> List[Tuple[str, float]]:
        """返回 (doc_id, score)，供检索质量诊断（recall@k）使用。"""
        q = self._vectorize(query)
        scores = []
        for i, tfidf in enumerate(self.tf):
            s = 0.0
            if len(q) < len(tfidf):
                for t, v in q.items():
                    if t in tfidf:
                        s += v * tfidf[t]
            else:
                for t, v in tfidf.items():
                    if t in q:
                        s += v * q[t]
            scores.append((self.ids[i], s / self.norm[i]))
        scores.sort(key=lambda x: -x[1])
        return scores[:topk]


def load_corpus(path: str) -> List[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
