# -*- coding: utf-8 -*-
"""内容平台检索服务。

**协议与 Search-R1 原版 retriever_server.py 完全一致**，所以
`searchr1_repro/verl_tool.py` 和 `searchr1_repro/retriever.py` 一行都不用改，
只把 URL 从 8000 换成 8010 即可完成场景迁移。

    请求  POST /retrieve
          {"queries": ["..."], "topk": 3, "return_scores": false}
    响应  {"result": [[{"document": {"id": "...", "contents": "..."}, "score": 0.8}, ...]]}

零依赖实现（stdlib only），CPU 即可跑；生产环境把 `ZeroDependencyRetriever`
换成 E5/bge-m3 + FAISS 即可，接口不变。

用法：
    python -m content_platform.retrieval_server --corpus data/content_corpus.jsonl --port 8010
    # 自检
    python -m content_platform.retrieval_server --corpus data/content_corpus.jsonl --self-test
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from searchr1_repro.retriever import ZeroDependencyRetriever, load_corpus  # noqa: E402

_RETRIEVER = None
_CORPUS_BY_ID = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):      # 静音默认访问日志
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/health"):
            self._json({"status": "ok", "n_docs": len(_CORPUS_BY_ID)})
        else:
            self._json({"error": "use POST /retrieve"}, 404)

    def do_POST(self):
        if not self.path.startswith("/retrieve"):
            self._json({"error": "not found"}, 404)
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n).decode("utf-8"))
        except Exception:                   # noqa: BLE001
            self._json({"error": "bad json"}, 400)
            return

        queries = payload.get("queries") or []
        topk = int(payload.get("topk", 3))
        return_scores = bool(payload.get("return_scores", False))

        result = []
        for q in queries:
            hits = _RETRIEVER.search_raw(q, topk)
            row = []
            for doc_id, score in hits:
                doc = _CORPUS_BY_ID.get(doc_id, {})
                item = {"document": {"id": doc_id, "contents": doc.get("contents", "")}}
                if return_scores:
                    item["score"] = float(score)
                row.append(item)
            result.append(row)
        self._json({"result": result})


def serve(corpus_path: str, host: str, port: int):
    global _RETRIEVER, _CORPUS_BY_ID
    corpus = load_corpus(corpus_path)
    _CORPUS_BY_ID = {str(r["id"]): r for r in corpus}
    _RETRIEVER = ZeroDependencyRetriever(corpus)
    print(f"[retrieval] {len(corpus)} docs indexed, serving on {host}:{port}")

    # 可选：后台预热，避免首个请求慢
    def warm():
        _RETRIEVER.search_raw("warmup", 3)
    threading.Thread(target=warm, daemon=True).start()

    ThreadingHTTPServer((host, port), Handler).serve_forever()


def self_test(corpus_path: str, port: int, query: str = "手机测评 播放量"):
    """起服务 -> 打一发请求 -> 校验协议 -> 退出。用于 CI / 排障。"""
    t = threading.Thread(target=serve, args=(corpus_path, "127.0.0.1", port), daemon=True)
    t.start()
    import time
    time.sleep(1.0)
    payload = json.dumps({"queries": [query], "topk": 3, "return_scores": True}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/retrieve", data=payload,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        res = json.loads(r.read().decode())
    assert "result" in res and res["result"], f"bad response: {res}"
    assert "document" in res["result"][0][0], "missing document field"
    print(f"[self-test] OK. query='{query}' ->")
    for it in res["result"][0]:
        c = it["document"]["contents"].split("\n")[0]
        print(f"   score={it.get('score', 0):.4f}  {c[:60]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/content_corpus.jsonl")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--query", default="手机测评 播放量")
    args = ap.parse_args()

    if args.self_test:
        self_test(args.corpus, args.port, args.query)
    else:
        serve(args.corpus, args.host, args.port)


if __name__ == "__main__":
    main()
