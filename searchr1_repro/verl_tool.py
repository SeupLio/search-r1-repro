# -*- coding: utf-8 -*-
"""verl multi-turn tool 接口的检索工具实现。

对应 Search-R1 原版的 `search_r1/search/retrieval_server.py` + `llm_agent/generation.py` 中的
search 调用逻辑。为了让"迁移到内容平台"时零改动，这里把检索后端做成可配置的 URL：

    维基场景:  http://127.0.0.1:8000/retrieve   (E5 + wiki-18)
    内容平台:  http://127.0.0.1:8010/retrieve   (content_platform/retrieval_server.py)

健壮性增强（论文没有，但线上必须）：
  * retry + on_error 降级，避免检索服务抖动直接毁掉一整条 rollout；
  * dedup_by_doc_id：多轮检索很容易重复召回，重复内容会白占 max_retrieved_tokens；
  * 记录 doc id，供后续失败归因计算 recall@k。
"""
from __future__ import annotations

import json
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple


class SearchTool:
    def __init__(self, config: Optional[dict] = None, **kwargs):
        cfg = dict(config or {})
        cfg.update(kwargs)
        self.url: str = cfg.get("retrieval_service_url", "http://127.0.0.1:8000/retrieve")
        self.topk: int = int(cfg.get("topk", 3))
        self.max_retrieved_tokens: int = int(cfg.get("max_retrieved_tokens", 500))
        self.timeout: int = int(cfg.get("timeout", 30))
        self.retry: int = int(cfg.get("retry", 2))
        self.dedup: bool = bool(cfg.get("dedup_by_doc_id", True))
        self.on_error: str = cfg.get("on_error", "return_empty")
        self._seen: set = set()

    # ---- verl tool 接口 ----
    def get_openai_tool_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": "search",
                "description": ("Query the search engine for external knowledge. "
                                "Returns the top passages."),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "search query"}},
                    "required": ["query"],
                },
            },
        }

    def execute(self, query: str, **_) -> Tuple[str, Dict[str, Any]]:
        """返回 (给模型看的文本, 供统计/归因用的元信息)。"""
        docs = self._call(query)
        if self.dedup:
            fresh = [(i, t) for i, t in docs if i not in self._seen]
            # 全被去过重则退回原始结果（宁可重复也不要空上下文）
            docs = fresh or docs
            for i, _t in docs:
                self._seen.add(i)
        text = self._format(docs)
        meta = {"query": query, "doc_ids": [i for i, _ in docs], "n_docs": len(docs)}
        return text, meta

    def reset(self):
        """每条 trajectory 开始前调用，清空去重状态。"""
        self._seen = set()

    # ---- 内部 ----
    def _call(self, query: str) -> List[Tuple[str, str]]:
        payload = json.dumps({"queries": [query], "topk": self.topk,
                              "return_scores": False}).encode()
        last_err = None
        for _ in range(max(1, self.retry + 1)):
            try:
                req = urllib.request.Request(
                    self.url, data=payload, headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    res = json.loads(resp.read().decode())
                out = []
                for item in res["result"][0]:
                    doc = item["document"]
                    contents = doc.get("contents", "")
                    title, _, text = contents.partition("\n")
                    out.append((str(doc.get("id", "")), f'"{title.strip().strip(chr(34))}"\n{text}'))
                return out
            except Exception as e:      # noqa: BLE001
                last_err = e
        if self.on_error == "raise":
            raise RuntimeError(f"retrieval service failed: {last_err}")
        return []

    def _format(self, docs: List[Tuple[str, str]]) -> str:
        parts, budget = [], self.max_retrieved_tokens * 4   # 1 token ~ 4 chars 粗估
        for i, (_id, contents) in enumerate(docs, start=1):
            title, _, text = contents.partition("\n")
            chunk = f"Doc {i}(Title: {title}) {text}"
            if sum(len(p) for p in parts) + len(chunk) > budget:
                break
            parts.append(chunk)
        return " ".join(parts)


# 供 rag / 离线评测复用的轻量入口
def retrieve(query: str, url: str = "http://127.0.0.1:8000/retrieve", topk: int = 3) -> List[str]:
    tool = SearchTool({"retrieval_service_url": url, "topk": topk, "dedup_by_doc_id": False})
    text, _ = tool.execute(query)
    return re.findall(r"Doc \d+\(Title: [^)]*\)", text)
