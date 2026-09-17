#!/usr/bin/env bash
# 起检索服务。维基场景 8000，内容平台场景 8010，互不干扰便于 A/B。
set -euo pipefail
MODE="${1:-wiki}"

cd "$(dirname "$0")/.."

if [ "$MODE" = "wiki" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate retriever
  # Search-R1 原版检索服务
  cd Search-R1/search_r1/search
  echo "== 启动 wiki 检索服务 (E5 + faiss) @ 127.0.0.1:8000 =="
  python retrieval_server.py \
    --config_file_name search_r1/search/retriever_config.yaml
else
  # 内容平台检索服务（纯 CPU 可跑，见 content_platform/retrieval_server.py）
  echo "== 启动内容平台检索服务 @ 127.0.0.1:8010 =="
  python -m content_platform.retrieval_server \
    --corpus data/content_corpus.jsonl \
    --port 8010
fi
