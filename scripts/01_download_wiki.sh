#!/usr/bin/env bash
# 下载 wiki-18 语料 + E5 flat 索引（Search-R1 官方脚本）
set -euo pipefail

SAVE_PATH="${1:-./data/wiki}"
mkdir -p "$SAVE_PATH"

cd "$(dirname "$0")/../Search-R1"
python scripts/download.py --save_path "$SAVE_PATH"

# 分片合并 + 解压（官方 README 原样）
cat "$SAVE_PATH"/part_* > "$SAVE_PATH/e5_Flat.index"
gzip -d "$SAVE_PATH/wiki-18.jsonl.gz"

echo "OK. 语料: $SAVE_PATH/wiki-18.jsonl"
echo "索引:   $SAVE_PATH/e5_Flat.index"
echo
echo "自检："
wc -l "$SAVE_PATH/wiki-18.jsonl" | awk '{print "  段落数(行): "$1}'
du -sh "$SAVE_PATH/e5_Flat.index" | awk '{print "  索引大小: "$1}'
head -c 300 "$SAVE_PATH/wiki-18.jsonl"; echo
