#!/usr/bin/env bash
# 在 7 个 benchmark + 自建内容平台评测集上评测训练出的 checkpoint
set -euo pipefail
CKPT="${1:-checkpoints/search_r1_ppo/global_step_500}"
OUTDIR="${2:-results/eval}"
mkdir -p "$OUTDIR"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate searchr1
cd "$(dirname "$0")/.."

# 1) 起一个 SGLang 推理服务（评测阶段不需要训练框架）
#    注意端口与训练时的 rollout 服务错开
python -m sglang.launch_server \
  --model-path "$CKPT" \
  --port 30000 \
  --mem-fraction-static 0.85 &
SGLANG_PID=$!
trap 'kill $SGLANG_PID 2>/dev/null || true' EXIT
sleep 60   # 等权重加载；生产环境请改成轮询 /health

# 2) 7 个 benchmark（in-domain: NQ/HotpotQA；out-of-domain: 其余 5 个）
for DS in nq_test hotpotqa_validation triviaqa_validation popqa_test \
          2wiki_validation musique_validation bamboogle; do
  echo "== eval $DS =="
  python eval/eval_em.py \
    --predictions "results/preds/${DS}.jsonl" \
    --out "$OUTDIR/${DS}.json" \
    --dump-wrong "$OUTDIR/${DS}_wrong.jsonl" || echo "[skip] $DS 无预测文件"
done

# 3) 内容平台自建评测集
echo "== eval ContentSearch-Bench =="
python eval/run_content_eval.py \
  --ckpt "$CKPT" \
  --eval data/content_eval.jsonl \
  --retrieval-url http://127.0.0.1:8010/retrieve \
  --out "$OUTDIR/content_bench.json"

# 4) 汇总成一张表（与论文表格同结构，便于直接对比）
python - <<'PY'
import json, os, glob
out = {}
for p in glob.glob("results/eval/*.json"):
    if p.endswith("_wrong.jsonl"):
        continue
    try:
        d = json.load(open(p, encoding="utf-8"))
        if isinstance(d, dict) and "EM" in d:
            out[os.path.basename(p).replace(".json", "")] = d["EM"]
    except Exception:
        pass
json.dump(out, open("results/eval/summary_all.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(json.dumps(out, ensure_ascii=False, indent=2))
PY
