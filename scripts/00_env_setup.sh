#!/usr/bin/env bash
# Search-R1 复现环境安装（严格按论文 + Search-R1 官方 README）
# 用法: bash scripts/00_env_setup.sh
set -euo pipefail

echo "==> [1/3] 创建 RL 训练环境 (searchr1)"
conda create -y -n searchr1 python=3.9
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate searchr1

# torch：论文用 2.4.0 + cu121
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# rollout 引擎：论文用 vLLM 0.6.3。
# 本复现改用 SGLang；若你要先对齐论文数字，把下面这行换成 vllm==0.6.3
pip install "vllm==0.6.3"
# pip install "sglang[all]"      # <- 改用 SGLang 时打开

# Search-R1 本体（内含 verl 0.3.x，必须用它自带的，不要另装最新 verl）
cd "$(dirname "$0")/.."
if [ ! -d "Search-R1" ]; then
  git clone https://github.com/PeterGriffinJin/Search-R1.git
fi
cd Search-R1
pip install -e .
pip3 install flash-attn --no-build-isolation
pip install wandb

echo "==> [2/3] 创建检索环境 (retriever)，建议独立以免 faiss 版本冲突"
conda create -y -n retriever python=3.10
conda activate retriever
conda install -y pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install transformers datasets pyserini uvicorn fastapi
conda install -y -c pytorch -c nvidia faiss-gpu=1.8.0

echo "==> [3/3] 完成。后续步骤："
cat <<'EOF'
  # 下载 wiki-18 语料与 e5 flat 索引
  bash scripts/01_download_wiki.sh
  # 处理 NQ + HotpotQA 训练/测试数据
  python scripts/02_prepare_data.py
  # 起检索服务（另开一个终端）
  bash scripts/03_launch_retrieval_server.sh
  # 训练（PPO / GRPO 二选一）
  bash scripts/04_train_ppo.sh
  bash scripts/05_train_grpo.sh
  # 评测 7 个 benchmark
  bash scripts/06_eval_all.sh
  # 把 wandb 曲线导成 csv 供出图
  python scripts/07_export_run_logs.py --entity <your_wandb_entity>
  # 出图（会把 results/runs/*.csv 自动当作真实数据）
  python -m analysis.make_figures
EOF
