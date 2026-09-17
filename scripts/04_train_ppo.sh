#!/usr/bin/env bash
# Search-R1 / PPO 训练（论文默认 RL 方法）
# 用法: bash scripts/04_train_ppo.sh [config.yaml] [n_gpus]
set -euo pipefail
CONFIG="${1:-configs/search_r1_ppo_qwen2.5-3b.yaml}"
NGPU="${2:-8}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate searchr1
cd "$(dirname "$0")/../Search-R1"

# ---- 检索服务健康检查（论文没提，但 90% 的"训练不涨"是检索没起来）----
if ! curl -sf -X POST http://127.0.0.1:8000/retrieve \
      -H 'Content-Type: application/json' \
      -d '{"queries":["test"],"topk":3,"return_scores":false}' >/dev/null; then
  echo "[FATAL] 检索服务 127.0.0.1:8000 不可用。先跑 scripts/03_launch_retrieval_server.sh wiki"
  exit 1
fi
echo "[ok] 检索服务在线"

export HYDRA_FULL_ERROR=1
export WANDB_PROJECT=search_r1_repro
# 复现性：论文 rollout temperature=1.0，本身就有采样方差，固定 seed 只保证可重启
export SEED="${SEED:-42}"

python3 -m verl.trainer.main_ppo \
  --config-path "$(realpath "$(dirname "$CONFIG")")" \
  --config-name "$(basename "$CONFIG" .yaml)" \
  data.train_files="${TRAIN_FILES:-../data/nq_hotpotqa_train.parquet}" \
  data.val_files="${VAL_FILES:-../data/nq_test.parquet}" \
  data.train_batch_size=512 \
  data.max_prompt_length=4096 \
  data.max_response_length=500 \
  actor_rollout_ref.model.path="${MODEL:-Qwen/Qwen2.5-3B}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
  actor_rollout_ref.actor.ppo_mini_batch_size=256 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=64 \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_turns=4 \
  actor_rollout_ref.rollout.multi_turn.tool_config_path=configs/search_tool_config.yaml \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  critic.optim.lr=1e-5 \
  critic.optim.lr_warmup_steps_ratio=0.015 \
  critic.model.path="${MODEL:-Qwen/Qwen2.5-3B}" \
  critic.ppo_micro_batch_size_per_gpu=64 \
  critic.model.enable_gradient_checkpointing=True \
  algorithm.adv_estimator=gae \
  algorithm.gamma=1.0 \
  algorithm.lam=1.0 \
  trainer.critic_warmup=0 \
  trainer.n_gpus_per_node="$NGPU" \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.total_training_steps=500 \
  trainer.save_freq=100 \
  trainer.test_freq=100 \
  trainer.project_name=search_r1_repro \
  trainer.experiment_name="${EXP_NAME:-ppo_qwen2.5-3b-base}" \
  trainer.logger='["console","wandb"]' \
  trainer.default_local_dir="${CKPT_DIR:-checkpoints/search_r1_ppo}" \
  "$@"
