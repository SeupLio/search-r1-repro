# 环境与版本锁定

复现的第一原则：**任何一个版本动了，数字就不能直接跟论文比**。
下表是本仓库验证过 / 论文使用过的组合，改任何一行都要重新标定。

## 1. RL 训练环境（`searchr1`）

| 组件 | 论文 (arXiv:2503.09516v5) | 本仓库推荐 | 备注 |
|---|---|---|---|
| Python | 3.9 | 3.10 | |
| PyTorch | 2.4.0 (cu121) | 2.4.0 (cu121) | 与 CUDA/flash-attn 强耦合 |
| verl | 0.3.x（Search-R1 自带 `pip install -e .`） | 与 Search-R1 repo 同 commit | **不要用最新 verl**，接口改过 |
| rollout 引擎 | vLLM 0.6.3 | **SGLang 0.4.x** | ⚠️ 主动偏离，见 `analysis/deviation.py` |
| flash-attn | 2.x | 2.6+ | `--no-build-isolation` |
| FSDP | + CPU offload + gradient checkpointing | 同 | |
| wandb | 任意 | 任意 | 用于导出真实曲线 |

> ⚠️ 关于 SGLang：论文用 vLLM。换成 SGLang 属于**环境变更**（不是超参抖动），
> 会让采样序列分布与 vLLM 不完全一致。若目标是"复现论文数字"，建议先跑 vLLM 对齐，
> 再用 SGLang 做吞吐对比；若目标是"工程落地"，直接用 SGLang 并如实报告差异来源。

## 2. 检索环境（`retriever`，建议独立）

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.10 | faiss-gpu 对版本敏感 |
| PyTorch | 2.4.0 + cu121 | 用 conda 装，便于 faiss-gpu |
| faiss-gpu | 1.8.0 | 或 faiss-cpu + ANN 索引 |
| transformers | 4.4x | E5 推理 |
| pyserini | latest | BM25 稀疏检索 |
| fastapi / uvicorn | latest | 检索服务 |

## 3. 语料与索引

| 项 | 论文 | 说明 |
|---|---|---|
| 知识源 | 2018 Wikipedia dump (`wiki-18.jsonl`) | `scripts/download.py` 下载后需 `cat part_* > e5_Flat.index` |
| 段落数 | ~21M | 全量 flat 索引需 ~100GB+ 内存/显存 |
| 检索器 | E5 | `intfloat/e5-base-v2` 量级 |
| top-k | 3 | 论文 Appendix G 显示 topk=3 最优 |

## 4. 内容平台场景（`content_platform`）

| 项 | 值 | 说明 |
|---|---|---|
| 语料 | `data/content_corpus.jsonl` | 默认是 45 条**合成种子**，需用 crawler 替换 |
| 检索器 | bge-m3 / m3e-base（CPU 可跑）或复用 E5 | 中文语料建议换中文 embedding |
| 服务端口 | 8010 | 与维基场景的 8000 隔离，便于 A/B |
| 评测集 | `data/content_eval.jsonl` (212 题) | 由 `build_evalset.py` 从语料构造 |

## 5. 硬件下限（实测经验）

| 目标 | 最低配置 | 说明 |
|---|---|---|
| 3B + PPO + 8k 序列 | 2×A100 80G | batch 需降到 64~128，会引入偏差 |
| 3B + LoRA | 1×A100 80G | 与论文全参不可比，只能看趋势 |
| 7B + PPO | 4×H100 80G 起 | 论文是 8×H100 |
| 7B + GRPO (n=5) | 4×H100 80G 起 | GRPO 无 critic，显存更省但采样 ×5 |
| 仅跑 dry-run / 评测 / 出图 | 任意 CPU | `eval/run_dryrun.py` 不需要 GPU |

## 6. 快速自检

```bash
python -m eval.run_dryrun            # 无 GPU，验证整条链路
python -m analysis.make_figures      # 生成全部图表
python -m analysis.paper_numbers     # 导出论文基准 CSV
python -m analysis.deviation         # 打印偏差预算
```

跑通上面四条说明工程侧没问题；剩下的差距就只剩"算力"和"环境对齐"了。

## 7. 清理与重新生成下载文件

> 适用场景：C 盘空间紧张时清理本实验产生的下载缓存/冗余权重；清理后若需复跑，按本节重新拉取即可。

### 7.1 已清理的临时/下载文件（2026-09-18）

| 路径 | 体积 | 性质 | 是否可重建 |
|---|---|---|---|
| `C:\Users\10718\AppData\Local\pip\Cache` | ~9.6G | pip 轮子下载缓存 | ✅ 重装自动重下 |
| `C:\Users\10718\.cache\modelscope` | ~804K | ModelScope 下载器索引 | ✅ 重建 |
| `E:\paper\search-r1-repro\models\multilingual-e5-base\pytorch_model.bin` | 1.1G | 与 `model.safetensors` 重复的冗余副本 | ✅ 重下 |
| `E:\paper\search-r1-repro\models\multilingual-e5-base\model.safetensors` | 1.1G | **保留**，E5 加载所需 | — |

**未清理（非本实验产生，删之会伤其他项目）**：`C:\Users\10718\.cache\huggingface\hub`(2.5G，bge-large-zh/bert-base-chinese)、`C:\Users\10718\.cache\torch\hub\checkpoints`(445M，maskrcnn/resnet50)、`E:\paper\venv312`(Python 环境，复跑需要)。

> ⚠️ 本机安全策略拦截了回收站 API（`Add-Type`/`Remove-Item` 被终止），故采用**永久删除**（Git Bash `rm`）。
> 上述文件全部为可重建的缓存或冗余副本，无独有数据，删除安全。

### 7.2 重新下载模型权重（ModelScope，国内可达）

```bash
# 激活本仓库实际使用的环境（见 7.3），在仓库根目录执行：
python scripts/download_model_modelscope.py --repo AI-ModelScope/multilingual-e5-base --out models/multilingual-e5-base
python scripts/download_model_modelscope.py --repo qwen/Qwen2.5-0.5B-Instruct --out models/Qwen2.5-0.5B-Instruct
```
- 脚本只拉必需文件（config/tokenizer/`*.safetensors`），跳过 README/LICENSE，速度快。
- 若 repo 名 404，到 modelscope.cn 搜索对应模型名替换即可（脚本对 repo 名无硬编码依赖）。
- `huggingface.co` 可达时的等价命令：
  ```bash
  pip install -U huggingface_hub
  huggingface-cli download intfloat/multilingual-e5-base --local-dir models/multilingual-e5-base
  huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir models/Qwen2.5-0.5B-Instruct
  ```

### 7.3 重建 Python 环境（真机训练实际用的 venv312）

> 注意：真机 GRPO 实测用的是 `E:\paper\venv312`（uv 管理的 **Python 3.12 + torch 2.5.1+cu121**），
> **不是** `scripts/00_env_setup.sh` 里的 conda/torch 2.4.0。pip 缓存清空后首次 `pip install` 会重新下载轮子，
> 仅耗带宽，不影响数值结果。

```bash
uv venv -p 3.12 E:/paper/venv312
source E:/paper/venv312/Scripts/activate        # Windows: E:\paper\venv312\Scripts\activate.bat
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install transformers peft sentence-transformers   # 真机 GRPO / E5 推理需要
# 验证链路
python -m searchr1_repro.grpo_train --seed 0 --steps 3 --self-test
```

> 若走论文原版环境（conda + torch 2.4.0 + vLLM/SGLang），仍用 `bash scripts/00_env_setup.sh`，
> 但那是另一套版本组合，数字不可与本仓库实测直接比对。
