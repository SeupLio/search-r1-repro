# Search-R1 复现 + 内容平台迁移（verl + SGLang）

> 在线仓库：https://github.com/SeupLio/search-r1-repro ｜ Demo 视频：`demo/search_r1_demo.mp4`

对 [Search-R1](https://github.com/PeterGriffinJin/Search-R1)（arXiv:2503.09516v5, COLM 2025）
的完整复现工程，并把它迁移到 **B站 / 小红书 / 抖音** 内容平台问答场景。

本仓库包含四件事：

| # | 目标 | 产物 |
|---|---|---|
| 1 | 复现 Search-R1 训练流程 | `configs/` `scripts/` `searchr1_repro/` |
| 2 | 对比曲线 + 失败案例归因 | `analysis/` `results/figures/` `eval/failure_attribution.py` |
| 3 | 迁移到内容平台 + 自建评测集 | `content_platform/` `data/content_eval.jsonl` |
| 4 | 复现报告 + 公众号推文 | `report/` |

---

## 先看这两份文件

- **`report/复现报告.md`** — 对比图表 + 偏差分析（差多少、为什么差）
- **`report/公众号推文.md`** — 面向读者的复盘

**⚠️ 关于数据可信度，请先读这段。** 本工程区分三类数字，绝不混用：

| 类型 | 来源 | 标记 |
|---|---|---|
| 论文值 | 从 PDF 表格逐条录入 | `analysis/paper_numbers.py`，图上标 "论文 Table X" |
| 重建曲线 | 依据论文 figure 坐标轴 + 文字描述重建的**示意曲线** | 文件带 `provenance=reconstructed_from_paper`，图上打红色水印 |
| 本地实跑 | 本仓库真实执行的 pipeline（dry-run / 自建评测集） | `results/dryrun/`、`results/failure_attribution/` |

## 第二轮补充：真机实测（RTX 3060 6GB，CUDA）

本机配备 **NVIDIA GeForce RTX 3060 Laptop GPU（6GB）**，已装 PyTorch 2.5.1+cu121，
两项实测已完成：

**① E5 稠密检索 recall@k 自检**（`analysis/recall_selfcheck.py`，196 题，真实 E5 权重）

| 语料规模 | 稀疏 TF-IDF R@3 | E5 稠密 R@3 |
|---|---|---|
| 45 | 0.990 | 0.980 |
| 500 | 0.668 | **0.689** |
| 5000 | **0.658** | 0.648 |

→ 关键修正：此前"top-3 召回 98.98%、检索几乎不漏"是 **45 条微型语料的假象**；
5000 条规模下召回仅约 0.65。**并且 E5 在元数据密集的中文短文本上并不优于词法检索。**

**② 真机 GRPO 训练**（`searchr1_repro/grpo_train.py`，Qwen2.5-0.5B-Instruct + LoRA）

3 个真实训练步跑通，step 3 出现 `useful_groups=1`（组内方差 → 真实梯度更新，loss=-0.98）。
即：多轮 rollout + 检索 token mask + EM 奖励 + 分组相对优势，**整条 RL 闭环在真机 GPU 上验证通过**。

> ⚠️ **已知限制**：本机 GPU 存在驱动级不稳定，`model.generate` 单进程连续调用约 6 次后硬崩
> （换 dtype / 关梯度检查点 / 开 KV cache / 周期 empty_cache 均无效），因此**多 seed 长训练曲线无法在此产出**。
> 训练器已做每步增量落盘，换到稳定 GPU 环境后直接跑即可：
> ```bash
> python -m searchr1_repro.grpo_train --seed 0 --steps 500
> ```

**模型权重来源**：huggingface.co 在受限网络下不可达，故提供
`scripts/download_model_modelscope.py` 从 ModelScope 拉取等价权重（文件格式与 HF 一致，
`from_pretrained` 可直接加载），已验证 Qwen2.5-0.5B-Instruct 与 multilingual-e5-base。

真机跑完后，把
`scripts/07_export_run_logs.py` 导出的 CSV 放进 `results/runs/`，
`analysis/make_figures.py` 会自动用真实曲线替换掉重建曲线，其余代码零改动。

---

## 30 秒跑通（不需要 GPU）

```bash
python -m content_platform.build_corpus     --out data/content_corpus.jsonl   # 45 篇内容
python -m content_platform.build_evalset    --out data/content_eval.jsonl     # 212 道题
python -m content_platform.retrieval_server --self-test                       # 检索服务协议自检
python -m eval.run_dryrun                                                     # 四系统对比 + 归因
python -m eval.export_failure_cases                                           # 失败案例库
python -m analysis.make_figures                                               # 10 张图
python -m analysis.deviation                                                  # 偏差预算
```

跑完会得到 `results/figures/*.png`、`results/dryrun/summary.json`、
`results/failure_attribution/cases_by_type.md`。

## 真机复现（需要 8×H100 或等效算力）

```bash
bash scripts/00_env_setup.sh                        # 装环境（searchr1 + retriever 两套）
bash scripts/01_download_wiki.sh /data/wiki         # wiki-18 语料 + E5 flat 索引
python scripts/02_prepare_data.py                   # NQ + HotpotQA -> parquet
bash scripts/03_launch_retrieval_server.sh wiki     # 检索服务 :8000
bash scripts/04_train_ppo.sh                        # PPO / Qwen2.5-3B
bash scripts/05_train_grpo.sh                       # GRPO / Qwen2.5-7B
bash scripts/06_eval_all.sh checkpoints/.../global_step_500
python scripts/07_export_run_logs.py --source wandb --entity <you>
python -m analysis.make_figures
```

版本与硬件要求见 **`ENV.md`**。

---

## 目录结构

```
search-r1-repro/
├── ENV.md                          版本锁定 / 硬件下限 / 快速自检
├── configs/
│   ├── search_r1_ppo_qwen2.5-3b.yaml    逐条对齐论文 Appendix B.2（PPO）
│   ├── search_r1_grpo_qwen2.5-7b.yaml   GRPO，group size = 5
│   └── search_tool_config.yaml          检索工具配置（URL 可切维基/内容平台）
├── searchr1_repro/                 复现核心
│   ├── template.py                     论文 Table 1 模板（逐字对齐）
│   ├── rollout.py                      Algorithm 1 多轮 rollout + 桩策略
│   ├── loss_mask.py                    retrieved token loss masking
│   ├── reward.py                       EM reward（论文口径）+ 内容平台扩展
│   ├── retriever.py                    HttpRetriever / 零依赖本地检索器
│   └── verl_tool.py                    verl multi-turn tool 接口实现
├── content_platform/               内容平台迁移
│   ├── schema.py                       B站/小红书/抖音统一 schema
│   ├── seed_data.py                    种子语料（合成，可替换）
│   ├── build_corpus.py                 转成 Search-R1 语料格式
│   ├── build_evalset.py                自建评测集 ContentSearch-Bench
│   ├── retrieval_server.py             协议与原版一致，端口 8010
│   ├── run_content_eval.py             在自建集上评测 checkpoint
│   └── crawlers/                       bilibili（可直抓）/ 小红书 / 抖音（合规路径）
├── eval/
│   ├── failure_attribution.py          9 类失败归因分类器
│   ├── run_dryrun.py                   无 GPU 端到端验证 + 四系统对比
│   ├── export_failure_cases.py         失败案例库（含完整轨迹）
│   └── eval_em.py                      统一评测入口
├── analysis/
│   ├── paper_numbers.py                论文全部表格数字（金标准）
│   ├── paper_curves.py                 论文曲线重建（打水印）
│   ├── deviation.py                    偏差因子分解
│   └── make_figures.py                 10 张图
├── scripts/                        00~07 全流程脚本
├── data/                           语料与评测集
└── results/                        figures / dryrun / failure_attribution / ...
```

---

## 三条核心设计说明

**1. 为什么区分 `RolloutState.prompt` 和 `RolloutState.text`？**
论文 Table 1 的模板里**字面包含** `<information>` / `</information>` token。
如果用 `context.count("<information>")` 判断"检索了几轮"，初始值就是 1；
用 `re.findall("<information>(.*?)</information>", context)` 抽检索内容，
第一个"文档"会是模板里的那句话。这个坑会让多轮状态机直接错位。

**2. 为什么检索服务协议要和原版一模一样？**
`searchr1_repro/verl_tool.py` 和 `retriever.py` 都只依赖
`POST /retrieve {"queries":[...], "topk":n}` → `{"result":[[{"document":{...}}]]}`。
所以从维基迁移到内容平台时，只改一个 URL（8000 → 8010），**训练代码一行不动**。

**3. 为什么不能用内置 `hash()` 做"确定性"分配？**
Python 3 对 `str` 的 `hash()` **每进程随机加盐**（`PYTHONHASHSEED`）。
用它把样本分配到不同行为模式，表面上"有 seed 所以可复现"，
实际上每次跑出来的分布都不一样——而且**不报错**。
本仓库统一用 `searchr1_repro.rollout.stable_hash()`（`zlib.crc32`）。
这个坑是在做失败归因时踩到的：两次运行 `correct` 占比 42.9% vs 45.8%。

**4. 为什么失败归因要排在"总分"前面？**
EM 总分不告诉你下一步该改什么。归因分类器的判定顺序刻意设计为：
先摘干净工程问题（格式 → 预算）→ 再看检索（召回 → query 质量）
→ 再看忠实性 → 最后才归给推理能力。
顺序错了，所有问题都会被笼统归成"模型不行"。

---

## 引用

```bibtex
@article{jin2025search,
  title={Search-r1: Training llms to reason and leverage search engines with reinforcement learning},
  author={Jin, Bowen and Zeng, Hansi and Yue, Zhenrui and Yoon, Jinsung and Arik, Sercan and Wang, Dong and Zamani, Hamed and Han, Jiawei},
  journal={arXiv preprint arXiv:2503.09516},
  year={2025}
}
```
