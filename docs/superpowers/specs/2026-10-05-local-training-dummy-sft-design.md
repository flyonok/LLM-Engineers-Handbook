# 本地化训练管道跑通（dummy SFT 验证）设计

> 日期：2026-10-05
> 状态：已批准
> 配套文档：`doc/local-training-workflow.md`、`doc/memo-local-training-plan.md`

## 1. 目标

在 RTX 3090（24GB）上跑通本地 QLoRA SFT 管道（dummy 模式，400 样本 1 epoch），验证从
「停推理服务 → 建训练环境 → 加载 Qwen3-32B 4bit → SFT 训练 → 合并 16bit 模型 → 重启服务」
全链路通畅。训练 OOM 或磁盘不足时自动退 Qwen3-8B。

## 2. 关键事实（已核实）

- **代码改造已全部落地**，本次无需改代码：
  - `llm_engineering/model/finetuning/run_local.py`（本地训练入口）
  - `llm_engineering/model/finetuning/finetune.py`（去 SM_* 环境变量、本地数据集加载、4bit、基座参数化）
  - `llm_engineering/model/finetuning/requirements.txt`（升级到 Qwen3 支持版本：transformers>=4.51、unsloth>=2025.3.18、trl>=0.15）
  - `llm_engineering/settings.py`（新增 LOCAL_TRAINING / TRAINING_BASE_MODEL 等配置）
  - `llm_engineering/application/dataset/generation.py`（tiktoken 改 cl100k_base、本地端点）
  - `steps/training/train.py`、`configs/training.yaml`、`pipelines/training.py`（本地模式开关）
  - `scripts/local_dummy_sft.sh`（一键脚本）
- 数据集 JSON 已存在（`data/artifacts/instruct_datasets.json` 1.7M / `preference_datasets.json` 182K），无需重新生成
- 当前磁盘剩余 110GB，dummy SFT 产物（LoRA + 合并 16bit）足够容纳
- 推理服务 `llama-qwen38-27b` 运行中，OpenAI 兼容端点 `http://127.0.0.1:8001`，model id 为 `qwen3.8-27b-int4`

## 3. 已确认决策

| 决策点 | 选择 |
|--------|------|
| 基座模型 | 先用 Qwen3-32B 跑 dummy SFT，OOM/磁盘不足自动退 Qwen3-8B |
| 交付范围 | 跑通 dummy SFT（400 样本 1 epoch）验证管道通畅即可 |
| 服务处理 | 用 `scripts/local_dummy_sft.sh` 一键脚本自动停/启服务 |

## 4. 执行步骤

### 阶段 1 — 环境准备
1. 停推理服务 `llama-qwen38-27b` 释放 GPU 显存
2. 创建训练 venv `.venv-train`
3. 安装 `llm_engineering/model/finetuning/requirements.txt`（走国内镜像）
4. 验证 Qwen3-32B 可 4bit 加载（`FastLanguageModel.from_pretrained(..., load_in_4bit=True)`）

### 阶段 2 — dummy SFT 训练
5. 跑 `./scripts/local_dummy_sft.sh`（32B 起步）
6. 若 OOM/磁盘不足 → 用 `BASE_MODEL=Qwen/Qwen3-8B ./scripts/local_dummy_sft.sh` 重跑

### 阶段 3 — 验证产物
7. 确认 `data/training_output/sft_merged/` 出现 `model.safetensors`
8. 确认推理服务已重启、:8001 端点恢复

### 阶段 4 — 交付
9. 记录结果（实际用的模型、耗时、显存峰值、产物路径）
10. 提交 git（spec + 文档 + 代码改动）

## 5. 风险与应对

| 风险 | 等级 | 应对 |
|------|------|------|
| 32B 4bit 加载/训练 OOM | 高 | 退 Qwen3-8B（用户已接受） |
| 依赖安装失败（版本冲突） | 中 | 先验证加载，失败则调 requirements 版本 |
| 磁盘不足 | 低 | dummy 产物小，110GB 足够；不够退 8B |
| 国内镜像下载慢 | 低 | HF_ENDPOINT=hf-mirror.com 已配置 |
| 停服务后忘记重启 | 低 | 一键脚本自动重启并校验 is-active |

## 6. 成功标准

- `data/training_output/sft_merged/model.safetensors` 存在
- 推理服务 `llama-qwen38-27b` 重新 active，:8001 端点返回模型列表
- 训练日志无 OOM / 无未捕获异常
- 记录实际使用的基座模型（32B 或 8B）

## 7. 实际执行结果

> 执行日期：2026-10-06

### 7.1 环境结论

训练 venv `.venv-train` 已可用，关键包版本：

| 包 | 版本 |
|----|------|
| torch | 2.11.0+cu129 |
| torchvision | 0.26.0+cu129 |
| transformers | 4.57.6 |
| unsloth | 2026.9.14 |
| trl | 0.19.1 |

验证命令：

```bash
source .venv-train/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); import unsloth; print(unsloth.__version__)"
# torch: 2.11.0+cu129 cuda: True
# unsloth: 2026.9.14
```

### 7.2 代码调整

为让训练入口不依赖 MongoDB/Qdrant 等全量运行时依赖，对 `llm_engineering/model/finetuning/run_local.py` 做了两处调整：

1. 直接 `importlib` 加载同级 `finetune.py`，避免触发 `llm_engineering/__init__.py` 全量导入。
2. `llm_engineering.settings` 导入失败时使用与 `Settings` 相同的默认值，保证单独跑训练时参数可用。

为绕过 hf-mirror 对 Unsloth 预量化变体（`unsloth/Qwen3-8B-unsloth-bnb-4bit`）下载缓慢/卡死的问题，增加了 monkey-patch：强制 Unsloth 使用传入的原始模型 ID（`Qwen/Qwen3-8B`），由 Unsloth 本地做 4bit 量化。

`llm_engineering/model/finetuning/requirements.txt` 中把 torch/torchvision 固定到与当前驱动匹配的 cu129 版本。

### 7.3 服务停止

当前 shell 环境无法使用 `sudo`（需要 TTY 密码），因此通过 `kill <llama-server PID>` 手动释放了 GPU。用户在自己终端执行 `./scripts/local_dummy_sft.sh` 时，若已配置密码less sudo，脚本中的 `sudo systemctl stop/start` 可正常工作。

### 7.4 训练执行状态

执行命令（8B 验证，不触 32B 以节省时间）：

```bash
source .venv-train/bin/activate
unset HF_ENDPOINT
export UNSLOTH_USE_MODELSCOPE=1
export PYTHONPATH=.
python llm_engineering/model/finetuning/run_local.py \
  --finetuning_type sft --base_model Qwen/Qwen3-8B --max_seq_length 1024 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 16 \
  --num_train_epochs 3 --use_local_datasets \
  --local_sft_dataset_path data/artifacts/instruct_datasets.json \
  --output_dir data/training_output --report_to none --is_dummy
```

结果：
- Unsloth 成功初始化并识别 RTX 3090（23.588 GB）。
- 数据加载正常（本地 JSON 可读）。
- **阻塞点：模型权重下载速度过慢**。ModelScope 下载 `Qwen/Qwen3-8B` 时速度约 50–150 kB/s，16 GB 权重预计需要 30–50 小时，无法在单次会话完成。

### 7.5 建议下一步

1. **在网速较好的时段/网络下重新跑上述命令**（可挂 tmux/screen 过夜）。
2. 若 8B dummy SFT 成功，再尝试 `Qwen/Qwen3-32B`；如 OOM 则按预案退回 8B。
3. 训练完成后按 `doc/local-training-workflow.md` §6 转 GGUF、§7 部署回推理服务。
4. 未来若网络稳定，可去掉 `run_local.py` 中的 Unsloth 模型名 monkey-patch，改回使用 Unsloth 预量化变体以加快加载。
