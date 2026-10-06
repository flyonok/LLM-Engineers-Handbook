# 本地训练 → GGUF 部署 分步清单（RTX 3090 + Qwen3-32B / 8B）

> 配套：`doc/memo-local-training-plan.md` §3.9、`llm_engineering/model/finetuning/run_local.py`
> 默认输出目录：`data/training_output/`

## 0. 前置检查

```bash
# GPU 空闲（训练前必须停掉推理服务释放 24GB 显存）
sudo systemctl stop llama-qwen38-27b
nvidia-smi

# 磁盘：32B 合并 16bit 约 64GB + GGUF 中间产物，预留 120GB+
df -h .
```

## 1. 训练环境（独立 venv，勿装进主项目 Poetry）

```bash
cd /home/ubuntu/amplifier/git/LLM-Engineers-Handbook
python3 -m venv .venv-train
source .venv-train/bin/activate
pip install -r llm_engineering/model/finetuning/requirements.txt

# 国内镜像（下载 Qwen3 权重 / 数据集）
export HF_ENDPOINT=https://hf-mirror.com
```

验证 Qwen3-32B 可 4bit 加载：

```python
# python -c 或临时脚本
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-32B",
    max_seq_length=1024,
    load_in_4bit=True,
)
print("OK", type(model).__name__)
```

若 OOM 或加载失败 → 改用 `Qwen/Qwen3-8B`（改 `--base_model` 或 `configs/training.yaml`）。

## 2. 数据集（本地 JSON，已有可跳过）

自带数据已够跑 dummy：

```bash
ls data/artifacts/instruct_datasets.json data/artifacts/preference_datasets.json
# instruct ~2500 条 / preference ~400 条
```

如需重新生成（走本地 llama.cpp 当 GPT 平替）：

```bash
# 先确认 :8001 在线
curl -s http://127.0.0.1:8001/v1/models

# .env 或 shell
export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
export OPENAI_MODEL_ID=local-model
export OPENAI_API_KEY=not-needed

poetry poe run-generate-instruct-datasets-pipeline
poetry poe run-generate-preference-datasets-pipeline
```

## 3. Dummy SFT（验证管道，~400 样本 1 epoch）

一键脚本（停服务 → 建 venv/装依赖 → dummy SFT → 重启服务）：

```bash
./scripts/local_dummy_sft.sh
# 全量 SFT：
IS_DUMMY=false ./scripts/local_dummy_sft.sh
# 换 8B 备选：
BASE_MODEL=Qwen/Qwen3-8B ./scripts/local_dummy_sft.sh
```

手动等价命令：

```bash
sudo systemctl stop llama-qwen38-27b
source .venv-train/bin/activate
export HF_ENDPOINT=https://hf-mirror.com

python llm_engineering/model/finetuning/run_local.py \
  --finetuning_type sft \
  --is_dummy \
  --base_model Qwen/Qwen3-32B \
  --max_seq_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --use_local_datasets \
  --local_sft_dataset_path data/artifacts/instruct_datasets.json \
  --report_to none

sudo systemctl start llama-qwen38-27b
```

成功标志：`data/training_output/sft_merged/` 出现 `model-0000x-of-00005.safetensors` 等分片 + tokenizer/config。

## 4. 全量 SFT

```bash
python llm_engineering/model/finetuning/run_local.py \
  --finetuning_type sft \
  --base_model Qwen/Qwen3-32B \
  --max_seq_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --use_local_datasets \
  --local_sft_dataset_path data/artifacts/instruct_datasets.json \
  --report_to none
```

OOM 时：先试 `--max_seq_length 1024`（默认已是）；仍 OOM 换 `Qwen/Qwen3-8B`。

## 5. DPO（基于 SFT 合并模型）

```bash
python llm_engineering/model/finetuning/run_local.py \
  --finetuning_type dpo \
  --base_model Qwen/Qwen3-32B \
  --max_seq_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --use_local_datasets \
  --local_dpo_dataset_path data/artifacts/preference_datasets.json \
  --sft_merged_model_dir data/training_output/sft_merged \
  --report_to none
```

输出：`data/training_output/dpo_merged/`

## 6. 转 GGUF

```bash
# llama.cpp（本机已装；如未装：git clone --depth 1 https://github.com/ggml-org/llama.cpp && cmake -B build -DGGML_CUDA=on && cmake --build build --config Release -j)
LLAMA_CPP=/path/to/llama.cpp   # 改成你的路径

# SFT 或 DPO 合并 16bit → GGUF
python $LLAMA_CPP/convert_hf_to_gguf.py \
  data/training_output/dpo_merged \
  --outfile data/training_output/dpo_merged-F16.gguf \
  --outtype f16

# 量化 Q4_K_M（与现有推理服务一致）
$LLAMA_CPP/build/bin/llama-quantize \
  data/training_output/dpo_merged-F16.gguf \
  data/training_output/dpo_merged-Q4_K_M.gguf \
  Q4_K_M
```

## 7. 部署回推理服务

```bash
# 停服务 → 备份旧 GGUF → 替换 → 启动
sudo systemctl stop llama-qwen38-27b

# 找到服务当前指向的模型路径（systemd unit 或 llama.cpp 启动脚本）
sudo systemctl cat llama-qwen38-27b | grep -E 'ExecStart|WorkingDirectory'

# 例（按实际路径替换）：
sudo cp /path/to/old/Qwen3.8-27B-Q4_K_M.gguf /path/to/old/Qwen3.8-27B-Q4_K_M.gguf.bak
sudo cp data/training_output/dpo_merged-Q4_K_M.gguf /path/to/models/

sudo systemctl start llama-qwen38-27b
curl -s http://127.0.0.1:8001/v1/models
```

建议后续把服务改名为 `llama-qwen3-32b.service`（备忘录 §3.9，可选）。

## 8. 评估（可选，走 :8001）

```bash
# 部署新 GGUF 后，用同一端点做生成 + judge
export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
export LOCAL_MODEL_NAME=local-model
export OPENAI_API_KEY=not-needed
export DATASET_HUGGINGFACE_WORKSPACE=local
export MODEL_HUGGINGFACE_WORKSPACE=local

python llm_engineering/model/evaluation/run_local.py --dummy   # 或去掉 --dummy
```

## 9. 故障速查

| 现象 | 处理 |
|------|------|
| SFT OOM | seq 1024 / batch 1 已是保守档；换 Qwen3-8B |
| DPO OOM | 同 SFT；DPO 双序列更吃显存 |
| `HF_ENDPOINT` 未生效 | 确认在训练 venv 同一 shell 里 export |
| 本地端点 404 / model name | `curl :8001/v1/models` 看实际 model id，改 `OPENAI_MODEL_ID` / `LOCAL_MODEL_NAME` |
| 磁盘不足 | `du -sh data/training_output/*`；清理 F16 GGUF（量化完可删） |
