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
  --base_model /home/ubuntu/vllm/Qwen3-8B \
  --max_seq_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --use_local_datasets \
  --local_dpo_dataset_path data/artifacts/preference_datasets.json \
  --sft_merged_model_dir data/training_output/sft_merged \
  --output_dir data/training_output \
  --report_to none \
  --is_dummy   # 验证管道；全量 DPO 去掉此参数
```

输出：`data/training_output/dpo_merged/`

## 6. 转 GGUF

> 磁盘注意：repo 根目录若空间紧张（< 50GB），可把 F16 中间文件放到 `/mnt/aitools/tmp/` 等大容量挂载点，量化完再删除。

```bash
# 本机实际路径（convert 与 quantize 可能在不同构建目录）
LLAMA_CPP=/mnt/aitools/src/llama.cpp-ddd4ec1428a6201e18975ea52b07c71e0f9aef26
LLAMA_QUANTIZE=/mnt/aitools/llama-dflash-pr/build/bin/llama-quantize

# SFT 或 DPO 合并 16bit → F16 GGUF
python "$LLAMA_CPP/convert_hf_to_gguf.py" \
  data/training_output/dpo_merged \
  --outfile /mnt/aitools/tmp/dpo_merged-F16.gguf \
  --outtype f16

# 量化 Q4_K_M（与现有推理服务一致）
"$LLAMA_QUANTIZE" \
  /mnt/aitools/tmp/dpo_merged-F16.gguf \
  data/training_output/dpo_merged-Q4_K_M.gguf \
  Q4_K_M

# 清理 F16 中间文件
rm /mnt/aitools/tmp/dpo_merged-F16.gguf
```

## 7. 部署回推理服务

### 7a. 临时验证（不改 systemd）

训练刚完成时，先用独立进程启动验证，确认 GGUF 可用：

```bash
nohup ./scripts/run_llama_server_manual.sh > /tmp/llama_qwen3_8b_dpo.log 2>&1 &
echo $! > /tmp/llama_qwen3_8b_dpo.pid
sleep 15

curl -s http://127.0.0.1:8001/v1/models
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "qwen3-8b-dpo-q4km", "messages": [{"role": "user", "content": "你好"}]}'
```

> 临时服务使用与 `llama-qwen38-27b.service` 相同的 `llama-server` 二进制，但不走 systemctl。

### 7b. 生产部署（需 sudo）

现有 `llama-qwen38-27b.service` 对 GGUF 文件做了 **SHA-256 + 文件大小强校验**，不能直接 `cp` 替换。需要修改启动脚本 `/usr/local/lib/voxsign/run_llama_qwen38_27b.sh` 中的：

- `model_path`
- `expected_model_size`
- `expected_model_sha`

然后：

```bash
sudo systemctl daemon-reload
sudo systemctl restart llama-qwen38-27b
```

或新建一个 service（推荐，避免改旧脚本）：

```bash
sudo cp /etc/systemd/system/llama-qwen38-27b.service /etc/systemd/system/llama-qwen3-8b.service
# 编辑新 service，指向新 GGUF 与新启动脚本
sudo systemctl daemon-reload
sudo systemctl enable --now llama-qwen3-8b
```

## 8. 评估（可选，走 :8001）

原 `llm_engineering/model/evaluation/run_local.py` 强依赖 HuggingFace 推送，本地-only 运行会 401。新增脚本可直接评估本地端点：

```bash
# 确保 :8001 已启动（临时服务或 systemd 服务）
python scripts/evaluate_local_endpoint.py --n_samples 10 --max_tokens 256
```

结果保存到 `data/training_output/local_eval_results.json`，并打印平均 accuracy / style。

如需使用原 HF 流程：

```bash
export OPENAI_BASE_URL=http://127.0.0.1:8001/v1
export LOCAL_MODEL_NAME=qwen3-8b-dpo-q4km
export OPENAI_API_KEY=not-needed
# 需配置 HF token 并确保 workspace 存在
python llm_engineering/model/evaluation/run_local.py --dummy
```

## 9. 故障速查

| 现象 | 处理 |
|------|------|
| SFT OOM | seq 1024 / batch 1 已是保守档；换 Qwen3-8B |
| DPO OOM | 同 SFT；DPO 双序列更吃显存 |
| `HF_ENDPOINT` 未生效 | 确认在训练 venv 同一 shell 里 export |
| 本地端点 404 / model name | `curl :8001/v1/models` 看实际 model id，改 `OPENAI_MODEL_ID` / `LOCAL_MODEL_NAME` |
| 磁盘不足 | `du -sh data/training_output/*`；清理 F16 GGUF（量化完可删） |
