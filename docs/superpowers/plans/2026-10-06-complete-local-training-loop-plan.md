# 本地训练闭环补全实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` (inline execution in this session).

**Goal:** 在 RTX 3090 上完成 SFT 之后的剩余步骤：DPO 训练 → GGUF 转换/量化 → 临时启动验证服务 → 端点评估 → 文档更新，最终把可复现的 GGUF 交接给用户部署进 systemd 推理服务。

**Architecture:** 复用已有的 `run_local.py` 训练入口和 llama.cpp 工具链；对 pinned 的 systemd service，采用“临时手动启动验证 + 提供生产部署命令”的折中方案，全程无需 sudo。

**Tech Stack:** Python 3.11, unsloth, transformers, trl, torch 2.11+cu129, llama.cpp (prebuilt at `/mnt/aitools/src/llama.cpp-ddd4ec1428a6201e18975ea52b07c71e0f9aef26/`).

## Global Constraints

- GPU: RTX 3090 24GB；DPO 必须 `max_seq_length=1024`，`per_device_train_batch_size=1`
- 无 sudo：不能修改 `/usr/local/lib/voxsign/run_llama_qwen38_27b.sh` 或 systemctl 配置
- 所有训练命令在 `.venv-train` 中执行
- 输出目录约定：`data/training_output/`
- 训练基座：本地 `/home/ubuntu/vllm/Qwen3-8B`

---

## Task 1: DPO 训练

**Files:**
- 无新增/修改（复用 `llm_engineering/model/finetuning/run_local.py`）

**Interfaces:**
- 输入：`data/training_output/sft_merged/`, `data/artifacts/preference_datasets.json`
- 输出：`data/training_output/dpo_merged/`

- [ ] **Step 1: 前置检查**

```bash
source .venv-train/bin/activate
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
ls -ld data/training_output/sft_merged
ls -l data/artifacts/preference_datasets.json
```

**Expected:** GPU memory used < 100 MiB, `sft_merged` 存在，preference 数据集存在。

- [ ] **Step 2: 运行 DPO 训练**

```bash
source .venv-train/bin/activate
export PYTHONPATH=.
python llm_engineering/model/finetuning/run_local.py \
  --finetuning_type dpo \
  --base_model /home/ubuntu/vllm/Qwen3-8B \
  --max_seq_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --use_local_datasets \
  --local_dpo_dataset_path data/artifacts/preference_datasets.json \
  --sft_merged_model_dir data/training_output/sft_merged \
  --output_dir data/training_output \
  --report_to none \
  --is_dummy
```

**Expected:** 训练完成，终端最后输出 `DPO complete. Merged model saved to ...`。

- [ ] **Step 3: 验证输出目录**

```bash
ls -lh data/training_output/dpo_merged/
du -sh data/training_output/dpo_merged/
```

**Expected:** 存在 `model-0000x-of-00005.safetensors` 分片 + config/tokenizer，总大小约 16 GB。

- [ ] **Step 4: 验证 loss 收敛**

```bash
grep -E "'loss'|train_loss" /home/ubuntu/.cursor/projects/home-ubuntu-amplifier-git-LLM-Engineers-Handbook/terminals/*.txt 2>/dev/null | tail -30
```

或读取训练日志中最后几行 loss，确认最终 loss 低于初始 loss。

- [ ] **Step 5: 提交训练产物元信息**

> 注：实际模型文件在 `.gitignore` 中，只提交脚本/文档变更。

---

## Task 2: DPO 合并模型快速推理验证（可选但推荐）

**Files:**
- 无新增/修改

- [ ] **Step 1: 用已有脚本加载 dpo_merged 生成回复**

```bash
source .venv-train/bin/activate
python scripts/verify_sft_model.py \
  --model_dir data/training_output/dpo_merged \
  --dtype 4bit \
  --max_new_tokens 128
```

**Expected:** 输出流畅中文回复，无加载报错。

---

## Task 3: GGUF 转换与量化

**Files:**
- 无新增/修改

**Interfaces:**
- 输入：`data/training_output/dpo_merged/`
- 输出：`data/training_output/dpo_merged-F16.gguf`, `data/training_output/dpo_merged-Q4_K_M.gguf`

- [ ] **Step 1: 转换 F16 GGUF**

> 磁盘调整：repo 根目录仅剩 ~38GB，F16 GGUF 约 16.4GB，量化后还要约 5GB。为留出安全余量，F16 中间文件写入 `/mnt/aitools/tmp/`（该挂载点有 600GB+ 空闲），最终 Q4_K_M 再落回 `data/training_output/`。

```bash
mkdir -p /mnt/aitools/tmp
LLAMA_CPP=/mnt/aitools/src/llama.cpp-ddd4ec1428a6201e18975ea52b07c71e0f9aef26
python "$LLAMA_CPP/convert_hf_to_gguf.py" \
  data/training_output/dpo_merged \
  --outfile /mnt/aitools/tmp/dpo_merged-F16.gguf \
  --outtype f16
```

**Expected:** 命令退出码 0，生成 `/mnt/aitools/tmp/dpo_merged-F16.gguf`。

- [ ] **Step 2: 量化 Q4_K_M**

> 实际发现 `/mnt/aitools/src/llama.cpp-.../build/bin/llama-quantize` 不存在；使用 `/mnt/aitools/llama-dflash-pr/build/bin/llama-quantize` 完成量化。

```bash
LLAMA_QUANTIZE=/mnt/aitools/llama-dflash-pr/build/bin/llama-quantize
"$LLAMA_QUANTIZE" \
  /mnt/aitools/tmp/dpo_merged-F16.gguf \
  data/training_output/dpo_merged-Q4_K_M.gguf \
  Q4_K_M
```

**Expected:** 命令退出码 0，生成 `data/training_output/dpo_merged-Q4_K_M.gguf`。

- [ ] **Step 3: 清理 F16 中间文件并验证 Q4_K_M 大小**

```bash
rm -f /mnt/aitools/tmp/dpo_merged-F16.gguf
ls -lh data/training_output/dpo_merged-Q4_K_M.gguf
df -h .
```

**Expected:** Q4_K_M ~4–5 GB，repo 根目录剩余空间明显回升。

---

## Task 4: 临时 llama-server 验证脚本

**Files:**
- 创建：`scripts/run_llama_server_manual.sh`

**Interfaces:**
- 输入：`data/training_output/dpo_merged-Q4_K_M.gguf`
- 输出：监听 `http://127.0.0.1:8001` 的 OpenAI 兼容服务

- [ ] **Step 1: 查找 llama-server 二进制路径**

```bash
grep -E "llama_bin|runtime_root" /usr/local/lib/voxsign/run_llama_qwen38_27b.sh | head -5
```

**Expected:** 看到 `runtime_root=/mnt/aitools/build/llama-b10217-sm86` 和 `llama_bin=.../llama-server`。

- [ ] **Step 2: 编写手动启动脚本**

创建 `scripts/run_llama_server_manual.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${REPO_DIR}/data/training_output/dpo_merged-Q4_K_M.gguf"
PORT="${LLAMA_QWEN_PORT:-8001}"
RUNTIME_ROOT="${LLAMA_RUNTIME_ROOT:-/mnt/aitools/build/llama-b10217-sm86}"
LLAMA_BIN="$RUNTIME_ROOT/bin/llama-server"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "llama-server not found: $LLAMA_BIN" >&2
  exit 1
fi

export LD_LIBRARY_PATH="$RUNTIME_ROOT/bin:/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/targets/x86_64-linux/lib"

cd "$RUNTIME_ROOT"

exec "$LLAMA_BIN" \
  --model "$MODEL_PATH" \
  --alias qwen3-8b-dpo-q4km \
  --host 127.0.0.1 \
  --port "$PORT" \
  --ctx-size 131072 \
  --parallel 1 \
  --batch-size 4096 \
  --ubatch-size 512 \
  --threads 8 \
  --threads-batch 16 \
  --n-gpu-layers 999 \
  --flash-attn on \
  --cache-type-k q4_0 \
  --cache-type-v q4_0 \
  --load-mode mmap+mlock \
  --no-mmproj \
  --jinja \
  --reasoning off \
  --temp 0.7 \
  --top-p 0.95 \
  --top-k 20 \
  --min-p 0.0 \
  --metrics \
  --slots \
  --no-webui \
  --log-verbosity 2
```

- [ ] **Step 3: 添加执行权限并提交脚本**

```bash
chmod +x scripts/run_llama_server_manual.sh
git add scripts/run_llama_server_manual.sh
# 提交使用 commit-tree（系统 git 不支持 --trailer）
TREE=$(git write-tree)
PARENT=$(git rev-parse HEAD)
COMMIT=$(git commit-tree -p "$PARENT" -m "feat: add manual llama-server launch script for trained GGUF" "$TREE")
git update-ref HEAD "$COMMIT"
```

- [ ] **Step 4: 启动临时服务**

> 在后台运行，避免阻塞当前 shell。

```bash
nohup ./scripts/run_llama_server_manual.sh > /tmp/llama_qwen3_8b_dpo.log 2>&1 &
echo $! > /tmp/llama_qwen3_8b_dpo.pid
sleep 15
```

- [ ] **Step 5: 验证服务在线**

```bash
curl -s http://127.0.0.1:8001/v1/models | head -c 500
```

**Expected:** 返回 JSON，包含 `id: "qwen3-8b-dpo-q4km"`。

- [ ] **Step 6: 测试生成**

```bash
curl -s http://127.0.0.1:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-8b-dpo-q4km",
    "messages": [{"role": "user", "content": "请用一句话介绍本地 QLoRA 微调的优势。"}],
    "max_tokens": 128
  }' | head -c 800
```

**Expected:** 返回 JSON，包含通顺中文回复。

---

## Task 5: 端点评估

**Files:**
- 创建：`scripts/evaluate_local_endpoint.py`

> 原 `llm_engineering/model/evaluation/run_local.py` 会强制 push 结果到 HuggingFace，本地-only 运行 401。新增轻量脚本，直接调用本地端点生成 + 评判，结果落盘。

- [ ] **Step 1: 编写本地评估脚本**

创建 `scripts/evaluate_local_endpoint.py`：

```python
# 加载本地 instruct 数据集
# 调用 OpenAI 兼容端点生成答案
# 调用同一端点做 accuracy / style 评判
# 保存 results 到 data/training_output/local_eval_results.json
```

- [ ] **Step 2: 运行本地评估**

```bash
source .venv-train/bin/activate
python scripts/evaluate_local_endpoint.py --n_samples 10 --max_tokens 256
```

**Expected:** 脚本退出码 0，输出平均 accuracy / style。

---

## Task 6: 文档更新与最终提交

**Files:**
- 修改：`doc/memo-local-training-plan.md`
- 修改：`doc/local-training-workflow.md`

- [ ] **Step 1: 在备忘录中追加 DPO/GGUF/评估实测结果**

在 `doc/memo-local-training-plan.md` 末尾新增“第六部分：本地训练闭环实测结果（2026-10-06）”，包含：
- DPO 命令与最终 loss
- GGUF 文件路径与大小
- 临时服务端点验证结果
- eval 脚本输出摘要

- [ ] **Step 2: 更新 workflow 部署说明**

在 `doc/local-training-workflow.md` §7 补充：
- 可以使用 `scripts/run_llama_server_manual.sh` 做临时验证
- 生产部署需用户 sudo 修改 service 配置或新建 service

- [ ] **Step 3: 提交所有变更**

```bash
git add doc/memo-local-training-plan.md doc/local-training-workflow.md
TREE=$(git write-tree)
PARENT=$(git rev-parse HEAD)
COMMIT=$(git commit-tree -p "$PARENT" -m "docs: record DPO/GGUF/eval results and deployment notes" "$TREE")
git update-ref HEAD "$COMMIT"
```

---

## Task 7: 生产部署交接

**Files:**
- 无代码变更

- [ ] **Step 1: 向用户提供部署命令**

在最终回复中给出：
- 新 GGUF 绝对路径：`/home/ubuntu/amplifier/git/LLM-Engineers-Handbook/data/training_output/dpo_merged-Q4_K_M.gguf`
- 停止临时服务：`kill $(cat /tmp/llama_qwen3_8b_dpo.pid)`
- 更新 systemd service 的示例配置片段
- `sudo systemctl daemon-reload && sudo systemctl restart llama-qwen38-27b`
