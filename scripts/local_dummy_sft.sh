#!/usr/bin/env bash
# 停推理服务 → dummy SFT → 重启推理服务（RTX 3090 本地 QLoRA 验证）
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="${LLAMA_SERVICE_NAME:-llama-qwen38-27b}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-32B}"
MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-1024}"
VENV_DIR="${VENV_DIR:-${REPO_DIR}/.venv-train}"
SFT_DATASET="${SFT_DATASET:-data/artifacts/instruct_datasets.json}"
OUTPUT_DIR="${OUTPUT_DIR:-data/training_output}"
IS_DUMMY="${IS_DUMMY:-true}"

log() { printf '\033[1;36m[local-sft]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[local-sft]\033[0m %s\n' "$*" >&2; exit 1; }

cd "$REPO_DIR"

# --- 0. 前置检查 ---
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Is the GPU driver installed?"
nvidia-smi || die "nvidia-smi failed"

free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
if (( free_gb < 20 )); then
  die "Only ${free_gb}GB free disk. Need ~20GB+ for dummy SFT (120GB+ for full 32B merge+GGUF)."
fi
log "Free disk: ${free_gb}GB"

# --- 1. 停止推理服务释放显存 ---
if systemctl list-unit-files | grep -q "^${SERVICE_NAME}\.service"; then
  log "Stopping ${SERVICE_NAME} ..."
  sudo systemctl stop "$SERVICE_NAME" || die "Failed to stop ${SERVICE_NAME}"
  sleep 3
  nvidia-smi || true
else
  log "Service ${SERVICE_NAME} not found; skipping stop (make sure GPU is free)."
fi

# --- 2. 训练 venv ---
if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  log "Creating venv at ${VENV_DIR} ..."
  python3 -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

if ! python -c "import unsloth" >/dev/null 2>&1; then
  log "Installing training requirements ..."
  pip install -r llm_engineering/model/finetuning/requirements.txt
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
log "HF_ENDPOINT=${HF_ENDPOINT}"

# --- 3. dummy / 全量 SFT ---
DUMMY_FLAG=()
if [[ "$IS_DUMMY" == "true" ]]; then
  DUMMY_FLAG=(--is_dummy)
  log "Running DUMMY SFT (400 samples, 1 epoch) ..."
else
  log "Running FULL SFT (3 epochs) ..."
fi

python llm_engineering/model/finetuning/run_local.py \
  --finetuning_type sft \
  --base_model "$BASE_MODEL" \
  --max_seq_length "$MAX_SEQ_LENGTH" \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --use_local_datasets \
  --local_sft_dataset_path "$SFT_DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --report_to none \
  "${DUMMY_FLAG[@]}"

log "SFT finished. Merged model: ${OUTPUT_DIR}/sft_merged/"

# --- 4. 重启推理服务 ---
if systemctl list-unit-files | grep -q "^${SERVICE_NAME}\.service"; then
  log "Starting ${SERVICE_NAME} ..."
  sudo systemctl start "$SERVICE_NAME"
  sleep 2
  systemctl is-active --quiet "$SERVICE_NAME" && log "${SERVICE_NAME} is active" || die "${SERVICE_NAME} failed to start"
else
  log "Service ${SERVICE_NAME} not found; skipping start."
fi

log "Done. Next steps: see doc/local-training-workflow.md §5 (DPO) → §6 (GGUF) → §7 (deploy)."
