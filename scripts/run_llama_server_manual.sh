#!/usr/bin/env bash
# 手动启动 llama-server 加载训练后的 Q4_K_M GGUF（用于临时验证，不改动 systemd）
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${REPO_DIR}/data/training_output/dpo_merged-Q4_K_M.gguf"
PORT="${LLAMA_QWEN_PORT:-8001}"

# 使用与 systemd service 相同的预编译运行时（保持与生产环境一致）
RUNTIME_ROOT="${LLAMA_RUNTIME_ROOT:-/mnt/aitools/build/llama-b10217-sm86}"
LLAMA_BIN="$RUNTIME_ROOT/bin/llama-server"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -x "$LLAMA_BIN" ]]; then
  echo "llama-server not found: $LLAMA_BIN" >&2
  echo "Set LLAMA_RUNTIME_ROOT to a llama.cpp build with llama-server." >&2
  exit 1
fi

# 与 service unit 中一致
export LD_LIBRARY_PATH="$RUNTIME_ROOT/bin:/usr/local/cuda-12.4/lib64:/usr/local/cuda-12.4/targets/x86_64-linux/lib"

cd "$RUNTIME_ROOT"

echo "Starting llama-server on 127.0.0.1:$PORT with model $MODEL_PATH"

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
