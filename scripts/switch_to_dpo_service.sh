#!/usr/bin/env bash
# Cut :8001 from the legacy 27B service to the trained Qwen3-8B-DPO GGUF.
set -euo pipefail

echo "Stopping llama-qwen38-27b ..."
sudo systemctl stop llama-qwen38-27b

echo "Starting llama-qwen3-8b-dpo ..."
sudo systemctl start llama-qwen3-8b-dpo

echo "Waiting for model load ..."
for i in $(seq 1 60); do
  if curl -sf http://127.0.0.1:8001/v1/models >/tmp/qwen3_dpo_models.json 2>/dev/null; then
    break
  fi
  sleep 2
done

echo "Service states:"
systemctl is-active llama-qwen38-27b || true
systemctl is-active llama-qwen3-8b-dpo || true
echo "Models:"
cat /tmp/qwen3_dpo_models.json 2>/dev/null || echo "(endpoint not ready yet)"
