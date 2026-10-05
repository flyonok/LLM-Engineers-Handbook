# Local QLoRA Training on RTX 3090 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the LLM Engineer's Handbook training pipeline running end-to-end on the local RTX 3090 using `Qwen/Qwen3-32B` (with `Qwen/Qwen3-8B` fallback), producing a merged 16-bit model that can be converted to GGUF and deployed back to the existing llama.cpp inference service.

**Architecture:** Replace the SageMaker estimator path with `llm_engineering/model/finetuning/run_local.py`, which calls the QLoRA `finetune()` function and merges the LoRA adapter to a 16-bit checkpoint on disk. The pipeline is driven either by `scripts/local_dummy_sft.sh` (stop service → train → restart service) or by `poetry poe run-training-pipeline` when `configs/training.yaml` has `local: true`. All training happens in an isolated venv (`.venv-train`) because the main Poetry environment pins older torch/transformers versions that do not support Qwen3.

**Tech Stack:** Python 3.11, unsloth, transformers>=4.51, trl>=0.15, torch>=2.4, bitsandbytes, RTX 3090 (sm_86, bf16), llama.cpp (existing systemd service `llama-qwen38-27b`).

## Global Constraints

- Base model: `Qwen/Qwen3-32B` (official); fallback to `Qwen/Qwen3-8B` if 32B OOM.
- Max seq length: `1024` for 32B; can be increased for 8B.
- Batch size: `1`; gradient accumulation `16`.
- Optimizer: `adamw_8bit`.
- Training venv: isolated at `.venv-train`, installed from `llm_engineering/model/finetuning/requirements.txt`.
- HF mirror: `export HF_ENDPOINT=https://hf-mirror.com` before any model/data download.
- Inference service must be stopped before training to free 24 GB VRAM.
- Output root: `data/training_output/`.
- Merged model format: `merged_16bit` (produced by unsloth `save_pretrained_merged`).

---

## Current State

Most code changes are already in place:

- `llm_engineering/model/finetuning/run_local.py` — new local entry point.
- `llm_engineering/model/finetuning/finetune.py` — supports local JSON datasets, 4-bit loading, Qwen3 chatml template.
- `llm_engineering/model/finetuning/requirements.txt` — upgraded for Qwen3.
- `llm_engineering/settings.py` — local training env vars added.
- `.env.example` — local endpoint and training defaults added.
- `configs/training.yaml` — `local: true`, base model, seq length, local dataset paths.
- `scripts/local_dummy_sft.sh` — one-button dummy SFT (stop service → train → restart).
- `steps/training/train.py` and `pipelines/training.py` — local/SageMaker switch.
- `llm_engineering/application/dataset/generation.py` — uses `OPENAI_BASE_URL` and `cl100k_base` tokenizer.
- `llm_engineering/model/evaluation/run_local.py` — wrapper for local endpoint evaluation.

Remaining work is primarily **execution and verification**: set up the isolated venv, prove the model loads, run dummy SFT, optionally run full SFT → DPO → GGUF → deploy, and verify evaluation works against the local endpoint.

---

### Task 1: Pre-flight Checks

**Files:**
- Read: `doc/local-training-workflow.md`
- Read: `doc/memo-local-training-plan.md`
- Modify: `.env` (copy from `.env.example` if missing)

**Interfaces:**
- Consumes: existing systemd service `llama-qwen38-27b`, NVIDIA driver, Python 3.11.
- Produces: confirmed GPU state, free disk space, populated `.env`.

- [ ] **Step 1: Verify GPU and disk**

```bash
nvidia-smi
df -h .
```

Expected: RTX 3090 visible with ~24 GB VRAM; at least 120 GB free disk for full 32B workflow (20 GB+ for dummy SFT).

- [ ] **Step 2: Confirm inference service status**

```bash
sudo systemctl status llama-qwen38-27b
# If active, stop it before training:
sudo systemctl stop llama-qwen38-27b
nvidia-smi
```

Expected: service stopped, VRAM usage near zero before training.

- [ ] **Step 3: Ensure `.env` exists**

```bash
cp -n .env.example .env
```

Expected: `.env` exists with `LOCAL_TRAINING=true`, `OPENAI_BASE_URL=http://127.0.0.1:8001/v1`, `TRAINING_BASE_MODEL=Qwen/Qwen3-32B`.

- [ ] **Step 4: Commit baseline state**

```bash
git add docs/superpowers/plans/2026-10-05-local-training-rtx3090.md
git commit -m "docs: add local RTX 3090 training implementation plan"
```

---

### Task 2: Create Isolated Training venv

**Files:**
- Modify: `.venv-train/` (created)
- Read: `llm_engineering/model/finetuning/requirements.txt`

**Interfaces:**
- Consumes: Python 3.11, CUDA-capable driver.
- Produces: `.venv-train` with unsloth + Qwen3-compatible transformers.

- [ ] **Step 1: Create and activate venv**

```bash
python3 -m venv .venv-train
source .venv-train/bin/activate
```

Expected: prompt shows `(.venv-train)`.

- [ ] **Step 2: Install training dependencies**

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -U pip
pip install -r llm_engineering/model/finetuning/requirements.txt
```

Expected: no errors; `pip show unsloth transformers trl` shows versions >= `2025.3.18`, `4.51.0`, `0.15.0`.

- [ ] **Step 3: Verify imports**

```bash
python - <<'PY'
import torch
import unsloth
import transformers
import trl
print(torch.__version__, transformers.__version__, trl.__version__)
print("CUDA available:", torch.cuda.is_available())
PY
```

Expected: CUDA available; versions match requirements.

- [ ] **Step 4: Commit nothing (venv is gitignored)**

No commit needed. Ensure `.venv-train` is not staged.

---

### Task 3: Verify Qwen3-32B Loads in 4-bit

**Files:**
- Modify: temporary verification script (optional)

**Interfaces:**
- Consumes: installed venv, HF mirror, internet/HF cache.
- Produces: confirmed 4-bit model load without OOM.

- [ ] **Step 1: Run load test**

```bash
source .venv-train/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
python - <<'PY'
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-32B",
    max_seq_length=1024,
    load_in_4bit=True,
)
print("OK", type(model).__name__)
print("VRAM used (MB):", torch.cuda.memory_allocated() // 1024 // 1024)
PY
```

Expected: prints `OK` and VRAM under ~18 GB; no OOM.

- [ ] **Step 2: If OOM, fall back to 8B and document**

```bash
python - <<'PY'
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-8B",
    max_seq_length=1024,
    load_in_4bit=True,
)
print("OK 8B", type(model).__name__)
PY
```

Expected: 8B loads with ~6 GB VRAM. Note in execution log if fallback is used.

---

### Task 4: Fix Known Code Issues Before First Training Run

**Files:**
- Modify: `llm_engineering/model/finetuning/finetune.py`
- Modify: `llm_engineering/model/finetuning/run_local.py` (optional learning_rate for DPO)

**Interfaces:**
- Consumes: current `finetune.py` DPO signature.
- Produces: DPO trainer that instantiates with new `trl` API.

- [ ] **Step 1: Inspect DPOTrainer construction**

Read lines 253–285 of `llm_engineering/model/finetuning/finetune.py`.

- [ ] **Step 2: Move `max_length` and `max_prompt_length` into `DPOConfig` if current `trl` rejects them as `DPOTrainer` kwargs**

If running dummy SFT first, this issue may not surface until DPO. When it does, change:

```python
# Before (current)
trainer = DPOTrainer(
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    beta=beta,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    max_length=max_seq_length // 2,
    max_prompt_length=max_seq_length // 2,
    args=DPOConfig(...),
)
```

```python
# After
args = DPOConfig(
    learning_rate=learning_rate,
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="linear",
    per_device_eval_batch_size=per_device_train_batch_size,
    warmup_steps=10,
    output_dir=output_dir,
    eval_steps=0.2,
    logging_steps=1,
    report_to=report_to if report_to != "none" else "none",
    seed=0,
    max_length=max_seq_length // 2,
    max_prompt_length=max_seq_length // 2,
)

trainer = DPOTrainer(
    model=model,
    ref_model=None,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)
```

Expected: `DPOTrainer` instantiates without `TypeError` about unexpected kwargs.

- [ ] **Step 3: Commit any fix**

```bash
git add llm_engineering/model/finetuning/finetune.py
git commit -m "fix: align DPOTrainer call with trl>=0.15 API"
```

---

### Task 5: Run Dummy SFT (Pipeline Validation)

**Files:**
- Read: `scripts/local_dummy_sft.sh`
- Output: `data/training_output/sft_merged/model.safetensors`

**Interfaces:**
- Consumes: venv, local dataset `data/artifacts/instruct_datasets.json`, free GPU.
- Produces: merged 16-bit SFT checkpoint.

- [ ] **Step 1: Run one-button dummy SFT**

```bash
export HF_ENDPOINT=https://hf-mirror.com
./scripts/local_dummy_sft.sh
```

Expected: script stops service, trains ~400 samples for 1 epoch, saves merged model, restarts service. Training loss should decrease.

- [ ] **Step 2: Verify merged checkpoint exists**

```bash
ls -lh data/training_output/sft_merged/model.safetensors
```

Expected: file exists; size ~64 GB for 32B or ~16 GB for 8B.

- [ ] **Step 3: Inspect logs for errors**

```bash
tail -n 50 data/training_output/output_sft/*.log 2>/dev/null || echo "No separate log file; logs printed to stdout"
```

Expected: no `KeyError`, no `OOM`, no `TypeError` from trainer construction.

- [ ] **Step 4: Commit no code change if run succeeded**

If the script succeeded without edits, no commit. If a fix was needed, commit it.

---

### Task 6: Run Full SFT (Optional — Long Running)

**Files:**
- Output: `data/training_output/sft_merged/model.safetensors` (overwritten)

**Interfaces:**
- Consumes: merged checkpoint from dummy run (will be overwritten).
- Produces: fully trained SFT merged checkpoint.

- [ ] **Step 1: Run full SFT**

```bash
source .venv-train/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
sudo systemctl stop llama-qwen38-27b

python llm_engineering/model/finetuning/run_local.py \
  --finetuning_type sft \
  --base_model Qwen/Qwen3-32B \
  --max_seq_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 3 \
  --use_local_datasets \
  --local_sft_dataset_path data/artifacts/instruct_datasets.json \
  --output_dir data/training_output \
  --report_to none

sudo systemctl start llama-qwen38-27b
```

Expected: 3 epochs complete; merged checkpoint saved. Time estimate: 9–16 hours for 32B on ~8M tokens.

- [ ] **Step 2: Verify checkpoint**

```bash
ls -lh data/training_output/sft_merged/model.safetensors
```

Expected: merged checkpoint exists.

---

### Task 7: Run DPO

**Files:**
- Input: `data/training_output/sft_merged`
- Output: `data/training_output/dpo_merged/model.safetensors`

**Interfaces:**
- Consumes: SFT merged checkpoint.
- Produces: DPO merged checkpoint.

- [ ] **Step 1: Run DPO (dummy or full)**

Dummy:

```bash
source .venv-train/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
sudo systemctl stop llama-qwen38-27b

python llm_engineering/model/finetuning/run_local.py \
  --finetuning_type dpo \
  --base_model Qwen/Qwen3-32B \
  --max_seq_length 1024 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 16 \
  --use_local_datasets \
  --local_dpo_dataset_path data/artifacts/preference_datasets.json \
  --sft_merged_model_dir data/training_output/sft_merged \
  --output_dir data/training_output \
  --report_to none \
  --is_dummy

sudo systemctl start llama-qwen38-27b
```

Full: remove `--is_dummy`.

Expected: DPO trains; `data/training_output/dpo_merged/` created.

- [ ] **Step 2: Verify DPO merged checkpoint**

```bash
ls -lh data/training_output/dpo_merged/model.safetensors
```

Expected: file exists.

---

### Task 8: Convert Merged Model to GGUF and Quantize

**Files:**
- Input: `data/training_output/dpo_merged` (or `sft_merged`)
- Output: `data/training_output/dpo_merged-Q4_K_M.gguf`

**Interfaces:**
- Consumes: llama.cpp installation with `convert_hf_to_gguf.py` and `llama-quantize`.
- Produces: Q4_K_M GGUF ready for the inference service.

- [ ] **Step 1: Set llama.cpp path**

```bash
# Adjust to actual installation path
export LLAMA_CPP=/path/to/llama.cpp
```

- [ ] **Step 2: Convert to F16 GGUF**

```bash
python "$LLAMA_CPP/convert_hf_to_gguf.py" \
  data/training_output/dpo_merged \
  --outfile data/training_output/dpo_merged-F16.gguf \
  --outtype f16
```

Expected: conversion completes; F16 GGUF created (~64 GB for 32B).

- [ ] **Step 3: Quantize to Q4_K_M**

```bash
"$LLAMA_CPP/build/bin/llama-quantize" \
  data/training_output/dpo_merged-F16.gguf \
  data/training_output/dpo_merged-Q4_K_M.gguf \
  Q4_K_M
```

Expected: Q4_K_M GGUF created (~18–20 GB for 32B).

- [ ] **Step 4: Clean up F16 intermediate (if disk tight)**

```bash
rm data/training_output/dpo_merged-F16.gguf
```

---

### Task 9: Deploy GGUF to Inference Service

**Files:**
- System: systemd unit `llama-qwen38-27b.service`
- Output: running llama.cpp service serving the fine-tuned GGUF

**Interfaces:**
- Consumes: Q4_K_M GGUF.
- Produces: OpenAI-compatible endpoint at `http://127.0.0.1:8001/v1`.

- [ ] **Step 1: Inspect service unit for model path**

```bash
sudo systemctl cat llama-qwen38-27b | grep -E 'ExecStart|WorkingDirectory|Model|model'
```

Expected: discover the directory and filename the service loads.

- [ ] **Step 2: Backup old model and replace**

```bash
sudo systemctl stop llama-qwen38-27b
sudo cp /path/to/old/Qwen3.8-27B-Q4_K_M.gguf /path/to/old/Qwen3.8-27B-Q4_K_M.gguf.bak
sudo cp data/training_output/dpo_merged-Q4_K_M.gguf /path/to/models/
# Optional: rename service or update ExecStart to point to new file
sudo systemctl start llama-qwen38-27b
```

Expected: service starts successfully.

- [ ] **Step 3: Verify endpoint**

```bash
curl -s http://127.0.0.1:8001/v1/models
```

Expected: JSON response listing the served model.

---

### Task 10: Evaluate via Local Endpoint

**Files:**
- Modify: `llm_engineering/model/evaluation/evaluate.py` if local dataset loading needs fixing
- Read: `llm_engineering/model/evaluation/run_local.py`

**Interfaces:**
- Consumes: running local endpoint, `data/artifacts/instruct_datasets.json`.
- Produces: accuracy/style scores.

- [ ] **Step 1: Run local evaluation in dummy mode**

```bash
python llm_engineering/model/evaluation/run_local.py --dummy
```

Expected: script calls `evaluate.py` with `OPENAI_BASE_URL=http://127.0.0.1:8001/v1` and `LOCAL_MODEL_NAME=local-model`.

- [ ] **Step 2: Fix dataset loading if it fails**

`evaluate.py` currently calls:

```python
dataset = load_dataset(dataset_name, split="test")
```

where `dataset_name = f"{DATASET_HUGGINGFACE_WORKSPACE}/llmtwin"`. For local evaluation, add a helper in `evaluate.py` (or a pre-step in `run_local.py`) that loads `data/artifacts/instruct_datasets.json` and creates a Hugging Face `Dataset` object with a `test` split, then set the dataset name to that in-memory/arrow path. The minimal fix is to add in `evaluate.py`:

```python
def _load_local_dataset(path: str):
    import json
    from datasets import Dataset, DatasetDict
    with open(path) as f:
        data = json.load(f)
    records = {"instruction": [], "answer": []}
    for split in ("train", "test"):
        for category in data.get(split, {}).values():
            for sample in category.get("samples", []):
                records["instruction"].append(sample["instruction"])
                records["answer"].append(sample["answer"])
    return DatasetDict({"test": Dataset.from_dict(records)})
```

and branch in `generate_answers` when `DATASET_HUGGINGFACE_WORKSPACE == "local"`.

Expected: evaluation produces accuracy/style scores without HF dataset lookup errors.

- [ ] **Step 3: Commit evaluation fix**

```bash
git add llm_engineering/model/evaluation/evaluate.py llm_engineering/model/evaluation/run_local.py
git commit -m "fix: support local dataset in evaluation via llama.cpp endpoint"
```

---

### Task 11: Final Verification and Handoff

**Files:**
- Read: `data/training_output/sft_merged/model.safetensors`
- Read: `data/training_output/dpo_merged/model.safetensors` (if DPO ran)
- Read: systemd service status

**Interfaces:**
- Consumes: all previous deliverables.
- Produces: final status report.

- [ ] **Step 1: Run verification checklist**

```bash
source .venv-train/bin/activate
python -c "import unsloth, transformers, torch; print(transformers.__version__, torch.cuda.is_available())"
ls -lh data/training_output/sft_merged/model.safetensors
ls -lh data/training_output/dpo_merged/model.safetensors 2>/dev/null || echo "DPO not run"
curl -s http://127.0.0.1:8001/v1/models
sudo systemctl is-active llama-qwen38-27b
```

Expected: all checks pass.

- [ ] **Step 2: Document run results**

Append a short summary to `docs/superpowers/plans/2026-10-05-local-training-rtx3090.md` under a new `## Execution Log` section: what ran, what OOM'd, what model was used, final checkpoint paths, endpoint status.

- [ ] **Step 3: Commit final state**

```bash
git add docs/superpowers/plans/2026-10-05-local-training-rtx3090.md
git commit -m "docs: record local training execution results"
```

---

## Risk Register

| Risk | Likely Symptom | Mitigation |
|------|---------------|------------|
| 32B QLoRA OOM | `CUDA out of memory` during `FastLanguageModel.from_pretrained` or first training step | Use `Qwen/Qwen3-8B`; keep `max_seq_length=1024`, batch 1, grad accum 16 |
| Old dependency incompatible with Qwen3 | `KeyError` in model config / tokenizer | Verify venv uses `requirements.txt`, not main Poetry env |
| `trl` DPO API changed | `TypeError: DPOTrainer.__init__() got an unexpected keyword argument 'max_length'` | Move lengths into `DPOConfig`, use `processing_class=tokenizer` |
| Disk full during merge/GGUF | `No space left on device` | Reserve 120 GB; delete F16 intermediate after quantization |
| Local endpoint 404 | `curl :8001/v1/models` fails | Check systemd unit; ensure service started after model swap |
| Evaluation loads HF dataset that does not exist | `RepositoryNotFoundError` for `local/llmtwin` | Add local JSON dataset branch in `evaluate.py` |

---

## Spec Coverage Check

- §3.1–3.2 (target environment and model) → Tasks 1–3.
- §3.3–3.4 (SageMaker → local path) → Tasks 5–7; `run_local.py` already implemented.
- §3.5 (dependency upgrade) → Task 2.
- §3.6 (file changes) → already in code; Task 4 covers any runtime API mismatch.
- §3.7 (VRAM constraints) → Tasks 3, 5, fallback handled in Task 3.
- §3.8 (training time) → Task 6 marked optional/long-running.
- §3.9 (GGUF deployment) → Tasks 8–9.
- §3.10 #5 (evaluation) → Task 10.
- `doc/local-training-workflow.md` §0–3 → Tasks 1–5; §4 → Task 6; §5 → Task 7; §6 → Task 8; §7 → Task 9; §8 → Task 10.
