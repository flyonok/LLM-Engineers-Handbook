# P0–P2 Local Inference Cutover Implementation Plan

> **For agentic workers:** Execute task-by-task. Steps use checkbox syntax.

**Goal:** Cut :8001 to the trained Qwen3-8B-DPO GGUF, wire RAG/ml_service to the local OpenAI-compatible backend, record the 8B/full-train decision, and push the local commits.

**Architecture:** Keep SageMaker inference as a fallback. When `USE_LOCAL_LLM_INFERENCE=true` (default for this fork), FastAPI RAG uses an OpenAI-compatible client against `OPENAI_BASE_URL` and returns the same `[{generated_text}]` shape expected by `InferenceExecutor`.

**Tech Stack:** systemd + llama.cpp, FastAPI, openai/langchain-openai, pydantic-settings

## Global Constraints

- Do not break the SageMaker path when local inference is disabled.
- Default local model alias: `qwen3-8b-dpo-q4km`
- Port remains `8001` for production cutover.
- Stay on Qwen3-8B for now; do not start a new full/32B training job in this change set.

---

### Task 1: P0 service cutover

- [ ] Stop `llama-qwen38-27b`, start `llama-qwen3-8b-dpo`
- [ ] Verify `curl http://127.0.0.1:8001/v1/models` returns `qwen3-8b-dpo-q4km`

### Task 2: P1 local inference client

- [ ] Add settings flags for local LLM inference
- [ ] Implement `LLMInferenceOpenAIEndpoint`
- [ ] Route `inference_pipeline_api` + RAG ChatOpenAI helpers through local base_url
- [ ] Update `.env.example`

### Task 3: P1 re-eval + P2 docs/push

- [ ] Re-run `scripts/evaluate_local_endpoint.py`
- [ ] Record P2 decision in `doc/memo-local-training-plan.md`
- [ ] Push existing local commits (exclude dirty ArticleDocument.json unless intentional)
