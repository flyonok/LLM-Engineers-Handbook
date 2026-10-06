# 本地化训练管道跑通（dummy SFT）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: 使用 superpowers:executing-plans 逐任务执行。步骤用 checkbox（`- [ ]`）跟踪。

**Goal:** 在 RTX 3090 上跑通本地 QLoRA SFT 管道（dummy 模式），验证全链路通畅，OOM/磁盘不足时退 Qwen3-8B。

**Architecture:** 代码改造已落地（`run_local.py`/`finetune.py`/`requirements.txt`/`scripts/local_dummy_sft.sh`）。本计划只做环境搭建 + 验证 + 跑通，不改代码。

**Tech Stack:** Python 3.11 venv、torch、unsloth、transformers>=4.51、trl、HF_ENDPOINT=hf-mirror.com、systemd 服务 `llama-qwen38-27b`。

## Global Constraints

- 基座模型：先 `Qwen/Qwen3-32B`，OOM/磁盘不足退 `Qwen/Qwen3-8B`
- 交付范围：dummy SFT（400 样本 1 epoch），不跑全量
- 服务处理：用 `scripts/local_dummy_sft.sh` 一键脚本自动停/启服务
- 训练 venv：独立 `.venv-train`，不装进主项目
- 国内镜像：`HF_ENDPOINT=https://hf-mirror.com`
- 成功标志：`data/training_output/sft_merged/model.safetensors` 存在 + 推理服务恢复

---

### Task 1: 停推理服务 + 创建训练 venv + 安装依赖

**Files:**
- 无代码改动
- 创建：`.venv-train/`（venv）

**Interfaces:**
- Consumes: `llm_engineering/model/finetuning/requirements.txt`
- Produces: 可用的 `.venv-train` 训练环境（含 unsloth/torch/transformers/trl）

- [ ] **Step 1: 停推理服务释放 GPU**

```bash
sudo systemctl stop llama-qwen38-27b
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv
```
Expected: 显存占用降到接近 0 MiB（推理服务释放 20GB）

- [ ] **Step 2: 创建训练 venv**

```bash
cd /home/ubuntu/amplifier/git/LLM-Engineers-Handbook
python3 -m venv .venv-train
source .venv-train/bin/activate
python --version
```
Expected: Python 3.11.x

- [ ] **Step 3: 安装训练依赖（国内镜像）**

```bash
export HF_ENDPOINT=https://hf-mirror.com
pip install -r llm_engineering/model/finetuning/requirements.txt
```
Expected: 安装 torch / transformers>=4.51 / unsloth>=2025.3.18 / trl>=0.15 等，无版本冲突
- 若 torch 下载慢：`pip install torch --index-url https://download.pytorch.org/whl/cu121`

- [ ] **Step 4: 验证关键包可导入**

```bash
python -c "import unsloth, transformers, trl, torch; print('unsloth', unsloth.__version__); print('transformers', transformers.__version__); print('torch', torch.__version__)"
```
Expected: 三行版本号打印，无 ImportError

- [ ] **Step 5: 提交进度（venv 不入库，仅记录）**

```bash
git status
```
Expected: 无新增代码改动（venv 应被 gitignore 或忽略）
- 若 `.venv-train` 未被 gitignore，检查 `.gitignore` 是否含 `.venv*`

---

### Task 2: 验证 Qwen3-32B 可 4bit 加载

**Files:**
- 无代码改动（临时验证脚本）

**Interfaces:**
- Consumes: Task 1 的 `.venv-train` 环境
- Produces: 确认 32B 4bit 可加载（或确认需退 8B）

- [ ] **Step 1: 写临时验证脚本**

创建 `/tmp/verify_qwen3_32b.py`：

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    "Qwen/Qwen3-32B",
    max_seq_length=1024,
    load_in_4bit=True,
)
print("OK", type(model).__name__)
```

- [ ] **Step 2: 运行验证**

```bash
source .venv-train/bin/activate
export HF_ENDPOINT=https://hf-mirror.com
python /tmp/verify_qwen3_32b.py
```
Expected: 打印 `OK PeftModelForCausalLM`（或类似），无 OOM
- 首次运行会从 hf-mirror 下载 Qwen3-32B 权重（约 60GB+），耗时较长，耐心等待
- 若 OOM 或加载失败 → 记录"32B 不可用"，Task 4 改用 8B

---

### Task 3: 跑 dummy SFT（32B 起步）

**Files:**
- 无代码改动
- 调用：`scripts/local_dummy_sft.sh`

**Interfaces:**
- Consumes: Task 1 venv、Task 2 验证结论、`data/artifacts/instruct_datasets.json`
- Produces: `data/training_output/sft_merged/` 合并模型

- [ ] **Step 1: 前置检查磁盘**

```bash
df -BG --output=avail . | tail -1
```
Expected: 剩余 >= 20GB（dummy SFT 产物小）

- [ ] **Step 2: 跑一键脚本（32B）**

```bash
cd /home/ubuntu/amplifier/git/LLM-Engineers-Handbook
./scripts/local_dummy_sft.sh
```
Expected:
- 停服务 → 建/复用 venv → 跑 dummy SFT（400 样本 1 epoch）→ 重启服务
- 训练日志显示 `Loaded dataset with 400 samples`
- 末尾打印 `SFT finished. Merged model: data/training_output/sft_merged/`
- 若 OOM（CUDA out of memory）→ 记录，转 Task 3.5 退 8B

- [ ] **Step 3: 验证产物**

```bash
ls -lh data/training_output/sft_merged/
```
Expected: 存在 `model.safetensors`（及 `config.json`/`tokenizer*`）

- [ ] **Step 4: 验证推理服务恢复**

```bash
systemctl is-active llama-qwen38-27b
curl -s http://127.0.0.1:8001/v1/models | head -c 200
```
Expected: `active` + 返回模型列表 JSON

---

### Task 3.5: 退 Qwen3-8B 重跑（仅当 Task 3 OOM/磁盘不足）

**Files:**
- 无代码改动

**Interfaces:**
- Consumes: Task 1 venv、Task 3 的 OOM 记录
- Produces: `data/training_output/sft_merged/`（8B 合并模型）

- [ ] **Step 1: 清理 32B 失败产物（若有）**

```bash
rm -rf data/training_output/output_sft data/training_output/sft_merged
```

- [ ] **Step 2: 用 8B 重跑**

```bash
BASE_MODEL=Qwen/Qwen3-8B ./scripts/local_dummy_sft.sh
```
Expected: 8B dummy SFT 成功，产物在 `data/training_output/sft_merged/`

- [ ] **Step 3: 验证产物 + 服务恢复（同 Task 3 Step 3-4）**

```bash
ls -lh data/training_output/sft_merged/
systemctl is-active llama-qwen38-27b
```

---

### Task 4: 记录结果 + 提交 git

**Files:**
- Modify: `docs/superpowers/specs/2026-10-05-local-training-dummy-sft-design.md`（补实际结果）
- 提交：所有已落地代码改动 + 文档

**Interfaces:**
- Consumes: Task 3/3.5 的产物
- Produces: 已提交的本地化训练改动

- [ ] **Step 1: 记录实际结果到 spec 文档**

在 spec 文档末尾追加"## 7. 实际执行结果"小节，记录：
- 实际使用的基座模型（32B 或 8B）
- 训练耗时
- 显存峰值（`nvidia-smi` 训练时截图或数值）
- 产物路径与大小

- [ ] **Step 2: 提交 git**

```bash
git add .env.example configs/training.yaml \
  llm_engineering/application/dataset/generation.py \
  llm_engineering/model/evaluation/evaluate.py \
  llm_engineering/model/evaluation/run_local.py \
  llm_engineering/model/finetuning/finetune.py \
  llm_engineering/model/finetuning/requirements.txt \
  llm_engineering/model/finetuning/run_local.py \
  llm_engineering/model/finetuning/sagemaker.py \
  llm_engineering/settings.py \
  pipelines/training.py pyproject.toml steps/training/train.py \
  scripts/local_dummy_sft.sh doc/ docs/
git status
git commit -m "feat: 本地化训练管道（Qwen3 QLoRA）+ dummy SFT 验证跑通

- run_local.py 本地训练入口替代 SageMaker
- finetune.py 去 SM_* 环境变量、本地数据集加载、4bit、基座参数化
- requirements.txt 升级到 Qwen3 支持版本
- generation.py tiktoken 改 cl100k_base、本地端点
- scripts/local_dummy_sft.sh 一键脚本
- 文档：local-training-workflow.md / memo-local-training-plan.md"
```
Expected: commit 成功，无遗漏文件

- [ ] **Step 3: 确认 GPU 已归还推理服务**

```bash
nvidia-smi --query-gpu=memory.used --format=csv
systemctl is-active llama-qwen38-27b
```
Expected: 显存被推理服务占用（约 20GB），服务 active
