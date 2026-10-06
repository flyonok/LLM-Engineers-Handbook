# 本地训练闭环补全设计

> **目标：** 在 RTX 3090 上完成 SFT 之后的剩余步骤：DPO 训练 → GGUF 转换/量化 → 临时启动验证服务 → 端点评估 → 文档更新，最终把可复现的 GGUF 交接给用户部署进 systemd 推理服务。

## 背景

- SFT dummy 训练已跑通，合并模型保存在 `data/training_output/sft_merged/`。
- 推理服务 `llama-qwen38-27b.service` 当前处于停止状态。
- 该 systemd unit 对 GGUF 文件做了 SHA-256 + 大小强校验，不能直接替换文件；需要用户用 sudo 更新启动脚本或新建 service。
- 当前训练基座为本地 `Qwen/Qwen3-8B` 权重（非 Instruct 版），但训练管道本身不依赖基座是否 Instruct。

## 设计原则

1. **先验证端到端，再做生产部署**：DPO/GGUF 完成后，用独立进程临时启动 `llama-server` 验证 `:8001` 推理可用，避免直接改生产服务。
2. **不依赖 sudo**：除最后一步明确交给用户外，其余步骤均在普通用户权限下完成。
3. **可复现**：所有命令写入脚本，输出路径与 workflow 文档保持一致。

## 总体架构

```
sft_merged (16bit HF 格式)
   │
   ▼
DPO 训练 ──► dpo_merged (16bit HF 格式)
   │
   ▼
convert_hf_to_gguf.py ──► dpo_merged-F16.gguf
   │
   ▼
llama-quantize ──► dpo_merged-Q4_K_M.gguf
   │
   ▼
手动 llama-server (:8001) ──► evaluation/run_local.py --dummy
   │
   ▼
文档更新 + git commit
   │
   ▼
交接给用户：sudo 更新 llama-qwen38-27b.service 指向新 GGUF
```

## 关键约束

- **显存**：RTX 3090 24GB；DPO 比 SFT 更吃显存，必须 seq_len=1024 / batch=1。
- **磁盘**：合并模型约 16GB，GGUF F16 约 16GB，Q4_K_M 约 4–5GB，预留 40GB+。
- **无 sudo**：不能修改 `/usr/local/lib/voxsign/run_llama_qwen38_27b.sh` 或 systemctl 配置。
- **依赖**：全部使用已有的 `.venv-train`；llama.cpp 使用 `/mnt/aitools/src/llama.cpp-ddd4ec1428a6201e18975ea52b07c71e0f9aef26/`。

## 组件设计

### 1. DPO 训练

复用 `llm_engineering/model/finetuning/run_local.py`，参数与 SFT 类似，但：
- `--finetuning_type dpo`
- `--sft_merged_model_dir data/training_output/sft_merged`
- `--local_dpo_dataset_path data/artifacts/preference_datasets.json`

输出：`data/training_output/dpo_merged/`

验证：
- 目录存在且包含 safetensors 分片
- 训练 loss 收敛
- 用 `scripts/verify_sft_model.py` 改测 DPO 输出（或直接加载 dpo_merged 生成回复）

### 2. GGUF 转换与量化

复用现有 llama.cpp 工具：
- `convert_hf_to_gguf.py` → F16
- `llama-quantize` → Q4_K_M

验证：
- 文件大小合理（F16 ~16GB，Q4_K_M ~4–5GB）
- `llama-server` 能加载且不报错

### 3. 临时推理服务

写一个独立脚本 `scripts/run_llama_server_manual.sh`，使用与 systemd service 相同的 llama-server 二进制，但：
- 指向新生成的 `dpo_merged-Q4_K_M.gguf`
- 监听 `:8001`
- alias 为 `qwen3-8b-dpo-q4km`
- 简化参数（去掉 draft MTP，保留 flash-attn、jinja、ctx-size=131072 等核心参数）

验证：
- `curl http://127.0.0.1:8001/v1/models` 返回模型名
- 生成一条测试回复

### 4. 评估

运行 `llm_engineering/model/evaluation/run_local.py --dummy`，环境变量指向 `:8001`。

验证：
- 脚本退出码 0
- 输出显示生成与评判结果

### 5. 文档与提交

更新：
- `doc/memo-local-training-plan.md`：记录 DPO/GGUF/评估实测结果
- `doc/local-training-workflow.md`：修正部署说明，补充手动启动脚本使用方式

提交所有新增/修改文件。

### 6. 生产部署交接

向用户提供：
- 新 GGUF 的绝对路径
- 建议修改的 systemd service 配置片段
- `sudo systemctl daemon-reload && sudo systemctl restart llama-qwen38-27b` 命令

## 风险与应对

| 风险 | 应对 |
|------|------|
| DPO OOM | seq_len=1024，batch=1；仍 OOM 则跳过 DPO，只部署 SFT 模型 |
| GGUF 转换失败 | 检查 llama.cpp 版本是否支持 Qwen3；必要时换更新版本 |
| 临时服务端口被占 | 服务已停止，若冲突则改用 :8002 并覆盖 eval 的 `OPENAI_BASE_URL` |
| eval 脚本对本地端点不兼容 | 根据报错微调 env 变量或 `LOCAL_MODEL_NAME` |

## 范围外（本次不做）

- 爬取/生成新的自定义训练数据
- 切换到 `Qwen3-8B-Instruct` 基础版
- 修改现有 systemd service 文件
- 全量 SFT（dummy 已验证管道）
