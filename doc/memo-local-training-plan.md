# 项目改造备忘录：本地化训练 + 国内环境适配 + 公开数据策略

> 创建日期：2026-10-03
> 状态：**代码改造已落地**（本地训练路径 + 国内环境适配配置；评估脚本与 GGUF 部署步骤仍按备忘录暂缓）
> 用途：记录本次讨论的全部结论与改造方案，作为后续改进项目的依据

---

## 第一部分：背景与目标

### 1.1 项目本质

本仓库是《LLM Engineer's Handbook》官方代码，目标是构建端到端 "LLM Twin" 系统：

- **数据链路**：爬虫采集 → MongoDB（原始层）→ 特征工程（清洗/分块/嵌入）→ Qdrant（向量库）→ GPT 生成 instruct/preference 数据集 → 微调 → RAG 推理服务
- **架构分工**：微调负责"写作风格与表达方式"，知识检索靠 RAG（推理时从向量库查内容再生成）
- 编排层：ZenML 管道（`pipelines/` 定义管道，`steps/` 是可复用步骤），核心包 `llm_engineering/` 按 DDD 分层

### 1.2 使用者背景与目标

- 用户：软件开发工程师，目前从事 **AI 应用开发** 方向
- 暂时没有个人博客，计划 **利用网络公开内容** 进行训练
- 概念转变：从"模仿自己的数字分身"变为"AI 应用开发领域的风格模型"——模型学到的是所选内容作者的风格 + 领域知识
- **核心目标：把训练环节从 AWS SageMaker 改为本地 RTX 3090 完成**，基座模型改用官方 **Qwen3-32B**（与推理服务保持一致；原计划的 "Qwen3.8-27B" 为社区非官方命名，已替换为最接近的官方型号）

---

## 第二部分：数据采集机制分析结论（已确认）

### 2.1 采集架构

```
configs/digital_data_etl_*.yaml（链接清单 + user_full_name）
  → ZenML 管道 digital_data_etl（pipelines/digital_data_etl.py）
    → step1: get_or_create_user（MongoDB 建作者）
    → step2: crawl_links（steps/etl/crawl_links.py）
      → CrawlerDispatcher 按域名正则路由（llm_engineering/application/crawlers/dispatcher.py）
        ├─ medium.com   → MediumCrawler（Selenium + BeautifulSoup）
        ├─ linkedin.com → LinkedInCrawler（已废弃，extract 直接抛 DeprecationWarning）
        ├─ github.com   → GithubCrawler（git clone 整仓源码，无需浏览器/账号）
        └─ 其他域名     → CustomArticleCrawler（兜底：langchain AsyncHtmlLoader + Html2Text）
  → 存 MongoDB：posts / articles / repositories / users 四个集合，按 link 去重，可重复跑
```

### 2.2 账号需求结论

- **采集环节基本不需要账号**：Medium 公开文章、GitHub 公开仓库、普通网页都是匿名抓取
- 唯一需要账号的 LinkedIn 爬虫已废弃（LinkedIn 升级反爬）
- 真正需要 API Key 的是**下游**：`OPENAI_API_KEY`（生成训练数据集）、`HUGGINGFACE_ACCESS_TOKEN`（推数据集/模型）、`COMET_API_KEY`（实验追踪）

### 2.3 国内环境适配结论

- 代码库**没有任何代理支持**（已全库搜索确认）
- 被墙/受限：Medium、Substack、LinkedIn、OpenAI API、HuggingFace、Docker Hub（不稳）、Chrome 下载
- 可达：GitHub（慢）、AWS 国际区、MongoDB Atlas、Qdrant Cloud、Comet
- **CustomArticleCrawler 兜底能力**：任何"服务端渲染 + 无登录墙"的公开文章页都能抓，无需写新代码。适用：掘金、博客园、CSDN（噪音多）、少数派、InfoQ、个人博客（Hexo/Hugo/WordPress 效果最佳）、开源项目文档站。不适用：知乎、公众号、头条/百家号（登录墙/JS 渲染/反爬）
- **项目自带作者数据可完全跳过采集**：`data/data_warehouse_raw_data/`（原始文章 1.3MB，内容本身就是 LLM 工程/AI 应用方向）+ `data/artifacts/`（清洗后文档、instruct/preference 数据集 JSON），用 `poetry poe run-import-data-warehouse-from-json` 直接导入

### 2.4 数据策略（无博客、用公开内容）

优先级排序：

1. **项目自带作者数据**（今天就能跑通流程，零代理零采集）
2. **GitHub 仓库**：优质 AI 应用项目（RAG/Agent 类），挑 README 和文档写得详细的"文档型"仓库
3. **国内平台公开文章**：围绕 1~3 个风格统一的作者/专栏收集 30~50 篇（掘金/博客园/InfoQ 等），填进自建 YAML
4. **JSON 手动导入兜底**：本地笔记/不可爬内容整理成 `data/data_warehouse_raw_data/ArticleDocument.json` 同格式 JSON，走 import 工具入库

质量原则：每篇文档约产 5 对训练样本，宁缺毋滥；作者风格尽量统一，避免"缝合怪"。

版权提醒：个人学习用途没问题；公开发布/商用训练出的模型或数据集时需注意。

---

## 第三部分：本地训练改造任务（核心）

### 3.1 现有环境（已核实）

| 项目 | 情况 |
|------|------|
| GPU | RTX 3090 24GB（24576 MiB，Ampere sm_86，支持 bf16），驱动 560.35.05 |
| 现有推理服务 | llama.cpp，systemd 服务 `llama-qwen38-27b.service`，当前跑 Qwen3.8-27B Q4_K_M GGUF（社区非官方版），OpenAI 兼容端点 `http://127.0.0.1:8001`。**计划同步切换为官方 Qwen3-32B 的 Q4_K_M GGUF，保持训练基座与推理服务一致** |
| 训练时操作 | 先 `sudo systemctl stop llama-qwen38-27b` 释放显存 |
| Python | 3.11.15 ✅ / Docker 28.0.4 ✅ / Poetry 未装 |

### 3.2 目标

- 基座模型：**`Qwen/Qwen3-32B`**（官方 dense 型号中最接近 27B 的一档；官方 Qwen3 dense 系列为 0.6B/1.7B/4B/8B/14B/32B。GGUF 不能用于训练，需从 HF 下载原始 safetensors 权重）
- 本地完成 **SFT → DPO** 全流程（QLoRA 方案），替代 `configs/training.yaml` + SageMaker
- 数据集生成与评估：继续用本地 OpenAI 兼容端点（`http://127.0.0.1:8001`）平替 OpenAI
- 显存硬约束：**24GB 以内**
- 备选：32B 跑不动则退到 **`Qwen/Qwen3-8B`**（官方）

### 3.3 现状代码链路（SageMaker 路径）

```
configs/training.yaml（finetuning_type/epochs/lr/is_dummy 等参数）
  → poetry poe run-training-pipeline
  → tools/run.py --run-training
  → pipelines/training.py（ZenML 管道）
  → steps/training/train.py
  → llm_engineering/model/finetuning/sagemaker.py::run_finetuning_on_sagemaker()
      用 sagemaker.huggingface.HuggingFace estimator 提交任务：
      - 实例 ml.g5.2xlarge（1×A10G 24GB）
      - entry_point=finetune.py，requirements=finetuning/requirements.txt
      - 环境变量注入 HF token / Comet key
  → llm_engineering/model/finetuning/finetune.py（真正干活的脚本）
```

### 3.4 finetune.py 关键事实（重要：它已经是半个 QLoRA 脚本）

读完全文后确认，**finetune.py 本身就是基于 unsloth 的 LoRA 训练脚本**，改造量比预想小：

| 位置 | 现状 | 改造需求 |
|------|------|---------|
| `load_model()` L29-57 | 已用 `FastLanguageModel.from_pretrained` + `get_peft_model`，`load_in_4bit` 参数已存在 | 只需在调用处把 `load_in_4bit=True` 传进去（当前默认 False 且无人传 True） |
| L277 | SFT 基座硬编码 `meta-llama/Llama-3.1-8B` | 参数化，改为 `Qwen/Qwen3-32B` |
| L297-298 | DPO 基座从 HF workspace 拉 `TwinLlama-3.1-8B` | 改为本地 SFT 输出目录 |
| L100-102 | SFT 数据集 = HF 拉 `{workspace}/llmtwin` + 混合 `mlabonne/FineTome-Alpaca-100k`（取 1 万条英文） | 改为本地数据源；**注意英文 Alpaca 混合数据对中文场景是否保留需决策** |
| L154 | DPO 数据集 = HF 拉 `{workspace}/llmtwin-dpo` | 改为本地数据源 |
| L256-258 | `os.environ["SM_OUTPUT_DATA_DIR"]` / `SM_MODEL_DIR` / `SM_NUM_GPUS` 是 SageMaker 注入的，本地运行**直接 KeyError** | 改为 argparse 普通参数带默认值 |
| L20-25 / L92-99 | alpaca 模板拼 instruction/output，chat_template="chatml" | Qwen 原生就是 chatml，**兼容，无需改** |
| L140 / L192 | `report_to="comet_ml"` | 本地可保留（免费档）或改 `"none"`/`"tensorboard"` |
| L218-222 | `save_model` 用 `save_pretrained_merged(..., "merged_16bit")` 合并后推 HF hub | 改为只存本地磁盘（合并 16bit 这步正好对接 llama.cpp 转 GGUF） |
| optimizer | 已是 `adamw_8bit` | 符合 QLoRA 要求，无需改；可评估换 `paged_adamw_8bit` |
| 梯度检查点 | unsloth 的 `FastLanguageModel` 默认开启 | 无需改 |

### 3.5 依赖版本问题（关键风险）

`llm_engineering/model/finetuning/requirements.txt` 是 2024 年的旧锁定：

```
torch==2.4.0 / transformers==4.43.3 / trl==0.9.6 / unsloth==2024.9.post2 / peft==0.12.0 / flash-attn==2.3.6
```

**Qwen3 于 2025 年 4 月发布，上述版本全部早于 Qwen3，无法识别 Qwen3 架构。** 本地训练环境必须升级：

- `transformers >= 4.51`（Qwen3 支持起点）
- `unsloth` 最新版（官方对 Qwen3 有专门支持）
- `trl` 升级到与新版 transformers 兼容的版本（注意 trl 新版 API 有变动，`SFTTrainer`/`DPOTrainer` 参数名可能不同，升级后需核对）
- `torch` 建议 2.4+（3090 无 sm_90 特性需求，普通 cu121/cu124 轮子即可）
- `flash-attn` 可升级或直接用 SDPA 替代
- 主项目的 `pyproject.toml` 不含训练依赖（训练依赖只在 SageMaker 容器的 requirements.txt 里），本地需单独建训练 venv 或新增 poetry group

### 3.6 改造文件清单

| 文件 | 动作 | 内容 |
|------|------|------|
| `llm_engineering/model/finetuning/finetune.py` | 修改 | 基座模型参数化；去 SM_* 环境变量；`load_in_4bit=True`；数据集改本地加载；DPO 基座指向本地 SFT 输出；保存只落盘不推 hub |
| `llm_engineering/model/finetuning/requirements.txt` | 修改 | 升级到支持 Qwen3 的版本组合 |
| `llm_engineering/model/finetuning/run_local.py`（新增） | 新增 | 本地入口：解析参数 → 调 `finetune()` → 保存；替代 SageMaker estimator 提交 |
| `steps/training/train.py` | 修改 | 增加开关：本地模式调 `run_local`，保留 SageMaker 路径 |
| `configs/training.yaml` | 修改 | 新增 `base_model`、`local: true`、`max_seq_length` 等参数 |
| `llm_engineering/application/dataset/generation.py` | 修改 | `ChatOpenAI` 加 `base_url=http://127.0.0.1:8001/v1`（或走环境变量）；L25 `tiktoken.encoding_for_model()` 对非 OpenAI 模型名会报错，改 `tiktoken.get_encoding("cl100k_base")` |
| `llm_engineering/settings.py` | 修改 | 新增 `OPENAI_BASE_URL`、`LOCAL_TRAINING` 等配置项 |
| 评估 `llm_engineering/model/evaluation/evaluate.py` | 暂缓/另议 | 现评估脚本用 vLLM 本地加载模型比答案，32B 在 24GB 上跑 vLLM 需 AWQ/GPTQ 量化；可改为调用本地 llama.cpp 端点做生成，judge 也用同端点 |

### 3.7 显存估算（24GB 硬约束）

**Qwen3-32B QLoRA（4bit NF4）：**

| 占用项 | 估算 |
|--------|------|
| 4bit 权重 | ~16.5-17.5 GB |
| LoRA 适配器（r=16/32）+ 梯度 + 8bit Adam 状态 | ~0.5-1 GB |
| 激活值（梯度检查点开，seq 2048，batch 1，unsloth 优化内核） | ~1-2 GB |
| logits/loss 缓冲（vocab ~151k，unsloth 分块计算） | ~0.5-1 GB |
| **合计** | **~19-22 GB，非常紧张，处于可行性边界** |

- SFT：建议直接从 seq_len 1024 / batch 1 / grad_accum 16 起步；seq 2048 大概率 OOM，可谨慎尝试
- **DPO 更吃显存**（chosen+rejected 双序列前向）：seq 1024 / batch 1 起步，OOM 风险高；`ref_model=None` + PEFT 模式可省掉独立参考模型（现代码已如此）
- 3090 不支持 FP8，无进一步压缩空间；实在不行开 CPU offload（速度大幅下降）
- **结论：32B 在 24GB 上属于"能跑但憋屈"，若无法接受 seq 1024 的限制，直接上 Qwen3-8B 更务实**

**Qwen3-8B 备选（宽松）：** 4bit 权重 ~5GB，总计 ~8-10GB，可 seq 4096 / batch 2-4。

### 3.8 训练时间粗估（3090 + unsloth）

假设数据集 ~1 万样本、平均 ~800 token/样本（约 8M token/epoch）：

| 方案 | 吞吐粗估 | 每 epoch | 3 epochs |
|------|---------|----------|----------|
| 32B QLoRA | ~400-850 tok/s | ~3-5.5 小时 | ~9-16 小时 |
| 8B QLoRA | ~2000-4000 tok/s | ~0.5-1 小时 | ~2-3 小时 |

建议：先 `is_dummy` 模式（400 样本 1 epoch）验证管道通畅，再决定是否上 32B 全量。

### 3.9 训练后部署闭环（与现有推理服务的衔接）

```
本地 SFT/DPO 输出（LoRA）
  → save_pretrained_merged 合并为 16bit 完整模型（落盘）
  → llama.cpp 的 convert_hf_to_gguf.py 转 GGUF
  → 量化 Q4_K_M（llama-quantize）
  → 替换 llama-qwen38-27b.service 指向的 GGUF 文件（建议同时把服务改名为与官方模型一致，如 llama-qwen3-32b.service）
  → systemctl start，推理服务即刻用上微调后的模型
```

注意：32B 合并后的 16bit 权重约 64GB 磁盘，转换过程内存峰值较高，需预留磁盘与内存（本机 78GB 内存足够，磁盘剩余 ~111GB **不够宽裕，必须先清理或外挂存储**）。

### 3.10 待确认事项

1. ~~Qwen3.8-27B 的原始 safetensors HF repo id~~ **已解决**：改用官方 `Qwen/Qwen3-32B`，权重直接从官方仓库下载（走 hf-mirror）；推理服务侧同步换用官方 Qwen3-32B 的 Q4_K_M GGUF 以保持一致
2. 新版 unsloth 对 Qwen3-32B 的支持情况（装环境时验证 `FastLanguageModel.from_pretrained` 能加载；unsloth 官方已宣称支持 Qwen3 全系）
3. SFT 是否保留 FineTome-Alpaca 英文混合数据（中文场景建议去掉或换中文通用指令集）
4. Comet ML 追踪保留还是改 tensorboard
5. 评估环节方案（vLLM 量化加载 vs 直接调 llama.cpp 端点）

### 3.11 风险登记

| 风险 | 等级 | 应对 |
|------|------|------|
| 32B QLoRA 在 24GB OOM | 很高 | seq_len 1024 / batch 1 起步；退 Qwen3-8B 备选（用户已接受） |
| 旧依赖不支持 Qwen3 | 高 | 升级 requirements（见 3.5），装环境时先验证模型可加载 |
| trl 新版 API 变动导致脚本报错 | 中 | 升级后核对 SFTTrainer/DPOTrainer 参数 |
| DPO 显存比 SFT 更高 | 中 | seq 1024 + batch 1 起步 |
| 磁盘不足（54GB 合并权重 + GGUF 转换中间产物） | 中 | 训练前清理磁盘或挂存储 |
| 本地端点生成数据集质量不如 GPT-4o-mini | 中 | 32B 本地模型生成质量大概率更好，但需验证 JSON 格式遵从度（ListPydanticOutputParser 解析失败会丢 batch） |

---

## 第四部分：整体路线图

```
阶段 0 环境准备
  ├─ 装 Poetry → poetry install --without aws
  ├─ cp .env.example .env（HF token 走 hf-mirror 也需要）
  ├─ poetry poe local-infrastructure-up（MongoDB + Qdrant + ZenML）
  └─ export HF_ENDPOINT=https://hf-mirror.com

阶段 1 数据（不依赖训练改造，可先行）
  ├─ 1a. poetry poe run-import-data-warehouse-from-json（用作者自带数据跑通）
  ├─ 1b. 自建 YAML 收集国内公开文章 + GitHub 仓库 → run-digital-data-etl
  └─ run-feature-engineering-pipeline

阶段 2 数据集生成（改 generation.py 指向本地端点）
  ├─ run-generate-instruct-datasets-pipeline
  └─ run-generate-preference-datasets-pipeline

阶段 3 本地训练改造（本备忘录第三部分，需先确认方案）
  ├─ 升级训练依赖 → 验证 Qwen3-32B 可 4bit 加载
  ├─ is_dummy 跑通 SFT → 全量 SFT → DPO
  └─ 合并 → 转 GGUF → Q4_K_M → 部署回推理服务（同步换官方 Qwen3-32B GGUF）

阶段 4 RAG 与推理服务
  └─ run-inference-ml-service（FastAPI + RAG，LLM 调用指向本地端点）
```

---

## 附：关键文件索引

| 主题 | 路径 |
|------|------|
| ETL 管道 | `pipelines/digital_data_etl.py`、`steps/etl/` |
| 爬虫 | `llm_engineering/application/crawlers/`（dispatcher/base/medium/github/custom_article/linkedin） |
| 数据集生成 | `llm_engineering/application/dataset/generation.py`（prompt 在 L173/L210） |
| 训练脚本 | `llm_engineering/model/finetuning/finetune.py` |
| SageMaker 提交 | `llm_engineering/model/finetuning/sagemaker.py` |
| 训练依赖 | `llm_engineering/model/finetuning/requirements.txt` |
| 训练配置 | `configs/training.yaml` |
| 评估 | `llm_engineering/model/evaluation/evaluate.py` |
| 全局配置 | `llm_engineering/settings.py`、`.env.example` |
| 自带数据 | `data/data_warehouse_raw_data/`、`data/artifacts/` |
| 任务命令 | `pyproject.toml` 的 `[tool.poe.tasks]` |
