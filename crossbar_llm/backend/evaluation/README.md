# Evaluation Pipeline 评测指南

本文档说明如何使用合并后的评测 pipeline 对 LLM 模型进行评测。

This guide explains how to evaluate LLM models using the unified evaluation pipeline.

## 目录

- [总览](#总览)
- [前置条件](#前置条件)
- [评测架构](#评测架构)
- [评测方式一：批量 Pipeline（推荐用于正式评测）](#评测方式一批量-pipeline推荐用于正式评测)
- [评测方式二：轻量级 CLI Pipeline](#评测方式二轻量级-cli-pipeline)
- [评测方式三：Python API](#评测方式三python-api)
- [完整评测示例](#完整评测示例)
- [输出结果说明](#输出结果说明)
- [配置参考](#配置参考)
- [故障排除](#故障排除)

---

## 总览

评测系统由两套 pipeline 组成，统一在 `evaluation/` 包下，共享相同的评判引擎（`AnswerEvaluator`）：

| Pipeline | 入口文件 | 适用场景 |
|----------|---------|---------|
| **批量 Pipeline** | `batch_pipeline.py` + `scripts/run_batch_test.sh` | 多模型并行评测、正式评测、生产环境 |
| **轻量级 Pipeline** | `evaluation/run_pipeline.py` | 单模型快速验证、开发调试、dry-run |

两套 pipeline 的核心流程相同：

```
加载测试集 → 模型推理 → LLM-as-Judge 评判 → 生成评测报告
```

## 前置条件

### 1. 安装依赖

```bash
# 使用 poetry（推荐）
poetry install

# 或使用 pip
pip install -r requirements.txt
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件（参考 `.env.example`），配置所需的 API key：

```bash
# 根据使用的模型提供商，配置对应的 key
OPENROUTER_API_KEY=sk-or-...
OPENAI_API_KEY=sk-...
GEMINI_API_KEY=...
```

### 3. 准备测试集

测试集支持 **JSONL**、**JSON**、**CSV** 三种格式。推荐使用 JSONL 格式。

**JSONL 格式示例**（每行一个 JSON 对象）：

```jsonl
{"instruction": "Does this protein possess transmembrane regions?", "input": "<protein>MFASCH...</protein>", "output": "Expected answer", "rationale": "Explanation..."}
{"instruction": "Identify the post-translational modifications.", "input": "<protein>MSSHKT...</protein>", "output": "Expected answer", "rationale": "Because..."}
```

> **字段说明**：`instruction` + `input` 会被自动拼接为完整问题；`output` 为期望答案；`rationale` 为评判参考依据。

---

## 评测架构

```
┌──────────────────────────────────────────────────────────────────────────────┐
│              统一评测包 (evaluation/)                                         │
│                                                                              │
│  ┌─────────────────────────────────┐  ┌──────────────────────────────────┐   │
│  │   evaluation/run_pipeline.py    │  │      batch_pipeline.py           │   │
│  │   (轻量级单模型评测)             │  │   (生产级多模型批量测试)          │   │
│  │                                 │  │                                  │   │
│  │ • 单模型评测                     │  │ • 多模型并行评测                  │   │
│  │ • CLI 命令行驱动                 │  │ • YAML 配置驱动                  │   │
│  │ • 支持 --dry-run                │  │ • 配置热重载                     │   │
│  │ • 支持 --config 读取配置         │  │ • 连接真实 KG (Neo4j)            │   │
│  │ • 输出: 单个 JSON 报告           │  │ • 集成 LLM-as-judge             │   │
│  │                                 │  │ • 输出: 多文件报告               │   │
│  └─────────────────────────────────┘  └──────────────────────────────────┘   │
│                                                                              │
│  共享模块:                                                                    │
│  ├── answer_evaluator.py   (LLM-as-judge 评分引擎)                           │
│  ├── compare_results.py    (多模型结果对比)                                   │
│  ├── evaluate_results.py   (LLM-as-judge 后处理)                             │
│  └── test_loader.py        (测试数据加载器)                                   │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 评测方式一：批量 Pipeline（推荐用于正式评测）

适用于多模型并行评测，使用 YAML 配置文件驱动，支持连接真实知识图谱 (Neo4j)。

### 步骤 1：修改配置文件

编辑 `config/batch_config.yaml`，设置测试集路径和评测参数：

```yaml
# =============================================================================
# 模型配置
# =============================================================================
provider: "OpenRouter"
models:
  - "deepseek-v3-2-251201"
  # - "Qwen/Qwen3-32B"       # 取消注释以添加更多模型

# =============================================================================
# 测试数据源 - 设置测试集路径
# =============================================================================
questions:
  benchmark:
    enabled: true
    file: "/mnt/vepfs/users/clzeng/workspace/BioBenchmark/in-house/drug_target_disease_v1/drugtgt01.jsonl"
    indices: []       # 留空表示测试所有问题；填入 [1, 2, 3] 可只测试前三题
    question_ids: []  # 也可以按 question_id 筛选

# =============================================================================
# 执行设置
# =============================================================================
execution:
  parallel: true
  max_workers: 4
  request_interval: 1.0
  retry:
    max_attempts: 3
    backoff_factor: 2.0
    initial_delay: 1.0

# =============================================================================
# 多步查询设置
# =============================================================================
multi_step:
  enabled: true
  max_steps: 5
  max_failures: 5
  min_results: 1

# =============================================================================
# 多跳推理设置（启用后覆盖 multi_step）
# =============================================================================
multi_hop:
  enabled: false       # 改为 true 以启用多跳推理
  max_steps: 5

# =============================================================================
# LLM-as-Judge 评判设置
# =============================================================================
judge:
  enabled: true
  model: "gpt-oss-120b"
  temperature: 0
  max_tokens: 2560

# =============================================================================
# 输出设置
# =============================================================================
output:
  base_dir: "batch_output"
  save_debug_logs: true
  save_raw_responses: true
```

### 步骤 2：运行评测

**方法 A：使用一键脚本（推荐）**

```bash
cd /path/to/CROssBAR_LLM
./scripts/run_batch_test.sh
```

该脚本会自动依次执行：
1. `batch_pipeline.py` — 对所有模型执行推理
2. `compare_results.py` — 生成多模型对比报告
3. `evaluate_results.py` — 使用 LLM-as-Judge 评判

**方法 B：手动执行**

```bash
cd /path/to/CROssBAR_LLM/crossbar_llm/backend

# 1. 运行批量推理
python batch_pipeline.py --config ../../config/batch_config.yaml --verbose

# 2. 生成对比报告（自动找到最新运行目录）
python compare_results.py --run-dir ../../batch_output/run_<timestamp> --format both

# 3. 运行 LLM-as-Judge 评判
python evaluate_results.py --run-dir ../../batch_output/run_<timestamp>
```

### 步骤 3：查看结果

评测完成后，结果保存在 `batch_output/run_<timestamp>/` 目录下：

```
batch_output/run_20260212_143000/
├── deepseek-v3-2-251201/
│   └── results.json              # 该模型的详细推理结果
├── results_summary.json          # 汇总报告（含 judge 评分）
├── results_by_question.md        # 按问题维度的对比报告
├── results_by_model.md           # 按模型维度的对比报告
└── logs/
    └── batch_run_*.log           # 运行日志
```

---

## 评测方式二：轻量级 CLI Pipeline

适用于单模型快速评测、开发调试和验证。

### 基本用法

```bash
cd /path/to/CROssBAR_LLM/crossbar_llm/backend

python -m evaluation.run_pipeline \
    --dataset /mnt/vepfs/users/clzeng/workspace/BioBenchmark/in-house/drug_target_disease_v1/drugtgt01.jsonl \
    --output ../../evaluation_report.json \
    --model-name deepseek-v3-2-251201 \
    --judge-model gpt-oss-120b
```

### 从配置文件加载设置

可以直接复用 `batch_config.yaml` 中的模型和 judge 设置：

```bash
cd /path/to/CROssBAR_LLM/crossbar_llm/backend

python -m evaluation.run_pipeline \
    --dataset /mnt/vepfs/users/clzeng/workspace/BioBenchmark/in-house/drug_target_disease_v1/drugtgt01.jsonl \
    --output ../../evaluation_report.json \
    --config ../../config/batch_config.yaml
```

> **提示**：`--config` 会读取配置文件中的模型名称和 judge 设置。CLI 参数（`--model-name`、`--judge-model` 等）优先于配置文件中的值。

### 启用多跳推理

```bash
cd /path/to/CROssBAR_LLM/crossbar_llm/backend

python -m evaluation.run_pipeline \
    --dataset /mnt/vepfs/users/clzeng/workspace/BioBenchmark/in-house/drug_target_disease_v1/drugtgt01.jsonl \
    --output ../../evaluation_report.json \
    --model-name deepseek-v3-2-251201 \
    --judge-model gpt-oss-120b \
    --multi-hop \
    --multi-hop-max-steps 5
```

### Dry-Run 模式（无需 API key）

用 mock 函数验证 pipeline 是否正常工作：

```bash
cd /path/to/CROssBAR_LLM/crossbar_llm/backend

python -m evaluation.run_pipeline \
    --dataset /mnt/vepfs/users/clzeng/workspace/BioBenchmark/in-house/drug_target_disease_v1/drugtgt01.jsonl \
    --output ../../evaluation_report.json \
    --dry-run
```

### CLI 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--dataset`, `-d` | string | (必填) | 测试数据集路径（JSONL/JSON/CSV） |
| `--output`, `-o` | string | (必填) | 评测报告输出路径（JSON） |
| `--model-name`, `-m` | string | `gemini-3-flash-preview` | 推理模型名称 |
| `--judge-model`, `-j` | string | `gemini-3-flash-preview` | Judge 模型名称 |
| `--config`, `-c` | string | (无) | 从 `batch_config.yaml` 加载设置 |
| `--multi-hop` | flag | `false` | 启用多跳推理 |
| `--multi-hop-max-steps` | int | `5` | 最大推理步数 |
| `--dry-run` | flag | `false` | 使用 mock 函数运行（无需 API key） |

---

## 评测方式三：Python API

适用于需要自定义推理函数或将评测嵌入其他系统的场景。

```python
from crossbar_llm.backend.evaluation import (
    TestDatasetLoader, EvaluationRunner, AnswerEvaluator, run_pipeline
)

# ---- 方式 A：一键评测 ----
report = run_pipeline(
    dataset_path="/mnt/vepfs/users/clzeng/workspace/BioBenchmark/in-house/drug_target_disease_v1/drugtgt01.jsonl",
    output_path="evaluation_report.json",
    model_name="deepseek-v3-2-251201",
    judge_model="gpt-oss-120b",
)

# ---- 方式 B：分步控制 ----
# 1. 加载测试数据
loader = TestDatasetLoader(
    "/mnt/vepfs/users/clzeng/workspace/BioBenchmark/in-house/drug_target_disease_v1/drugtgt01.jsonl"
)
questions = loader.get_questions()

# 2. 定义推理函数并运行
def my_model(question: str) -> dict:
    # 替换为你的模型推理逻辑
    return {"answer": "模型生成的答案"}

runner = EvaluationRunner(model_inference_fn=my_model, model_name="my-model")
results = runner.run_batch(
    questions,
    progress_callback=lambda cur, total: print(f"进度: {cur}/{total}")
)

# 3. LLM-as-Judge 评判
def llm_judge(prompt: str) -> str:
    # 替换为你的 judge LLM 调用逻辑
    return '{"pass": true, "reason": "correct", "novelty_score": 5, "reasoning_similarity_score": 5}'

evaluator = AnswerEvaluator(llm_judge_fn=llm_judge)
for r in results:
    score = evaluator.evaluate(
        question=r["question"],
        model_answer=r.get("model_answer", ""),
        expected=r.get("expected", ""),
        rationale=r.get("rationale", ""),
        trace=r.get("trace"),
    )
    print(f"Pass: {score['pass']}, Novelty: {score['novelty_score']}/10")
```

---

## 完整评测示例

以下是一个完整的端到端评测示例，假设测试集路径为：

```
/mnt/vepfs/users/clzeng/workspace/BioBenchmark/in-house/drug_target_disease_v1/drugtgt01.jsonl
```

### 使用批量 Pipeline

```bash
# 1. 进入项目根目录
cd /path/to/CROssBAR_LLM

# 2. 编辑配置文件，设置测试集路径
#    修改 config/batch_config.yaml 中的 questions.benchmark.file 为:
#    /mnt/vepfs/users/clzeng/workspace/BioBenchmark/in-house/drug_target_disease_v1/drugtgt01.jsonl

# 3. 确认模型列表（可在 models 下添加或删除模型）

# 4. 一键运行评测
./scripts/run_batch_test.sh

# 5. 查看结果
#    评测完成后，终端会输出 judge 统计摘要
#    详细报告保存在 batch_output/run_<timestamp>/ 目录中
cat batch_output/run_*/results_by_question.md
```

### 使用轻量级 Pipeline

```bash
# 1. 进入 backend 目录
cd /path/to/CROssBAR_LLM/crossbar_llm/backend

# 2. 先用 dry-run 验证 pipeline 能正常加载测试集
python -m evaluation.run_pipeline \
    --dataset /mnt/vepfs/users/clzeng/workspace/BioBenchmark/in-house/drug_target_disease_v1/drugtgt01.jsonl \
    --output ../../evaluation_report.json \
    --dry-run

# 3. 确认无误后，使用真实模型评测
python -m evaluation.run_pipeline \
    --dataset /mnt/vepfs/users/clzeng/workspace/BioBenchmark/in-house/drug_target_disease_v1/drugtgt01.jsonl \
    --output ../../evaluation_report.json \
    --model-name deepseek-v3-2-251201 \
    --judge-model gpt-oss-120b

# 4. 查看评测报告
python -c "
import json
with open('../../evaluation_report.json') as f:
    report = json.load(f)
js = report['judge_summary']
print(f\"Pass rate: {js['pass']}/{js['total']} ({js['pass_rate']:.1%})\")
print(f\"Avg novelty: {js['avg_novelty_score']}/10\")
print(f\"Avg reasoning similarity: {js['avg_reasoning_similarity_score']}/10\")
"
```

---

## 输出结果说明

### 批量 Pipeline 输出

| 文件 | 说明 |
|------|------|
| `<model>/results.json` | 每个模型的详细推理结果（含每题答案、查询、耗时等） |
| `results_summary.json` | 汇总报告，包含所有模型的 judge 评分统计 |
| `results_by_question.md` | Markdown 格式，按问题维度对比各模型的表现 |
| `results_by_model.md` | Markdown 格式，按模型维度汇总评测指标 |
| `logs/batch_run_*.log` | 详细运行日志 |

### 轻量级 Pipeline 输出

输出为单个 JSON 报告，包含：

```json
{
  "generated_at": "2026-02-12T14:30:00",
  "model_name": "deepseek-v3-2-251201",
  "judge_model": "gpt-oss-120b",
  "dataset_path": "/mnt/.../drugtgt01.jsonl",
  "run_summary": {
    "total_questions": 50,
    "successful": 48,
    "failed": 2,
    "success_rate": 0.96
  },
  "judge_summary": {
    "total": 48,
    "pass": 40,
    "fail": 8,
    "pass_rate": 0.8333,
    "avg_novelty_score": 6.5,
    "avg_reasoning_similarity_score": 7.2
  },
  "results": [
    {
      "question_index": 1,
      "question": "...",
      "model_answer": "...",
      "expected": "...",
      "judge": {
        "pass": true,
        "reason": "...",
        "novelty_score": 7,
        "reasoning_similarity_score": 8
      }
    }
  ]
}
```

### 评分维度说明

| 维度 | 范围 | 说明 |
|------|------|------|
| **Pass/Fail** | bool | 模型输出是否在语义上匹配期望答案 |
| **Novelty Score** | 0–10 | 相比期望答案，模型回答的原创性和额外信息量 |
| **Reasoning Similarity Score** | 0–10 | 模型推理过程与期望 rationale 的相似度 |

---

## 配置参考

### `config/batch_config.yaml` 完整配置项

| 配置项 | 说明 | 示例值 |
|--------|------|--------|
| `provider` | 模型提供商 | `"OpenRouter"` |
| `models` | 要评测的模型列表 | `["deepseek-v3-2-251201"]` |
| `questions.benchmark.enabled` | 是否使用 benchmark 文件 | `true` |
| `questions.benchmark.file` | 测试集文件路径 | `"/mnt/.../drugtgt01.jsonl"` |
| `questions.benchmark.indices` | 按索引筛选题目（1-based），留空测试所有题 | `[1, 2, 3]` 或 `[]` |
| `questions.benchmark.question_ids` | 按 ID 筛选题目 | `["cd787a9f"]` 或 `[]` |
| `execution.parallel` | 是否并行执行多模型 | `true` |
| `execution.max_workers` | 最大并发数 | `4` |
| `execution.request_interval` | API 调用间隔（秒） | `1.0` |
| `multi_step.enabled` | 是否启用多步查询 | `true` |
| `multi_step.max_steps` | 最大查询步数 | `5` |
| `multi_hop.enabled` | 是否启用多跳推理（覆盖 multi_step） | `false` |
| `multi_hop.max_steps` | 多跳推理最大步数 | `5` |
| `judge.enabled` | 是否启用 LLM-as-Judge | `true` |
| `judge.model` | Judge 使用的模型 | `"gpt-oss-120b"` |
| `judge.temperature` | Judge 采样温度 | `0` |
| `judge.max_tokens` | Judge 最大输出 token | `2560` |
| `output.base_dir` | 输出根目录 | `"batch_output"` |

### 测试数据格式

测试集支持 `instruction` + `input` 格式（自动拼接为完整问题）和直接的 `question` 格式：

```jsonl
{"instruction": "问题描述", "input": "输入数据", "output": "期望答案", "rationale": "推理依据"}
{"question": "完整问题文本", "output": "期望答案", "rationale": "推理依据"}
```

---

## 故障排除

| 问题 | 解决方案 |
|------|---------|
| `FileNotFoundError` | 检查测试集路径是否正确，建议使用绝对路径 |
| `Unsupported file format` | 确保文件扩展名为 `.jsonl`、`.json` 或 `.csv` |
| API key 相关错误 | 检查 `.env` 文件中是否正确配置了对应的 API key |
| Judge 输出解析错误 | 确保 `judge.max_tokens` 足够大（建议 ≥ 256） |
| 模型推理超时 | 增加 `execution.retry.max_attempts`，或减小并发数 |
| 新颖性评分偏低 | 模型可能在重复期望答案，调整 temperature |
| 批量 Pipeline 找不到配置 | 确保从项目根目录运行 `./scripts/run_batch_test.sh` |
| `ModuleNotFoundError` | 确保在 `crossbar_llm/backend` 目录下运行 CLI 命令 |

---

## License

本评测 pipeline 是 CROssBAR-LLM 项目的一部分，采用 GPL-3.0 许可证。
