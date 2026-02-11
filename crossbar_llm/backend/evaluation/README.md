# Evaluation Pipeline

模块化评测 pipeline，用于测试和评分 LLM 模型输出，包含完整的**多跳推理 (Multi-Hop Reasoning)** 支持。

A modular evaluation pipeline for testing and scoring LLM model outputs, with full **multi-hop reasoning** support.

## 目录 (Table of Contents)

- [总览 (Overview)](#总览-overview)
- [架构 (Architecture)](#架构-architecture)
- [安装 (Installation)](#安装-installation)
- [快速开始 (Quick Start)](#快速开始-quick-start)
- [多跳推理 (Multi-Hop Reasoning)](#多跳推理-multi-hop-reasoning)
  - [什么是多跳推理？](#什么是多跳推理)
  - [通过 FastAPI 使用多跳推理](#通过-fastapi-使用多跳推理)
  - [通过评测 Pipeline 测评多跳推理](#通过评测-pipeline-测评多跳推理)
  - [多跳推理验证测例](#多跳推理验证测例)
- [模块文档 (Module Documentation)](#模块文档-module-documentation)
- [完整示例 (Complete Example)](#完整示例-complete-example)
- [配置 (Configuration)](#配置-configuration)
- [错误处理 (Error Handling)](#错误处理-error-handling)
- [最佳实践 (Best Practices)](#最佳实践-best-practices)
- [故障排除 (Troubleshooting)](#故障排除-troubleshooting)

## 总览 (Overview)

评测 pipeline 的功能：
1. 从多种格式（JSONL、JSON、CSV）加载测试数据集
2. 对测试问题执行模型推理（支持普通推理和多跳推理）
3. 使用 LLM-as-judge 方法评测模型答案，包含新颖性和推理相似度评分
4. 提供一键式命令行 pipeline（`run_pipeline`），串联上述三步并生成报告

The evaluation pipeline:
1. Loads test datasets from various formats (JSONL, JSON, CSV)
2. Runs model inference on test questions (supports both standard and multi-hop reasoning)
3. Evaluates model answers using an LLM-as-judge approach with novelty and reasoning similarity scoring
4. Provides a one-command CLI pipeline (`run_pipeline`) that chains all three steps and generates a report

## 架构 (Architecture)

Pipeline 由四个组件构成：

```
┌─────────────────────────────────────────────────────────────┐
│                    run_pipeline.py                           │
│         (一键式 pipeline / One-command pipeline)              │
│                                                             │
│  ┌───────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │ TestDataset   │→ │ EvaluationRunner │→ │ Answer       │  │
│  │ Loader        │  │                  │  │ Evaluator    │  │
│  │               │  │ multi_hop: ✓     │  │              │  │
│  │ Formats:      │  │ evidence: ✓     │  │ trace-aware  │  │
│  │ JSONL/JSON/CSV│  │ trace: ✓        │  │ judging: ✓   │  │
│  └───────────────┘  └──────────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

| 组件 (Module)                        | 文件 (File)              | 职责 (Responsibility)                                    |
|--------------------------------------|--------------------------|----------------------------------------------------------|
| **TestDatasetLoader**                | `test_loader.py`         | 加载并标准化测试数据，解析 `multi_hop` 标记               |
| **EvaluationRunner**                 | `evaluation_runner.py`   | 运行模型推理，收集结果（含 evidence / trace / hop_count）  |
| **AnswerEvaluator**                  | `answer_evaluator.py`    | LLM-as-judge 评分（感知多跳 trace）                       |
| **run_pipeline** (CLI)               | `run_pipeline.py`        | 串联以上三步，生成 JSON 报告；支持 `--dry-run`             |

## 安装 (Installation)

评测模块是 CROssBAR-LLM 的一部分，安装依赖：

```bash
# 使用 poetry
poetry install

# 或使用 pip
pip install -r requirements.txt
```

## 快速开始 (Quick Start)

### Python API

```python
from crossbar_llm.backend.evaluation import (
    TestDatasetLoader, EvaluationRunner, AnswerEvaluator
)

# 1. 加载测试数据集
loader = TestDatasetLoader("data/questions.json")
questions = loader.get_questions()

# 2. 定义模型推理函数
def my_model(question: str) -> dict:
    return {"answer": "模型生成的答案"}

# 3. 运行推理
runner = EvaluationRunner(
    model_inference_fn=my_model,
    model_name="my-model",
    output_dir="results"
)
results = runner.run_batch(questions)

# 4. LLM-as-judge 评测
def llm_judge(prompt: str) -> str:
    # 调用你的 LLM API
    return '{"pass": true, "reason": "correct", ...}'

evaluator = AnswerEvaluator(llm_judge_fn=llm_judge)
for r in results:
    score = evaluator.evaluate(
        question=r["question"],
        model_answer=r["model_answer"],
        expected=r["expected"],
        rationale=r["rationale"],
        trace=r.get("trace"),  # 多跳推理 trace（如有）
    )
    print(f"Pass: {score['pass']}, Novelty: {score['novelty_score']}/10")
```

### 命令行 (CLI)

```bash
# 使用真实 LLM（需要 .env 中的 API key）
cd crossbar_llm/backend
python -m evaluation.run_pipeline \
    --dataset ../../questions.json \
    --output ../../evaluation_report.json \
    --model-name gemini-3-flash-preview \
    --judge-model gemini-3-flash-preview

# 使用 mock 函数进行试运行（不需要 API key）
python -m evaluation.run_pipeline \
    --dataset ../../questions.json \
    --output ../../evaluation_report.json \
    --dry-run
```

### 一键式 Pipeline 函数

```python
from crossbar_llm.backend.evaluation import run_pipeline

report = run_pipeline(
    dataset_path="questions.json",
    output_path="report.json",
    model_inference_fn=my_model_fn,   # 可选，None 时自动构建
    llm_judge_fn=my_judge_fn,         # 可选，None 时自动构建
    model_name="gemini-3-flash-preview",
    judge_model="gemini-3-flash-preview",
)
# report 包含: run_summary, judge_summary, results
```

---

## 多跳推理 (Multi-Hop Reasoning)

### 什么是多跳推理？

多跳推理是一种知识图谱推理策略，LLM 在每一步根据已有证据决定下一步行为：

| Action | 名称      | 说明                                              |
|--------|-----------|---------------------------------------------------|
| **A**  | CONTINUE  | 继续探索当前节点（不同属性或关系）                  |
| **B**  | JUMP      | 跳转到另一个节点（需指定 `node_type` + `identifier`）|
| **C**  | ANSWER    | 已收集足够证据，终止并给出最终答案                  |
| **D**  | OVERVIEW  | 不聚焦于特定节点，执行全局概览查询                  |

每一步产生一条 trace 记录，包含 `step`、`action`、`reason`、`status`，以及可选的 `cypher`、`result_count`、`jump_target` 等字段。

决策 prompt 模板定义在 `tools/multi_hop_utils.py` 中的 `MULTI_HOP_DECISION_TEMPLATE`。

### 通过 FastAPI 使用多跳推理

FastAPI 后端提供了 `POST /run_multi_hop/` 端点。

#### 请求格式

```bash
curl -X POST http://localhost:8000/run_multi_hop/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What drugs target BRCA1 through its associated pathways?",
    "llm_type": "gemini-3-flash-preview",
    "api_key": "env",
    "provider": "google",
    "max_steps": 5,
    "top_k": 5,
    "verbose": false
  }'
```

**请求参数说明 (Request Parameters)：**

| 参数         | 类型   | 必填 | 默认值 | 说明                                          |
|-------------|--------|------|--------|-----------------------------------------------|
| `question`  | string | ✅   | —      | 需要多跳推理的问题                             |
| `llm_type`  | string | ✅   | —      | 使用的 LLM 模型名称                            |
| `api_key`   | string | ✅   | —      | API 密钥，传 `"env"` 则使用 `.env` 中配置的密钥 |
| `provider`  | string | ❌   | `null` | 模型提供商（如 `"google"`, `"openai"`），使用 `"env"` api_key 时推荐提供 |
| `max_steps` | int    | ❌   | `5`    | 最大推理步数                                   |
| `top_k`     | int    | ❌   | `5`    | 每步查询返回的结果数                            |
| `verbose`   | bool   | ❌   | `false`| 是否启用详细日志                               |

#### 响应格式

```json
{
  "answer": "Aspirin and Tamoxifen target BRCA1 through DNA repair pathways.",
  "evidence": [
    {"drug": "Aspirin", "pathway": "DNA Repair"},
    {"drug": "Tamoxifen", "pathway": "Estrogen Signaling"}
  ],
  "trace": [
    {
      "step": 1,
      "action": "B",
      "reason": "需要查询 BRCA1 蛋白的相关通路",
      "status": "jump",
      "jump_target": {"node_type": "Protein", "identifier": "BRCA1"},
      "cypher": "MATCH (p:Protein {name: 'BRCA1'})-[:ASSOCIATED_WITH]->(pw:Pathway) RETURN pw",
      "result_count": 3
    },
    {
      "step": 2,
      "action": "B",
      "reason": "跳转到 DNA Repair 通路查看靶向药物",
      "status": "jump",
      "jump_target": {"node_type": "Pathway", "identifier": "DNA Repair"},
      "cypher": "MATCH (d:Drug)-[:TARGETS]->(pw:Pathway {name: 'DNA Repair'}) RETURN d",
      "result_count": 2
    },
    {
      "step": 3,
      "action": "C",
      "reason": "已收集到足够的药物-通路关系证据",
      "status": "terminate"
    }
  ],
  "logs": "..."
}
```

**响应字段说明 (Response Fields)：**

| 字段       | 说明                                                 |
|-----------|------------------------------------------------------|
| `answer`  | 最终自然语言答案                                      |
| `evidence`| 推理过程中收集的所有证据列表                           |
| `trace`   | 推理路径的完整 trace，每一步包含 action/reason/status  |
| `logs`    | 服务端日志（verbose=true 时包含详细信息）               |

#### Python 调用示例

```python
import requests

response = requests.post(
    "http://localhost:8000/run_multi_hop/",
    json={
        "question": "What drugs target BRCA1 through its associated pathways?",
        "llm_type": "gemini-3-flash-preview",
        "api_key": "env",
        "provider": "google",
        "max_steps": 5,
    },
)
data = response.json()

# 查看推理路径
for step in data["trace"]:
    print(f"Step {step['step']}: [{step['action']}] {step['reason']}")

# 查看收集的证据
print(f"\nEvidence count: {len(data['evidence'])}")
print(f"Answer: {data['answer']}")
```

### 通过评测 Pipeline 测评多跳推理

有两种方式将多跳推理融入评测：

#### 方式 1：通过 `batch_config.yaml` 启用多跳推理

在 `config/batch_config.yaml` 中启用 `multi_hop`：

```yaml
# Multi-hop Reasoning Settings
# 启用后，LLM 在每一步决定：
#   A. 继续探索当前节点
#   B. 跳转到其他节点
#   C. 停止并给出最终答案
#   D. 执行全局概览查询
# 启用后会覆盖 multi_step 设置
multi_hop:
  enabled: true    # 改为 true 以启用
  max_steps: 5     # 最大推理步数
```

然后运行批量 pipeline：

```bash
cd crossbar_llm/backend
python batch_pipeline.py
```

#### 方式 2：通过测试数据集标记多跳问题

在测试数据集中，使用 `"multi_hop": true` 标记需要多跳推理的问题：

```json
[
  {
    "question": "What drugs target BRCA1 through its associated pathways?",
    "output": "Aspirin, Tamoxifen",
    "rationale": "BRCA1 is involved in DNA repair pathways, which are targeted by these drugs.",
    "multi_hop": true
  },
  {
    "question": "What is the gene symbol for TP53?",
    "output": "TP53",
    "multi_hop": false
  }
]
```

使用 `run_pipeline` 评测时，pipeline 会自动：
- `TestDatasetLoader` 解析 `multi_hop` 字段
- `EvaluationRunner` 在结果中记录 `evidence`、`trace`、`hop_count`（当模型推理函数返回这些字段时）
- `AnswerEvaluator` 将多跳 trace 附加到 judge prompt 中进行评分

```python
from crossbar_llm.backend.evaluation import run_pipeline

# 模型推理函数返回多跳推理结果
def multi_hop_model(question: str) -> dict:
    return {
        "answer": "Aspirin targets BRCA1 through DNA repair",
        "evidence": [{"drug": "Aspirin", "pathway": "DNA Repair"}],
        "trace": [
            {"step": 1, "action": "B", "status": "jump",
             "reason": "explore BRCA1 pathways",
             "jump_target": {"node_type": "Protein", "identifier": "BRCA1"}},
            {"step": 2, "action": "C", "status": "terminate",
             "reason": "sufficient evidence collected"},
        ],
    }

report = run_pipeline(
    dataset_path="multi_hop_questions.json",
    output_path="multi_hop_report.json",
    model_inference_fn=multi_hop_model,
    llm_judge_fn=my_judge_fn,
    model_name="my-model",
    judge_model="gpt-4",
)

# 查看多跳推理特有的结果字段
for r in report["results"]:
    if r.get("multi_hop"):
        print(f"Question: {r['question']}")
        print(f"  Hop count: {r['hop_count']}")
        print(f"  Evidence items: {len(r['evidence'])}")
        print(f"  Judge pass: {r['judge']['pass']}")
```

#### 方式 3：命令行 dry-run 快速验证

```bash
cd crossbar_llm/backend
python -m evaluation.run_pipeline \
    --dataset multi_hop_questions.json \
    --output multi_hop_report.json \
    --dry-run
```

### 多跳推理验证测例

项目中提供了完整的多跳推理单元测试，位于 `test/` 目录：

| 测试文件                         | 测试内容                               |
|----------------------------------|---------------------------------------|
| `test/test_multi_hop_reasoning.py`  | 多跳推理核心逻辑（决策解析、证据汇总、推理循环、trace 生成） |
| `test/test_multi_hop_evaluation.py` | 评测 pipeline 的多跳支持（数据加载、Runner 结果字段、Judge trace 感知） |

运行测试：

```bash
cd /path/to/CROssBAR_LLM
python -m pytest test/test_multi_hop_reasoning.py test/test_multi_hop_evaluation.py -v
```

#### 设计多跳推理验证测例示例

以下是验证多跳推理路径的测例设计模式（不需要真实 LLM 或 Neo4j 连接）：

```python
import json
import unittest
from unittest.mock import MagicMock

def test_drug_protein_pathway_multi_hop():
    """
    验证多跳推理路径：Drug → Protein → Pathway
    场景：查询 "哪些药物通过相关通路靶向 BRCA1？"
    期望路径：
      Step 1: JUMP to Protein:BRCA1（查询蛋白信息）
      Step 2: JUMP to Pathway:DNA_Repair（查询相关通路）
      Step 3: ANSWER（收集足够证据，给出答案）
    """
    # 预设 LLM 的多跳决策序列
    decisions = [
        {
            "action": "B",
            "reason": "需要查询 BRCA1 蛋白的信息",
            "jump_target": {"node_type": "Protein", "identifier": "BRCA1"},
        },
        {
            "action": "B",
            "reason": "跳转到 DNA Repair 通路查看靶向药物",
            "jump_target": {"node_type": "Pathway", "identifier": "DNA_Repair"},
        },
        {
            "action": "C",
            "reason": "已收集到足够证据",
        },
    ]

    # 模拟知识图谱查询结果
    mock_neo4j = MagicMock()
    mock_neo4j.execute_query.side_effect = [
        [{"protein": "BRCA1", "function": "DNA Repair"}],       # Step 1
        [{"drug": "Aspirin"}, {"drug": "Tamoxifen"}],            # Step 2
    ]

    # 运行 reasoner 并验证 trace
    # (使用 test/test_multi_hop_reasoning.py 中的 _StandaloneMultiHopReasoner)
    result = reasoner.run("What drugs target BRCA1 through pathways?")

    # 验证
    assert len(result["trace"]) == 3
    assert result["trace"][0]["action"] == "B"
    assert result["trace"][0]["jump_target"]["identifier"] == "BRCA1"
    assert result["trace"][1]["action"] == "B"
    assert result["trace"][1]["jump_target"]["identifier"] == "DNA_Repair"
    assert result["trace"][2]["action"] == "C"
    assert len(result["evidence"]) == 3  # 1 + 2 items
```

---

## 模块文档 (Module Documentation)

### Module 1: TestDatasetLoader (`test_loader.py`)

加载并标准化来自各种文件格式的测试数据。

#### 支持的文件格式

**JSONL 格式**（推荐用于大数据集）：
```jsonl
{"question_id": "q1", "question": "What is...?", "output": "Expected answer", "rationale": "Explanation...", "multi_hop": true}
{"question_id": "q2", "instruction": "Identify...", "input": "data", "output": "Expected", "rationale": "Because..."}
```

**JSON 格式**：
```json
[
  {
    "question": "What drugs target BRCA1 through pathways?",
    "output": "Aspirin",
    "rationale": "BRCA1 → DNA Repair → Aspirin",
    "multi_hop": true
  },
  {
    "question": "What is 2+2?",
    "output": "4"
  }
]
```

也支持对象形式：
```json
{
  "questions": [
    {"question_id": "q1", "question": "...", "output": "...", "rationale": "..."}
  ]
}
```

**CSV 格式**：
```csv
question_id,question,expected,rationale
q1,"What is...?","Expected answer","Explanation..."
```

#### 标准化后的 Question 对象结构

```python
{
    "question_index": 1,          # 1-based 序号
    "question_id": "unique_id",   # 唯一 ID
    "question": "问题文本",
    "expected": "期望答案",
    "rationale": "推理说明",
    "multi_hop": True,            # 是否需要多跳推理
    "metadata": {...}             # 源文件中的其他字段
}
```

> **注意**：`multi_hop` 字段不会出现在 `metadata` 中，它是一级字段。

#### 使用示例

```python
from crossbar_llm.backend.evaluation import TestDatasetLoader

loader = TestDatasetLoader("questions.jsonl")

# 加载所有问题
all_questions = loader.get_questions()

# 按索引/ID 过滤
filtered = loader.filter_questions(indices=[1, 2, 3], question_ids=["q5"])

# 获取单个问题
q = loader.get_question_by_index(1)
q = loader.get_question_by_id("q5")

# 迭代
for q in loader:
    if q["multi_hop"]:
        print(f"[Multi-Hop] {q['question']}")
```

### Module 2: EvaluationRunner (`evaluation_runner.py`)

运行模型推理并收集结果。自动检测多跳推理输出。

#### 模型推理函数接口

推理函数接受一个 `question` (str) 参数，返回 dict：

```python
# 标准推理
def standard_model(question: str) -> dict:
    return {
        "answer": "答案文本",
        "query": "MATCH (n) RETURN n LIMIT 5",     # 可选
        "query_result": [...],                       # 可选
        "metadata": {"tokens": 123},                 # 可选
    }

# 多跳推理（返回 evidence + trace）
def multi_hop_model(question: str) -> dict:
    return {
        "answer": "多跳推理答案",
        "evidence": [{"drug": "Aspirin"}, {"protein": "BRCA1"}],
        "trace": [
            {"step": 1, "action": "B", "status": "jump", "reason": "..."},
            {"step": 2, "action": "C", "status": "terminate", "reason": "..."},
        ],
    }
```

当返回的 dict 中包含 `evidence` 或 `trace` 字段时（或问题被标记为 `multi_hop: true`），`EvaluationRunner` 会自动在结果中添加：
- `multi_hop: True`
- `evidence`: 证据列表
- `trace`: 推理路径
- `hop_count`: 推理步数

#### 使用示例

```python
from crossbar_llm.backend.evaluation import EvaluationRunner

runner = EvaluationRunner(
    model_inference_fn=my_model,
    model_name="gpt-4",
    output_dir="evaluation_results"
)

# 单题推理
result = runner.run_single(question_data)

# 批量推理 + 进度回调
results = runner.run_batch(
    questions,
    progress_callback=lambda cur, total: print(f"{cur}/{total}")
)

# 汇总统计
summary = runner.get_summary()
# => {"model_name": "gpt-4", "total_questions": 10, "success_rate": 0.9, ...}

# 保存结果
runner.save_results()           # 保存到 output_dir
runner.save_results("out.json") # 保存到指定路径
```

### Module 3: AnswerEvaluator (`answer_evaluator.py`)

使用 LLM-as-judge 评测模型答案。支持多跳推理 trace 感知评分。

#### 评分维度

| 维度                               | 范围       | 说明                                                |
|------------------------------------|-----------|-----------------------------------------------------|
| **Pass/Fail**                      | bool      | 模型输出是否在语义上匹配 benchmark 答案                |
| **Novelty Score**                  | 0–10      | 相比 benchmark，答案的原创性和额外信息量              |
| **Reasoning Similarity Score**     | 0–10      | 模型推理过程与 benchmark rationale 的相似度           |
| **Rationale Match**                | bool      | 推理过程是否匹配 benchmark rationale                  |

#### 多跳 Trace 感知评测

当传入 `trace` 参数时，evaluator 会在 judge prompt 中附加多跳推理路径摘要：

```
Multi-Hop Reasoning Trace:
  Step 1: action=B status=jump reason=explore BRCA1 pathways
  Step 2: action=C status=terminate reason=sufficient evidence
```

这使 judge 能同时评估答案正确性和推理路径的合理性。

#### 使用示例

```python
from crossbar_llm.backend.evaluation import AnswerEvaluator

evaluator = AnswerEvaluator(llm_judge_fn=my_judge_fn)

# 评测普通答案
result = evaluator.evaluate(
    question="What proteins does Caffeine target?",
    model_answer="Adenosine receptors A1 and A2A",
    expected="Adenosine receptors A1 and A2A",
    rationale="Caffeine is a competitive antagonist...",
)

# 评测多跳推理答案（传入 trace）
result = evaluator.evaluate(
    question="What drugs target BRCA1 through pathways?",
    model_answer="Aspirin targets BRCA1 via DNA repair",
    expected="Aspirin",
    rationale="BRCA1 → DNA Repair → Aspirin",
    trace=[
        {"step": 1, "action": "B", "status": "jump", "reason": "explore BRCA1"},
        {"step": 2, "action": "C", "status": "terminate", "reason": "done"},
    ],
)
print(f"Pass: {result['pass']}, Novelty: {result['novelty_score']}/10")

# 批量评测（支持 trace）
data = [
    {"question": "Q1", "model_answer": "A1", "expected": "E1", "rationale": "R1",
     "trace": [{"step": 1, "action": "C", "status": "terminate", "reason": "done"}]},
    {"question": "Q2", "model_answer": "A2", "expected": "E2", "rationale": "R2"},
]
results = evaluator.batch_evaluate(data)
```

#### LLM Judge 函数接口

Judge 函数接受一个 `prompt` (str)，返回包含以下字段的 JSON 字符串：

```json
{
  "pass": true,
  "reason": "正确性说明",
  "rationale_match": false,
  "novelty_score": 7,
  "reasoning_similarity_score": 8
}
```

evaluator 内置了健壮的 JSON 解析，能处理常见格式问题（markdown 代码块、Python bool、尾随逗号等）。

---

## 完整示例 (Complete Example)

### 标准评测 + 多跳推理混合评测

```python
from crossbar_llm.backend.evaluation import (
    TestDatasetLoader, EvaluationRunner, AnswerEvaluator, run_pipeline
)
import json

# --- 方式 A：使用 run_pipeline 一键评测 ---
def my_model(question: str) -> dict:
    # 对于多跳问题，返回 evidence + trace
    if "through" in question or "pathway" in question:
        return {
            "answer": "Drug X targets Protein Y via Pathway Z",
            "evidence": [{"drug": "X", "protein": "Y", "pathway": "Z"}],
            "trace": [
                {"step": 1, "action": "B", "status": "jump",
                 "reason": "lookup protein", "jump_target": {"node_type": "Protein", "identifier": "Y"}},
                {"step": 2, "action": "C", "status": "terminate", "reason": "done"},
            ],
        }
    return {"answer": "Simple answer"}

def my_judge(prompt: str) -> str:
    return json.dumps({
        "pass": True, "reason": "Correct",
        "rationale_match": True, "novelty_score": 6, "reasoning_similarity_score": 7,
    })

report = run_pipeline(
    dataset_path="questions.json",
    output_path="report.json",
    model_inference_fn=my_model,
    llm_judge_fn=my_judge,
    model_name="my-model",
    judge_model="mock-judge",
)

# --- 方式 B：分步控制 ---
loader = TestDatasetLoader("questions.json")
questions = loader.get_questions()

runner = EvaluationRunner(model_inference_fn=my_model, model_name="my-model")
results = runner.run_batch(questions)

evaluator = AnswerEvaluator(llm_judge_fn=my_judge)
for r in results:
    r["judge"] = evaluator.evaluate(
        question=r["question"],
        model_answer=r.get("model_answer", ""),
        expected=r.get("expected", ""),
        rationale=r.get("rationale", ""),
        trace=r.get("trace"),
    )

# 分别统计普通问题和多跳问题
multi_hop_results = [r for r in results if r.get("multi_hop")]
standard_results = [r for r in results if not r.get("multi_hop")]
print(f"Multi-hop: {len(multi_hop_results)}, Standard: {len(standard_results)}")
```

## 配置 (Configuration)

### Judge 模型配置

AnswerEvaluator 支持任意 LLM 提供商，只需实现 `(prompt: str) -> str` 接口：

**OpenAI:**
```python
from openai import OpenAI
client = OpenAI(api_key="sk-...")

def llm_judge(prompt: str) -> str:
    response = client.chat.completions.create(
        model="gpt-4", messages=[{"role": "user", "content": prompt}],
        temperature=0, max_tokens=256,
    )
    return response.choices[0].message.content
```

**Google (Gemini):**
```python
import google.generativeai as genai
genai.configure(api_key="...")
model = genai.GenerativeModel("gemini-pro")

def llm_judge(prompt: str) -> str:
    return model.generate_content(prompt).text
```

**OpenRouter:**
```python
from openai import OpenAI
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key="sk-or-...")

def llm_judge(prompt: str) -> str:
    response = client.chat.completions.create(
        model="anthropic/claude-3-opus", messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    return response.choices[0].message.content
```

### 批量 Pipeline 配置 (`config/batch_config.yaml`)

```yaml
# 模型配置
provider: "OpenRouter"
models:
  - "gemini-3-flash-preview"

# 测试数据源
questions:
  benchmark:
    enabled: true
    file: "Benchmark/v0119filtered.small.test.jsonl"

# 多跳推理（启用后覆盖 multi_step）
multi_hop:
  enabled: true
  max_steps: 5

# LLM-as-judge 评测
judge:
  enabled: true
  model: "gpt-oss-120b"
  temperature: 0
  max_tokens: 2560
```

## 错误处理 (Error Handling)

所有模块包含完善的错误处理：

```python
from crossbar_llm.backend.evaluation import TestDatasetLoader, EvaluationRunner

# 文件加载错误
try:
    loader = TestDatasetLoader("nonexistent.jsonl")
    loader.load()
except FileNotFoundError as e:
    print(f"文件未找到: {e}")
except ValueError as e:
    print(f"格式错误: {e}")

# 推理错误被捕获到结果中
def flaky_model(question):
    if "error" in question.lower():
        raise ValueError("Simulated error")
    return {"answer": "OK"}

runner = EvaluationRunner(flaky_model, "test-model")
results = runner.run_batch(questions)

for r in results:
    if not r["success"]:
        print(f"Question {r['question_index']} failed: {r['error']}")
```

## 最佳实践 (Best Practices)

1. **大数据集使用 JSONL 格式** — 更好的内存效率
2. **验证数据格式** — 使用 loader 的标准化确保一致性
3. **监控进度** — 对长时间评测使用 progress callback
4. **增量保存** — 每批次后保存结果以防数据丢失
5. **选择合适的 judge 模型** — 更强的模型（GPT-4、Claude Opus）给出更好的评测
6. **缓存结果** — 保存评测结果以避免重复调用 LLM
7. **多跳问题标记 `multi_hop: true`** — 让 pipeline 和 judge 感知推理路径
8. **检查 trace 合理性** — 验证推理路径中的 action 序列是否符合预期

## 故障排除 (Troubleshooting)

| 问题                           | 解决方案                                                        |
|-------------------------------|----------------------------------------------------------------|
| "File not found" 错误          | 检查文件路径是否正确，使用绝对路径                                |
| "Unsupported file format"     | 检查文件扩展名（`.jsonl`/`.json`/`.csv`）                        |
| "Judge output parse error"    | 确保 judge 模型返回有效 JSON，`max_tokens >= 256`               |
| 所有新颖性评分都很低            | 可能是模型在重复 benchmark 答案，调整 temperature                 |
| 多跳推理结果缺少 trace          | 确保推理函数返回 `evidence` 和 `trace` 字段                     |
| `multi_hop` 字段未被识别       | 字段名必须使用下划线：`"multi_hop": true`（`multi-hop` 连字符无效）|
| 多跳推理步数过多               | 减小 `max_steps` 或优化决策 prompt                               |

## License

本评测 pipeline 是 CROssBAR-LLM 项目的一部分，采用 GPL-3.0 许可证。
