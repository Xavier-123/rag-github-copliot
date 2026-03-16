# 企业级 Agentic RAG + Self-Improving RAG 系统

> 面向研究实验的完整 RAG 对话系统，支持快速对比不同检索/评估策略。

---

## 系统架构

```
User Query
    │
    ▼
┌─────────────────────────────────────────────────────┐
│               Agentic RAG Layer                      │
│                                                      │
│  QueryUnderstandingAgent (rewrite/intent/decompose)  │
│       └─▶ TaskPlanningAgent                          │
│               └─▶ RetrievalPlanningAgent             │
│                       └─▶ MultiRetrieverAgent ◀──┐  │
│                               └─▶ RerankAgent    │  │
│                                     └─▶ ContextEngineeringAgent │
│                                           └─▶ ReasoningAgent    │
│                                                 └─▶ ReflectionAgent ──┘ (retry) │
│                                                       └─▶ Answer Generation     │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│               Memory Layer                           │
│  ConversationMemory │ KnowledgeMemory │ EpisodicMemory│
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│           Self-Improving Layer                       │
│  Evaluator (RAGAS / LLM-as-Judge)                    │
│       └─▶ FeedbackLearning                           │
│               └─▶ SystemOptimizer                    │
└─────────────────────────────────────────────────────┘
```

## 核心组件

### Agentic RAG Layer（4 大核心 Agent）

| Agent | 职责 | 关键算法 |
|-------|------|---------|
| `QueryUnderstandingAgent` | 查询改写、意图检测、复杂度估计、查询分解 | LLM few-shot prompting |
| `TaskPlanningAgent` | 生成多步任务执行计划 | LLM-based DAG planning |
| `RetrievalPlanningAgent` | 选择检索策略（Vector/Graph/Web/Hybrid） | Rule-based + LLM strategy selection |
| `MultiRetrieverAgent` | 并行执行多个检索器 | `ThreadPoolExecutor` 并行检索 |
| `RerankAgent` | 精排文档，LLM 重新评分 | LLM cross-encoder scoring |
| `ContextEngineeringAgent` | 去重、压缩、过滤、排序上下文 | MMR + Deduplication + Compression |
| `ReasoningAgent` | Chain-of-Thought / 多跳推理 | CoT prompting，最多 N 次 retrieve→reason 迭代 |
| `ReflectionAgent` | 答案验证、幻觉检测、置信度打分 | LLM self-critique |

### Memory Layer

| 模块 | 存储 | 说明 |
|------|------|------|
| `ConversationMemory` | In-memory / Redis | 滑动窗口对话历史 |
| `KnowledgeMemory` | ChromaDB / FAISS / Milvus / Weaviate | 文档分块 + 向量索引 |
| `EpisodicMemory` | In-memory / Redis | 系统经验（策略性能、评估分数） |

### Retrieval Layer

| 检索器 | 算法 | 适用场景 |
|--------|------|---------|
| `VectorRetriever` | 向量近似最近邻（ANN） | 语义相似检索 |
| `GraphRetriever` | Neo4j Cypher 图遍历 | 实体关系查询 |
| `WebRetriever` | Serper / Tavily / Bing API | 实时信息查询 |
| `HybridRetriever` | Dense + BM25 → RRF 融合 | 综合覆盖 |

### Self-Improving Layer

| 模块 | 算法 | 参考 |
|------|------|------|
| `Evaluator` | RAGAS / LLM-as-Judge / DSPy Evaluate | Faithfulness, Relevance, Precision, Recall |
| `FeedbackLearning` | 分析历史 Episode，更新策略偏好 | 自适应检索选择 |
| `SystemOptimizer` | 动态调整 `rerank_top_k`、策略映射、Prompt | DSPy / AutoRAG 接口兼容 |

---

## 项目结构

```
.
├── agents/                        # Agentic RAG 所有 Agent 模块
│   ├── base_agent.py              # AgentInput/AgentOutput/BaseAgent 抽象
│   ├── query_understanding_agent.py
│   ├── task_planning_agent.py
│   ├── retrieval_planning_agent.py
│   ├── multi_retriever_agent.py
│   ├── rerank_agent.py
│   ├── context_engineering_agent.py
│   ├── reasoning_agent.py
│   └── reflection_agent.py
├── retrieval/                     # 检索后端
│   ├── base_retriever.py
│   ├── vector_retriever.py        # ChromaDB / FAISS / Milvus / Weaviate
│   ├── graph_retriever.py         # Neo4j GraphRAG
│   ├── web_retriever.py           # Serper / Tavily / Bing
│   └── hybrid_retriever.py        # Dense + BM25 + RRF
├── memory/                        # 记忆层
│   ├── conversation_memory.py
│   ├── knowledge_memory.py
│   └── episodic_memory.py
├── evaluation/                    # 自动评估
│   ├── evaluator.py
│   ├── llm_judge.py
│   └── ragas_evaluator.py
├── optimization/                  # 自我改进
│   ├── feedback_learning.py
│   └── system_optimizer.py
├── pipeline/
│   └── rag_pipeline.py            # 端到端流水线编排
├── config/
│   └── settings.py                # 统一配置（pydantic-settings）
├── utils/
│   └── helpers.py                 # 公共工具函数
├── tests/                         # 单元 + 集成测试
├── main.py                        # CLI 入口
├── requirements.txt
└── .env.example
```

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 OpenAI / Azure OpenAI API Key
```

### 3. 启动交互式对话

```bash
python main.py chat
```

### 4. 导入文档（JSONL 格式）

每行一个 JSON 对象，需包含 `"content"` 字段：

```bash
python main.py ingest --file my_documents.jsonl
```

### 5. Python API

```python
from pipeline.rag_pipeline import RAGPipeline

pipeline = RAGPipeline()

# 添加文档
pipeline.add_documents([
    {"content": "RAG is Retrieval Augmented Generation.", "source": "intro.txt"}
])

# 对话
response = pipeline.chat("What is RAG?", session_id="user-001")
print(response.answer)
print(f"Confidence: {response.confidence:.2f}")
print(f"Sources: {response.sources}")
```

---

## 配置说明

所有配置通过 `.env` 文件或环境变量（`RAG_` 前缀）设置：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RAG_OPENAI_API_KEY` | – | OpenAI API Key |
| `RAG_OPENAI_MODEL` | `gpt-4o-mini` | 对话模型 |
| `RAG_VECTOR_STORE_TYPE` | `chroma` | 向量库类型 |
| `RAG_DEFAULT_RETRIEVAL_MODE` | `hybrid` | 默认检索策略 |
| `RAG_EVALUATION_FRAMEWORK` | `llm_judge` | 评估框架 |
| `RAG_EVALUATION_ENABLED` | `true` | 是否启用评估 |
| `RAG_FEEDBACK_LEARNING_ENABLED` | `true` | 是否启用自我改进 |
| `RAG_CONFIDENCE_THRESHOLD` | `0.7` | Reflection Agent 最低置信阈值 |

完整配置列表见 `.env.example`。

---

## 运行测试

```bash
pytest tests/ -v
```

---

## 扩展说明

### 添加新的检索器

继承 `retrieval.base_retriever.BaseRetriever`，实现 `retrieve()` 和 `add_documents()` 方法，然后在 `MultiRetrieverAgent` 的 `retriever_map` 中注册。

### 添加新的评估框架

继承或实现与 `LLMJudge` 相同的 `.evaluate(query, answer, context, reference)` 接口，返回 `EvaluationResult` 对象。

### 启用 RAGAS 评估

```bash
pip install ragas datasets
# 在 .env 中设置：
RAG_EVALUATION_FRAMEWORK=ragas
```

### 启用 DSPy 优化（未来扩展）

在 `SystemOptimizer.apply()` 中集成 DSPy `teleprompter.compile()` 调用。

---

## 技术参考

- **RRF**: Cormack et al., 2009 – *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*
- **RAGAS**: Es et al., 2023 – *RAGAS: Automated Evaluation of Retrieval Augmented Generation*
- **LLM-as-Judge**: Zheng et al., 2023 – *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*
- **Self-RAG**: Asai et al., 2023 – *Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection*
- **DSPy**: Khattab et al., 2023 – *DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines*

