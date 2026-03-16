# 企业级 RAG 对话系统

基于 **LangGraph + LlamaIndex + GraphRAG + RAGatouille + DSPy + RAGAS** 构建的完整企业级 RAG 对话系统。

## 系统架构

```
User Query
    │
    ▼
Memory Retrieval         ← 从历史记忆检索相关上下文
    │
    ▼
Query Understanding Agent   ← 查询改写 / 分类 / 分解（LLM）
    │
    ▼
Task Planning Agent         ← 工具选择 / RAG 策略（LLM）
    │
    ▼
Retrieval Strategy Agent    ← 确定检索来源（规则）
    │
    ▼
Adaptive Retrieval Controller  ← 动态调整检索策略（复杂度评估）
    │
    ▼
Multi-Retriever Agents      ← 并行多路检索（向量 / 图谱 / Web）
    │
    ▼
Rerank Agent                ← Cross Encoder / ColBERT 重排序
    │
    ▼
Context Engineering Agent   ← 上下文过滤 / 压缩 / 拼接
    │
    ▼
Reasoning Agent             ← Chain-of-Thought 推理 / 答案生成
    │
    ▼
Reflection Agent            ← 答案验证 / 幻觉检测 / 自我反思
    │
    ├── passed → Memory Update → Evaluation (RAGAS) → Feedback (DSPy) → END
    └── failed → Reasoning Agent（最多重试 N 次）
```

## 技术选型

| 层次 | 技术 | 用途 |
|------|------|------|
| 工作流编排 | **LangGraph** | Agent 有向图状态机 |
| 数据与索引 | **LlamaIndex** | 文档加载、切分、向量索引 |
| 图谱检索 | **GraphRAG** | 知识图谱构建与多跳检索 |
| 精排重排 | **RAGatouille** | ColBERT 语义重排序 |
| 评估优化 | **RAGAS** | 多维度 RAG 质量评估 |
| 提示优化 | **DSPy** | 自动 Prompt 工程优化 |
| 向量数据库 | **ChromaDB** | 本地持久化向量存储 |

## 项目结构

```
rag_system/
├── config/
│   └── settings.py              # 全局配置（Pydantic BaseSettings）
├── agents/
│   ├── base.py                  # Agent 基类 & AgentState TypedDict
│   ├── query_understanding_agent.py   # 查询理解
│   ├── task_planning_agent.py         # 任务规划
│   ├── retrieval_strategy_agent.py    # 检索策略
│   ├── adaptive_retrieval_controller.py  # 自适应检索控制器
│   ├── multi_retriever_agents.py      # 多路并行检索
│   ├── rerank_agent.py                # 重排序
│   ├── context_engineering_agent.py   # 上下文工程
│   ├── reasoning_agent.py             # 推理生成
│   └── reflection_agent.py            # 反思验证
├── memory/
│   ├── memory_retrieval.py      # 记忆检索
│   └── memory_update.py         # 记忆更新
├── retrieval/
│   ├── vector_retriever.py      # LlamaIndex 向量检索
│   ├── graph_retriever.py       # GraphRAG 图谱检索
│   └── web_retriever.py         # DuckDuckGo Web 检索
├── data/
│   └── index_manager.py         # LlamaIndex 索引管理
├── evaluation/
│   ├── evaluator.py             # RAGAS 评估器
│   └── feedback.py              # DSPy 反馈优化器
└── workflow/
    └── graph.py                 # LangGraph 工作流定义
main.py                          # 命令行入口
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 OpenAI API Key
```

### 3. 运行系统

```bash
# 交互式对话
python main.py

# 单次查询
python main.py --query "什么是 RAG？"

# 加载文档目录后对话
python main.py --index-dir ./docs
```

### 4. 代码集成

```python
from rag_system import RAGWorkflow

workflow = RAGWorkflow()
result = workflow.run("什么是检索增强生成？", session_id="user_001")
print(result["answer"])
print(result["evaluation_scores"])
```

## 加载文档到知识库

```python
from rag_system.data.index_manager import IndexManager

manager = IndexManager()

# 从目录批量加载（支持 PDF / TXT / MD / DOCX）
manager.load_documents_from_directory("./my_docs")

# 添加单个文件
manager.load_documents_from_file("./manual.pdf")

# 直接添加文本
manager.add_text("需要索引的文本内容", metadata={"source": "manual"})
```

## 运行测试

```bash
pip install pytest
pytest tests/ -v
```

## 模块设计原则

1. **解耦架构**：每个 Agent 实现 `BaseAgent` 接口，通过 `AgentState` 共享状态，无直接依赖
2. **优雅降级**：每个模块均有 Mock/Fallback 实现，外部服务不可用时系统仍可运行
3. **可观测性**：使用 `loguru` 记录每个 Agent 的执行耗时、输入输出摘要
4. **可扩展性**：复杂模块（GraphRAG、DSPy 优化）预留标准接口，支持按需接入
5. **类型安全**：全面使用 Python 类型注解和 Pydantic 模型

## 配置说明

主要配置项（可通过 `.env` 文件或环境变量覆盖）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `LLM_MODEL_NAME` | `gpt-4o-mini` | LLM 模型名称 |
| `LLM_API_KEY` | - | OpenAI API Key |
| `VECTOR_STORE_TOP_K` | `10` | 向量检索返回数量 |
| `RERANK_METHOD` | `cross_encoder` | 重排方法 |
| `GRAPH_RAG_ENABLED` | `false` | 是否启用 GraphRAG |
| `WEB_SEARCH_ENABLED` | `true` | 是否启用 Web 搜索 |
| `EVAL_ENABLED` | `true` | 是否启用 RAGAS 评估 |
| `MAX_RETRIES` | `3` | 反思重试最大次数 |
