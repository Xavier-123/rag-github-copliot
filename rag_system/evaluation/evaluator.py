"""
RAG Evaluator（评估模块）
===========================
使用 RAGAS 框架对 RAG 系统输出进行自动化评估。

RAGAS 评估指标：
  - Faithfulness（忠实性）       : 答案是否完全基于检索上下文，避免幻觉
  - Answer Relevancy（答案相关性）: 答案是否与问题高度相关
  - Context Precision（上下文精度）: 检索到的文档是否与问题高度相关
  - Context Recall（上下文召回）   : 是否检索到了回答问题所需的全部信息

工作流位置：MemoryUpdate → [Evaluator] → FeedbackOptimizer → (结束)

调用关系：
  - 输入 state 字段  : user_query, answer, context, retrieved_docs
  - 输出 state 字段  : evaluation_scores（Dict[str, float]）

RAGAS 工作原理：
  - 使用 LLM 作为评判器，无需人工标注的 ground truth（无参考评估）
  - Faithfulness  : LLM 判断答案的每个陈述是否有上下文支撑
  - Answer Relevancy : LLM 基于答案反向生成问题，计算与原问题的相似度
"""

from __future__ import annotations

from typing import Dict, List, Optional

from loguru import logger

from rag_system.agents.base import AgentState
from rag_system.config.settings import get_settings


class RAGEvaluator:
    """
    RAGAS 评估器
    ============
    自动评估 RAG 系统输出质量，提供多维度量化评估分数。

    评估数据构造：
    - RAGAS 需要 Dataset 格式：{"question": [...], "answer": [...], "contexts": [...]}
    - contexts 是字符串列表（每个检索文档为一个元素）
    - 评估结果以字典形式返回：{"faithfulness": 0.85, "answer_relevancy": 0.90, ...}

    注意：RAGAS 评估会调用 LLM API（有一定延迟和费用），
          可通过配置 evaluation.enabled=False 跳过评估。
    """

    def __init__(self) -> None:
        self._cfg = get_settings()
        self._logger = logger.bind(module="RAGEvaluator")

    def run(self, state: AgentState) -> AgentState:
        """
        评估入口（LangGraph 节点函数）。
        执行 RAGAS 评估并将分数写入 state["evaluation_scores"]。
        """
        if not self._cfg.evaluation.enabled:
            return {**state, "evaluation_scores": {}}

        question = state.get("user_query", "")
        answer = state.get("answer", "")
        context = state.get("context", "")

        if not all([question, answer]):
            self._logger.warning("缺少必要的评估输入，跳过评估")
            return {**state, "evaluation_scores": {}}

        scores = self._evaluate(question, answer, context)
        self._logger.info(f"RAGAS 评估完成: {scores}")

        return {**state, "evaluation_scores": scores}

    def _evaluate(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> Dict[str, float]:
        """
        执行 RAGAS 评估。
        若 RAGAS 库不可用或评估失败，返回默认分数。
        """
        try:
            return self._run_ragas(question, answer, context)
        except ImportError:
            self._logger.warning("RAGAS 库未安装，使用启发式评估")
            return self._heuristic_evaluate(question, answer, context)
        except Exception as e:
            self._logger.error(f"RAGAS 评估失败: {e}")
            return self._heuristic_evaluate(question, answer, context)

    def _run_ragas(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> Dict[str, float]:
        """
        使用 RAGAS 进行标准化评估。

        RAGAS Dataset 格式：
        - question  : 用户问题列表
        - answer    : 系统回答列表
        - contexts  : 检索上下文列表（每个元素是一个字符串列表）
        """
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision

        # 将上下文字符串分割为文档列表（按分隔符 "---" 分割）
        context_docs = [c.strip() for c in context.split("---") if c.strip()]
        if not context_docs:
            context_docs = [context]

        eval_dataset = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [context_docs],
            }
        )

        # 选择评估指标
        metrics = []
        metric_names = self._cfg.evaluation.ragas_metrics
        if "faithfulness" in metric_names:
            metrics.append(faithfulness)
        if "answer_relevancy" in metric_names:
            metrics.append(answer_relevancy)
        if "context_precision" in metric_names:
            metrics.append(context_precision)

        result = evaluate(eval_dataset, metrics=metrics)
        return {k: float(v) for k, v in result.items()}

    def _heuristic_evaluate(
        self,
        question: str,
        answer: str,
        context: str,
    ) -> Dict[str, float]:
        """
        启发式评估（RAGAS 不可用时的降级方案）。
        基于简单规则估算各项指标：
        - 答案长度合理性
        - 关键词重叠率
        - 上下文覆盖率
        """
        # 答案相关性：问题关键词在答案中的覆盖率
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        relevancy = len(question_words & answer_words) / max(len(question_words), 1)

        # 忠实性：答案内容在上下文中的覆盖率（简化的 bigram 重叠率）
        answer_tokens = answer.split()
        context_tokens = context.split()
        answer_ngrams = set(zip(answer_tokens, answer_tokens[1:]))
        context_ngrams = set(zip(context_tokens, context_tokens[1:]))
        faithfulness = (
            len(answer_ngrams & context_ngrams) / max(len(answer_ngrams), 1)
            if answer_ngrams
            else 0.5
        )

        return {
            "faithfulness": min(faithfulness * 2, 1.0),  # 归一化
            "answer_relevancy": min(relevancy * 3, 1.0),
            "context_precision": 0.7,  # 默认值
        }
