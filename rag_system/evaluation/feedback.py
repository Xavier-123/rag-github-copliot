"""
Feedback Optimizer（反馈优化模块）
====================================
使用 DSPy 框架对 RAG 系统提示词和参数进行自动化优化。

DSPy 优化原理：
  - 将 Prompt 工程转化为优化问题
  - 使用 BootstrapFewShot / MIPRO / BayesianSignatureOptimizer 等算法
    自动搜索最优的 Few-shot 示例和提示词
  - 通过评估指标（如 RAGAS 分数）作为优化目标函数

工作流位置：RAGEvaluator → [FeedbackOptimizer] → (更新 Prompt / 记忆)

调用关系：
  - 输入 state 字段  : evaluation_scores, user_query, answer, context
  - 输出 state 字段  : feedback_applied（是否应用了优化）

使用场景：
  - 在线优化 : 每次对话后基于 RAGAS 分数实时调整
  - 离线优化 : 积累一批低质量样本后批量运行 DSPy 优化器
"""

from __future__ import annotations

from typing import Dict, Optional

from loguru import logger

from rag_system.agents.base import AgentState
from rag_system.config.settings import get_settings


class FeedbackOptimizer:
    """
    DSPy 反馈优化器
    ================
    基于评估分数分析系统瓶颈并触发相应的优化策略。

    优化策略：
    1. 低忠实性（faithfulness < 阈值）→ 强化上下文工程，增加压缩步骤
    2. 低相关性（answer_relevancy < 阈值）→ 优化查询改写 Prompt
    3. 低精度（context_precision < 阈值）→ 调整检索 top_k，增加重排严格度
    4. 整体低分 → 触发 DSPy BootstrapFewShot 全链路优化

    [预留接口]：
    - _run_dspy_optimization() 可接入完整 DSPy 优化流程
    - 生产环境中应将优化后的 Prompt 持久化保存并热加载
    """

    def __init__(self) -> None:
        self._cfg = get_settings()
        self._logger = logger.bind(module="FeedbackOptimizer")
        self._optimization_history: list = []
        self._feedback_count = 0

    def run(self, state: AgentState) -> AgentState:
        """
        反馈优化入口（LangGraph 节点函数）。
        分析评估分数并决定是否触发优化。
        """
        if not self._cfg.evaluation.enabled:
            return {**state, "feedback_applied": False}

        scores: Dict[str, float] = state.get("evaluation_scores", {})
        if not scores:
            return {**state, "feedback_applied": False}

        # 分析分数并生成优化建议
        feedback_applied = self._analyze_and_optimize(scores, state)

        return {**state, "feedback_applied": feedback_applied}

    def _analyze_and_optimize(
        self, scores: Dict[str, float], state: AgentState
    ) -> bool:
        """
        分析评估分数，识别优化机会并触发优化。

        Returns:
            是否应用了优化操作
        """
        threshold = self._cfg.evaluation.feedback_threshold
        applied = False

        # 检查各指标是否低于阈值
        low_metrics = {k: v for k, v in scores.items() if v < threshold}

        if not low_metrics:
            self._logger.info(f"所有评估指标达标（threshold={threshold}），无需优化")
            return False

        self._logger.warning(f"发现低分指标（< {threshold}）: {low_metrics}")

        # 分指标优化策略
        for metric, score in low_metrics.items():
            if metric == "faithfulness":
                self._optimize_for_faithfulness(score, state)
                applied = True
            elif metric == "answer_relevancy":
                self._optimize_for_relevancy(score, state)
                applied = True
            elif metric == "context_precision":
                self._optimize_for_precision(score, state)
                applied = True

        # 记录优化历史
        self._optimization_history.append(
            {
                "scores": scores,
                "low_metrics": low_metrics,
                "query": state.get("user_query", "")[:100],
            }
        )
        self._feedback_count += 1

        # 积累足够样本时触发 DSPy 批量优化
        if self._feedback_count % 50 == 0:
            self._run_dspy_optimization()

        return applied

    def _optimize_for_faithfulness(self, score: float, state: AgentState) -> None:
        """
        忠实性优化策略。
        低忠实性通常意味着 LLM 产生了幻觉或脱离了上下文。
        优化方向：加强上下文工程的过滤和压缩。
        """
        self._logger.info(f"忠实性优化（当前分数: {score:.3f}）: 建议启用上下文压缩")

    def _optimize_for_relevancy(self, score: float, state: AgentState) -> None:
        """
        相关性优化策略。
        低相关性通常意味着查询改写效果差或检索精度不足。
        优化方向：优化 QueryUnderstandingAgent 的 Prompt。
        """
        self._logger.info(f"相关性优化（当前分数: {score:.3f}）: 建议优化查询改写 Prompt")

    def _optimize_for_precision(self, score: float, state: AgentState) -> None:
        """
        精度优化策略。
        低精度通常意味着检索了过多不相关文档。
        优化方向：降低 top_k，提高重排过滤阈值。
        """
        self._logger.info(f"精度优化（当前分数: {score:.3f}）: 建议降低 top_k 或提高过滤阈值")

    def _run_dspy_optimization(self) -> None:
        """
        触发 DSPy 批量优化流程。

        [预留接口] 完整 DSPy 优化示例：
        ```python
        import dspy
        from dspy.teleprompt import BootstrapFewShot

        # 配置 DSPy LLM
        lm = dspy.OpenAI(model=self._cfg.llm.model_name, api_key=self._cfg.llm.api_key)
        dspy.settings.configure(lm=lm)

        # 定义 RAG 签名
        class RAGSignature(dspy.Signature):
            context = dspy.InputField(desc="检索到的相关文档")
            question = dspy.InputField(desc="用户问题")
            answer = dspy.OutputField(desc="基于文档的准确回答")

        # 配置优化器
        teleprompter = BootstrapFewShot(metric=ragas_metric)
        optimized_program = teleprompter.compile(
            RAGModule(),
            trainset=self._build_training_set(),
        )
        # 保存优化后的程序
        optimized_program.save("./data/optimized_rag.json")
        ```
        """
        self._logger.info(
            f"[预留] 触发 DSPy 批量优化（已积累 {len(self._optimization_history)} 条样本）"
        )

    def get_optimization_stats(self) -> dict:
        """获取优化统计信息"""
        if not self._optimization_history:
            return {"total_feedback": 0, "avg_scores": {}}

        all_scores = [h["scores"] for h in self._optimization_history]
        all_metrics = set(k for s in all_scores for k in s.keys())
        avg_scores = {
            metric: sum(s.get(metric, 0) for s in all_scores) / len(all_scores)
            for metric in all_metrics
        }
        return {
            "total_feedback": self._feedback_count,
            "avg_scores": {k: round(v, 3) for k, v in avg_scores.items()},
        }
