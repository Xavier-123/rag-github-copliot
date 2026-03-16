"""
Rerank Agent（重排序 Agent）
============================
对多路检索结果进行精排，提升最终传入 LLM 的文档质量。
支持两种重排方法：
  - Cross Encoder  : 使用交叉注意力模型对 (query, doc) 对进行精确打分
  - ColBERT        : 通过 RAGatouille 库实现的细粒度 token 级别重排

工作流位置：MultiRetrieverAgents → [RerankAgent] → ContextEngineeringAgent

调用关系：
  - 输入 state 字段  : retrieved_docs, rewritten_query
  - 输出 state 字段  : reranked_docs（按相关性降序排列，截取 top_k）

技术说明：
  - Cross Encoder 使用 sentence-transformers 库，通过 BERT 双编码器实现精确的
    query-document 相关性打分（计算量较大，适合 < 50 篇文档）
  - ColBERT（via RAGatouille）使用后期交互机制，兼顾效率和精度
"""

from __future__ import annotations

from typing import List, Optional

from rag_system.agents.base import AgentState, BaseAgent, RetrievedDocument
from rag_system.config.settings import get_settings


class RerankAgent(BaseAgent):
    """
    重排序 Agent
    ============
    加载重排模型并对检索文档进行精排，截取 top_k 高质量文档传递给下游。

    模型懒加载策略：
    - 重排模型首次 run() 时才加载，避免系统启动时的内存开销
    - 使用 _model_loaded 标志位防止重复加载

    方法选择：
    - method="cross_encoder" : 使用 CrossEncoder 精确打分（默认，精度高）
    - method="colbert"       : 使用 RAGatouille ColBERT（速度快，适合大规模检索）
    """

    def __init__(self, settings=None) -> None:
        super().__init__("RerankAgent", settings)
        self._cfg = get_settings()
        self._method = self._cfg.rerank.method
        self._top_k = self._cfg.rerank.top_k
        self._cross_encoder = None
        self._colbert_model = None
        self._model_loaded = False

    def _load_model(self) -> None:
        """懒加载重排模型"""
        if self._model_loaded:
            return
        try:
            if self._method == "cross_encoder":
                from sentence_transformers import CrossEncoder
                self._cross_encoder = CrossEncoder(self._cfg.rerank.cross_encoder_model)
                self._logger.info(f"Cross Encoder 加载成功: {self._cfg.rerank.cross_encoder_model}")
            elif self._method == "colbert":
                from ragatouille import RAGPretrainedModel
                self._colbert_model = RAGPretrainedModel.from_pretrained(
                    self._cfg.rerank.colbert_model
                )
                self._logger.info(f"ColBERT 模型加载成功: {self._cfg.rerank.colbert_model}")
        except ImportError as e:
            self._logger.warning(f"重排模型库未安装，降级为分数排序: {e}")
        except Exception as e:
            self._logger.warning(f"重排模型加载失败，降级为分数排序: {e}")
        finally:
            self._model_loaded = True

    def _execute(self, state: AgentState) -> AgentState:
        docs: List[RetrievedDocument] = state.get("retrieved_docs", [])
        query: str = state.get("rewritten_query", state.get("user_query", ""))

        if not docs:
            self._logger.warning("无检索文档，跳过重排")
            return {**state, "reranked_docs": []}

        # 懒加载模型
        self._load_model()

        # 执行重排
        if self._method == "cross_encoder" and self._cross_encoder:
            reranked = self._rerank_cross_encoder(query, docs)
        elif self._method == "colbert" and self._colbert_model:
            reranked = self._rerank_colbert(query, docs)
        else:
            # 降级：按原始检索分数排序
            reranked = sorted(docs, key=lambda d: d.score, reverse=True)

        # 截取 top_k
        top_docs = reranked[: self._top_k]
        self._logger.info(
            f"重排完成: {len(docs)} → {len(top_docs)} 篇文档（方法: {self._method}）"
        )

        return {**state, "reranked_docs": top_docs}

    def _rerank_cross_encoder(
        self, query: str, docs: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        Cross Encoder 重排
        ===================
        构建 (query, doc_content) 对，批量送入 Cross Encoder 打分。
        Cross Encoder 使用全注意力机制，比双编码器更精确但计算量更大。
        """
        pairs = [[query, doc.content] for doc in docs]
        scores = self._cross_encoder.predict(pairs)  # type: ignore

        # 更新文档分数并排序
        for doc, score in zip(docs, scores):
            doc.score = float(score)

        return sorted(docs, key=lambda d: d.score, reverse=True)

    def _rerank_colbert(
        self, query: str, docs: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        ColBERT 重排（via RAGatouille）
        ================================
        ColBERT 后期交互机制：分别编码 query 和 document，
        通过 MaxSim 算子计算细粒度 token 级相关性。
        相比 Cross Encoder 速度更快，适合大规模候选集。
        """
        doc_texts = [doc.content for doc in docs]
        # RAGatouille rerank 返回 {content, score, rank} 列表
        results = self._colbert_model.rerank(  # type: ignore
            query=query,
            documents=doc_texts,
            k=len(docs),
        )

        # 重建有序文档列表
        content_to_doc = {doc.content: doc for doc in docs}
        reranked = []
        for res in results:
            doc = content_to_doc.get(res["content"])
            if doc:
                doc.score = float(res.get("score", 0.0))
                reranked.append(doc)

        return reranked
