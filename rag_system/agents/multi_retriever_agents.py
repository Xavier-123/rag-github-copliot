"""
Multi-Retriever Agents（多检索器并行执行 Agent）
================================================
并行执行多路信息检索，包括：
  - VectorRetrieverAgent : 向量数据库语义检索（LlamaIndex）
  - GraphRetrieverAgent  : 知识图谱检索（GraphRAG / Mock）
  - WebRetrieverAgent    : 实时 Web 搜索（DuckDuckGo）

工作流位置：AdaptiveRetrievalController → [MultiRetrieverAgents] → RerankAgent

调用关系：
  - 输入 state 字段  : retrieval_sources, rewritten_query, sub_queries
  - 输出 state 字段  : retrieved_docs（Annotated List，支持并行 append）

并行执行策略：
  - LangGraph 支持图节点并行执行（fanout/fanin）
  - MultiRetrieverAgents 作为编排器，同时调用三个子检索器
  - 子检索器结果通过 state["retrieved_docs"] 的 Annotated[List, operator.add] 合并
"""

from __future__ import annotations

import concurrent.futures
from typing import List, Optional

from rag_system.agents.base import AgentState, BaseAgent, RetrievedDocument
from rag_system.config.settings import get_settings
from rag_system.retrieval.vector_retriever import VectorRetriever
from rag_system.retrieval.graph_retriever import GraphRetriever
from rag_system.retrieval.web_retriever import WebRetriever


class MultiRetrieverAgents(BaseAgent):
    """
    多检索器并行执行 Agent
    ======================
    统一编排向量、图谱、Web 三路检索器，并行获取多源信息。

    并行执行机制：
    - 使用 ThreadPoolExecutor 在单节点内并行调用多个检索器
    - 每个检索器失败不影响其他检索器，错误被捕获并记录
    - 所有检索结果合并为统一的 RetrievedDocument 列表

    扩展说明：
    - 在 LangGraph 中，也可将三个检索器注册为独立的并行节点
      实现真正的图级并行（推荐用于生产环境）
    """

    def __init__(self, settings=None) -> None:
        super().__init__("MultiRetrieverAgents", settings)
        cfg = get_settings()
        self._vector_retriever = VectorRetriever(settings=cfg)
        self._graph_retriever = GraphRetriever(settings=cfg)
        self._web_retriever = WebRetriever(settings=cfg)

        self._retriever_map = {
            "vector": self._vector_retriever,
            "graph": self._graph_retriever,
            "web": self._web_retriever,
        }

    def _execute(self, state: AgentState) -> AgentState:
        sources: List[str] = state.get("retrieval_sources", ["vector"])
        query: str = state.get("rewritten_query", state.get("user_query", ""))
        sub_queries: List[str] = state.get("sub_queries", [])

        # 对于 multi_hop 查询，同时检索主查询和子查询
        queries_to_retrieve = [query]
        if sub_queries:
            queries_to_retrieve.extend(sub_queries[:3])  # 最多追加 3 个子查询

        all_docs: List[RetrievedDocument] = []

        # 并行执行各路检索
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(sources)) as executor:
            future_map = {}
            for source in sources:
                retriever = self._retriever_map.get(source)
                if retriever:
                    future = executor.submit(
                        self._retrieve_safe, retriever, queries_to_retrieve, source
                    )
                    future_map[future] = source

            for future in concurrent.futures.as_completed(future_map):
                source = future_map[future]
                try:
                    docs = future.result()
                    all_docs.extend(docs)
                    self._logger.info(f"检索来源 '{source}' 返回 {len(docs)} 篇文档")
                except Exception as e:
                    self._logger.error(f"检索来源 '{source}' 失败: {e}")

        self._logger.info(f"多路检索完成，共获取 {len(all_docs)} 篇文档")

        # 去重（按内容哈希）
        all_docs = self._deduplicate(all_docs)

        return {**state, "retrieved_docs": all_docs}

    def _retrieve_safe(
        self,
        retriever,
        queries: List[str],
        source: str,
    ) -> List[RetrievedDocument]:
        """安全执行单路检索，异常时返回空列表"""
        try:
            docs = []
            for q in queries:
                docs.extend(retriever.retrieve(q))
            return docs
        except Exception as e:
            self._logger.warning(f"检索器 '{source}' 异常: {e}")
            return []

    def _deduplicate(self, docs: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """基于内容哈希去除重复文档"""
        seen = set()
        unique = []
        for doc in docs:
            key = hash(doc.content.strip()[:200])
            if key not in seen:
                seen.add(key)
                unique.append(doc)
        removed = len(docs) - len(unique)
        if removed:
            self._logger.debug(f"去重：移除 {removed} 篇重复文档")
        return unique
