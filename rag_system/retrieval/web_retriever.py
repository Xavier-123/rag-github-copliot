"""
Web Retriever（Web 搜索检索器）
================================
使用 DuckDuckGo 等搜索引擎进行实时 Web 信息检索，
补充知识库中可能没有的最新信息。

技术栈：
  - DuckDuckGo Search API (duckduckgo-search) : 免费无需 API Key
  - 支持扩展到 Google Custom Search / Bing Search API

检索流程：
  query → 搜索引擎 API → 获取标题+摘要 → 封装为 RetrievedDocument
"""

from __future__ import annotations

from typing import List

from loguru import logger

from rag_system.agents.base import RetrievedDocument
from rag_system.config.settings import get_settings


class WebRetriever:
    """
    Web 检索器
    ==========
    通过搜索引擎 API 获取实时 Web 内容，弥补知识库时效性不足的问题。

    使用说明：
    - 默认使用 DuckDuckGo（无需 API Key，免费使用）
    - 可扩展支持 Google / Bing（需配置对应 API Key）
    - 搜索结果仅使用标题和摘要，不做全文抓取（避免额外延迟）

    超时与重试：
    - 每次搜索设置独立超时，防止阻塞整体工作流
    - 失败时返回空列表，由 MultiRetrieverAgents 记录日志
    """

    def __init__(self, settings=None) -> None:
        self._cfg = settings or get_settings()
        self._logger = logger.bind(module="WebRetriever")

    def retrieve(self, query: str) -> List[RetrievedDocument]:
        """
        执行 Web 搜索检索。

        Args:
            query: 查询文本

        Returns:
            RetrievedDocument 列表（来源标记为 "web"）
        """
        if not self._cfg.web_search.enabled:
            return []

        engine = self._cfg.web_search.engine.lower()

        if engine == "duckduckgo":
            return self._search_duckduckgo(query)
        else:
            self._logger.warning(f"不支持的搜索引擎: {engine}，使用 Mock 检索")
            return self._mock_retrieve(query)

    def _search_duckduckgo(self, query: str) -> List[RetrievedDocument]:
        """
        使用 DuckDuckGo 执行搜索。
        duckduckgo-search 库提供 text() 方法，返回标题、正文摘要和 URL。
        """
        try:
            from duckduckgo_search import DDGS
            results = []
            with DDGS() as ddgs:
                search_results = list(
                    ddgs.text(
                        query,
                        max_results=self._cfg.web_search.max_results,
                    )
                )

            for i, r in enumerate(search_results):
                content = f"标题: {r.get('title', '')}\n摘要: {r.get('body', '')}"
                results.append(
                    RetrievedDocument(
                        doc_id=f"web-{i}",
                        content=content,
                        source="web",
                        score=1.0 - i * 0.05,  # 按搜索排名估算相关性
                        metadata={
                            "url": r.get("href", ""),
                            "title": r.get("title", ""),
                        },
                    )
                )
            self._logger.debug(f"DuckDuckGo 搜索返回 {len(results)} 条结果")
            return results

        except ImportError:
            self._logger.warning("duckduckgo-search 未安装，使用 Mock 检索")
            return self._mock_retrieve(query)
        except Exception as e:
            self._logger.error(f"DuckDuckGo 搜索失败: {e}")
            return self._mock_retrieve(query)

    def _mock_retrieve(self, query: str) -> List[RetrievedDocument]:
        """Mock Web 检索（搜索 API 不可用时的占位实现）"""
        self._logger.debug("使用 Mock Web 检索")
        return [
            RetrievedDocument(
                doc_id="mock-web-1",
                content=(
                    f"[Mock Web 检索] 查询: '{query}' — "
                    "Web 搜索结果占位内容。"
                    "生产环境中，此处将返回搜索引擎的实时搜索结果。"
                ),
                source="web",
                score=0.60,
                metadata={"is_mock": True, "url": "https://example.com"},
            )
        ]
