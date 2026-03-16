"""
Web Retriever.

Performs live internet searches to retrieve up-to-date information that may
not be present in the local knowledge base.  Supports three providers:

- **Serper** (Google Search API)
- **Tavily** (purpose-built search for LLMs)
- **Bing** (Microsoft Bing Search v7 API)

The provider is selected from ``settings.web_search_provider``.

Algorithm
---------
1. Send the query to the selected search API.
2. Parse the JSON response to extract snippet + URL pairs.
3. Return them as unified document dicts.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError

from config.settings import settings
from retrieval.base_retriever import BaseRetriever


class WebRetriever(BaseRetriever):
    """
    Live web search retrieval.

    Results from the search API are formatted as documents with the page
    snippet as ``content`` and the URL as ``source``.
    """

    def __init__(self) -> None:
        super().__init__("WebRetriever")
        self._provider = settings.web_search_provider.lower()

    # ------------------------------------------------------------------
    # BaseRetriever implementation
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search the web and return the top-K results.

        Args:
            query:   Search query string.
            top_k:   Maximum number of search results to return.
            filters: Currently unused for web search.

        Returns:
            List of document dicts from search results.
        """
        dispatch = {
            "serper": self._search_serper,
            "tavily": self._search_tavily,
            "bing": self._search_bing,
        }
        search_fn = dispatch.get(self._provider, self._mock_search)
        try:
            return search_fn(query, top_k)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Web search failed (%s); returning empty results.", exc)
            return []

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Web retriever does not support adding documents (read-only)."""
        self.logger.info("WebRetriever is read-only; add_documents is a no-op.")

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _search_serper(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Search using the Serper.dev Google Search API.

        Endpoint: ``POST https://google.serper.dev/search``

        Args:
            query:  Search query.
            top_k:  Maximum results to return.

        Returns:
            List of document dicts.
        """
        if not settings.serper_api_key:
            self.logger.warning("Serper API key not configured; using mock.")
            return self._mock_search(query, top_k)

        payload = json.dumps({"q": query, "num": top_k}).encode()
        req = Request(
            "https://google.serper.dev/search",
            data=payload,
            headers={
                "X-API-KEY": settings.serper_api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            return self._parse_serper_results(data, top_k)
        except URLError as exc:
            self.logger.warning("Serper request failed: %s", exc)
            return []

    def _parse_serper_results(
        self, data: Dict[str, Any], top_k: int
    ) -> List[Dict[str, Any]]:
        docs = []
        for item in data.get("organic", [])[:top_k]:
            docs.append({
                "content": item.get("snippet", item.get("title", "")),
                "source": item.get("link", ""),
                "score": 1.0,
                "metadata": {
                    "title": item.get("title", ""),
                    "provider": "serper",
                },
            })
        return docs

    def _search_tavily(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Search using the Tavily Search API.

        Endpoint: ``POST https://api.tavily.com/search``

        Args:
            query:  Search query.
            top_k:  Maximum results to return.

        Returns:
            List of document dicts.
        """
        if not settings.tavily_api_key:
            self.logger.warning("Tavily API key not configured; using mock.")
            return self._mock_search(query, top_k)

        payload = json.dumps({
            "api_key": settings.tavily_api_key,
            "query": query,
            "max_results": top_k,
            "search_depth": "basic",
        }).encode()
        req = Request(
            "https://api.tavily.com/search",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            return self._parse_tavily_results(data, top_k)
        except URLError as exc:
            self.logger.warning("Tavily request failed: %s", exc)
            return []

    def _parse_tavily_results(
        self, data: Dict[str, Any], top_k: int
    ) -> List[Dict[str, Any]]:
        docs = []
        for item in data.get("results", [])[:top_k]:
            docs.append({
                "content": item.get("content", item.get("snippet", "")),
                "source": item.get("url", ""),
                "score": item.get("score", 1.0),
                "metadata": {
                    "title": item.get("title", ""),
                    "provider": "tavily",
                },
            })
        return docs

    def _search_bing(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Search using the Microsoft Bing Web Search API v7.

        Endpoint: ``GET https://api.bing.microsoft.com/v7.0/search``

        Args:
            query:  Search query.
            top_k:  Maximum results to return.

        Returns:
            List of document dicts.
        """
        if not settings.bing_search_api_key:
            self.logger.warning("Bing API key not configured; using mock.")
            return self._mock_search(query, top_k)

        from urllib.parse import urlencode  # noqa: PLC0415
        params = urlencode({"q": query, "count": top_k})
        req = Request(
            f"https://api.bing.microsoft.com/v7.0/search?{params}",
            headers={"Ocp-Apim-Subscription-Key": settings.bing_search_api_key},
        )
        try:
            with urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            return self._parse_bing_results(data, top_k)
        except URLError as exc:
            self.logger.warning("Bing request failed: %s", exc)
            return []

    def _parse_bing_results(
        self, data: Dict[str, Any], top_k: int
    ) -> List[Dict[str, Any]]:
        docs = []
        pages = data.get("webPages", {}).get("value", [])
        for item in pages[:top_k]:
            docs.append({
                "content": item.get("snippet", ""),
                "source": item.get("url", ""),
                "score": 1.0,
                "metadata": {
                    "title": item.get("name", ""),
                    "provider": "bing",
                },
            })
        return docs

    @staticmethod
    def _mock_search(query: str, top_k: int) -> List[Dict[str, Any]]:
        """Return placeholder web results for testing."""
        return [
            {
                "content": f"[Mock web result {i+1}] Information found online about: {query}",
                "source": f"https://mock-search.example.com/result/{i+1}",
                "score": 1.0 - i * 0.05,
                "metadata": {"mock": True, "provider": "mock"},
            }
            for i in range(min(top_k, 3))
        ]
