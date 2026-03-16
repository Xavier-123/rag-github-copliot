"""
Graph Retriever (GraphRAG).

Retrieves documents / entities by traversing a knowledge graph stored in
**Neo4j**.  This is particularly effective for queries about relationships,
hierarchies, or multi-hop connections between entities.

Algorithm
---------
1. Extract named entities from the query (NER via LLM or simple heuristics).
2. Find matching nodes in the Neo4j graph.
3. Traverse the graph up to N hops to collect related context.
4. Format node properties and relationship paths as text documents.

This is a **partial implementation** – the graph traversal logic is complete
but the NER step and relationship-to-text formatting can be extended for
domain-specific use cases.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from config.settings import settings
from retrieval.base_retriever import BaseRetriever


class GraphRetriever(BaseRetriever):
    """
    Knowledge-graph based retrieval using Neo4j.

    The retriever performs Cypher queries to find nodes and paths related
    to the entities mentioned in the user query.
    """

    _MAX_HOPS = 2  # Maximum graph traversal depth

    def __init__(self) -> None:
        super().__init__("GraphRetriever")
        self._driver: Optional[Any] = None

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
        Retrieve context by graph traversal for the given query.

        Args:
            query:   Natural-language query (entities are extracted from this).
            top_k:   Maximum number of result paths / nodes to return.
            filters: Optional additional Cypher WHERE clauses (not yet implemented).

        Returns:
            List of document dicts where each document represents a graph path
            or entity with its properties.
        """
        entities = self._extract_entities(query)
        if not entities:
            self.logger.warning("No entities extracted; returning empty results.")
            return []

        driver = self._get_driver()
        if driver is None:
            self.logger.warning("Neo4j unavailable; returning mock graph results.")
            return self._mock_graph_results(query, top_k)

        documents: List[Dict[str, Any]] = []
        for entity in entities[:3]:  # Limit to top-3 entities
            docs = self._traverse_graph(driver, entity, top_k)
            documents.extend(docs)

        return documents[:top_k]

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the Neo4j graph as nodes.

        Each document becomes a ``Document`` node with ``content`` and
        ``source`` properties.

        Args:
            documents: List of document dicts.
        """
        driver = self._get_driver()
        if driver is None:
            self.logger.warning("Neo4j unavailable; skipping graph ingestion.")
            return

        with driver.session() as session:
            for doc in documents:
                session.run(
                    """
                    MERGE (d:Document {source: $source})
                    SET d.content = $content,
                        d.metadata = $metadata
                    """,
                    source=doc.get("source", "unknown"),
                    content=doc.get("content", ""),
                    metadata=str(doc.get("metadata", {})),
                )
        self.logger.info("Ingested %d documents into Neo4j.", len(documents))

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_driver(self) -> Optional[Any]:
        """Lazily initialise the Neo4j driver."""
        if self._driver is not None:
            return self._driver
        if not settings.neo4j_password:
            self.logger.info("Neo4j password not configured; skipping connection.")
            return None
        try:
            from neo4j import GraphDatabase  # noqa: PLC0415
            self._driver = GraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_username, settings.neo4j_password),
            )
            self.logger.info("Connected to Neo4j at %s", settings.neo4j_uri)
            return self._driver
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Cannot connect to Neo4j (%s).", exc)
            return None

    def _extract_entities(self, query: str) -> List[str]:
        """
        Extract named entities from the query using simple heuristics.

        Production implementation should use an NER model or LLM-based
        entity extraction.  This heuristic treats capitalised words and
        quoted phrases as potential entities.

        Args:
            query: Natural-language query string.

        Returns:
            List of entity strings.
        """
        import re  # noqa: PLC0415
        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"', query)
        # Extract sequences of capitalised words (naive NER)
        capitalised = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        entities = quoted + capitalised
        # De-duplicate while preserving order
        seen: set = set()
        unique = []
        for e in entities:
            if e not in seen:
                seen.add(e)
                unique.append(e)
        return unique or [query]  # fallback: use full query

    def _traverse_graph(
        self, driver: Any, entity: str, top_k: int
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher traversal query for a single entity.

        The query finds all paths up to :attr:`_MAX_HOPS` hops from nodes
        whose name matches the entity string.

        Args:
            driver: Active Neo4j driver instance.
            entity: Entity name to start traversal from.
            top_k:  Maximum number of paths to return.

        Returns:
            List of document dicts describing the retrieved paths.
        """
        cypher = f"""
        MATCH path = (n)-[*1..{self._MAX_HOPS}]->(m)
        WHERE toLower(n.name) CONTAINS toLower($entity)
           OR toLower(n.content) CONTAINS toLower($entity)
        RETURN path
        LIMIT $limit
        """
        results: List[Dict[str, Any]] = []
        try:
            with driver.session() as session:
                for record in session.run(cypher, entity=entity, limit=top_k):
                    path_text = self._path_to_text(record["path"])
                    results.append({
                        "content": path_text,
                        "source": f"neo4j://graph/{entity}",
                        "score": 1.0,
                        "metadata": {"entity": entity, "type": "graph_path"},
                    })
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Graph traversal failed for entity '%s': %s", entity, exc)
        return results

    @staticmethod
    def _path_to_text(path: Any) -> str:
        """
        Convert a Neo4j path object into a human-readable text string.

        Args:
            path: ``neo4j.graph.Path`` object.

        Returns:
            Text describing the path as a chain of (node → rel → node) triples.
        """
        try:
            nodes = list(path.nodes)
            rels = list(path.relationships)
            parts = []
            for i, node in enumerate(nodes):
                name = node.get("name") or node.get("content", str(node.id))[:80]
                parts.append(name)
                if i < len(rels):
                    parts.append(f"--[{rels[i].type}]-->")
            return " ".join(parts)
        except Exception:  # noqa: BLE001
            return str(path)

    @staticmethod
    def _mock_graph_results(query: str, top_k: int) -> List[Dict[str, Any]]:
        """Return placeholder results when Neo4j is unavailable."""
        return [
            {
                "content": f"[Mock graph path {i+1}] Entity relationships for: {query}",
                "source": f"neo4j://mock/{i+1}",
                "score": 0.9 - i * 0.05,
                "metadata": {"mock": True, "type": "graph_path"},
            }
            for i in range(min(top_k, 2))
        ]
