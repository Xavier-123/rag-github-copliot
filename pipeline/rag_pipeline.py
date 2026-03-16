"""
RAG Pipeline Orchestrator.

This is the central coordinator of the entire Agentic RAG + Self-Improving RAG
system.  It wires together all agents, memory, retrieval, evaluation, and
optimization components into a coherent, observable end-to-end workflow.

System Workflow (Happy Path)
----------------------------
    User Query
        │
        ▼
    1. ConversationMemory.get_history(session_id)
        │
        ▼
    2. QueryUnderstandingAgent
       (rewrite, intent, complexity, decompose)
        │
        ▼
    3. TaskPlanningAgent
       (build task plan)
        │
        ▼
    4. RetrievalPlanningAgent
       (select retrieval strategies)
        │
        ▼
    5. MultiRetrieverAgent
       (parallel retrieval: vector / graph / web / hybrid)
        │
        ▼
    6. RerankAgent
       (LLM-based re-ranking → top-K)
        │
        ▼
    7. ContextEngineeringAgent
       (dedup, filter, compress, order → context string)
        │
        ▼
    8. ReasoningAgent
       (CoT or multi-hop reasoning)
        │
        ▼
    9. Answer Generation
       (final LLM call with context + reasoning)
        │
        ▼
   10. ReflectionAgent
       (verify, hallucination check, confidence score)
        │
        ├─ confidence OK ──▶ proceed
        └─ confidence low ─▶ retry (max 2 retries)
        │
        ▼
   11. ConversationMemory.add_turn(session_id, ...)
        │
        ▼
   12. Evaluator.evaluate(...)   [if enabled]
        │
        ▼
   13. FeedbackLearning.learn(...)
        │
        ▼
   14. SystemOptimizer.apply(learning_update)
        │
        ▼
   15. EpisodicMemory.add_episode(...)
        │
        ▼
    ChatResponse → caller

Component Interaction
---------------------
- Each agent communicates via :class:`~agents.base_agent.AgentInput` /
  :class:`~agents.base_agent.AgentOutput` data classes.
- The pipeline passes context forward by enriching the ``metadata`` dict.
- Retry logic is governed by :class:`~agents.reflection_agent.ReflectionAgent`
  confidence scores.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agents.base_agent import AgentInput
from agents.context_engineering_agent import ContextEngineeringAgent
from agents.multi_retriever_agent import MultiRetrieverAgent
from agents.query_understanding_agent import QueryUnderstandingAgent
from agents.reasoning_agent import ReasoningAgent
from agents.reflection_agent import ReflectionAgent
from agents.rerank_agent import RerankAgent
from agents.retrieval_planning_agent import RetrievalPlanningAgent
from agents.task_planning_agent import TaskPlanningAgent
from config.settings import settings
from evaluation.evaluator import Evaluator
from memory.conversation_memory import ConversationMemory
from memory.episodic_memory import EpisodicMemory
from optimization.feedback_learning import FeedbackLearning
from optimization.system_optimizer import SystemOptimizer
from utils.helpers import build_llm_client, get_logger, get_model_name, is_llm_available, truncate_text


logger = get_logger("RAGPipeline")


# ---------------------------------------------------------------------------
# Response data class
# ---------------------------------------------------------------------------

@dataclass
class ChatResponse:
    """
    The final response returned to the caller after a complete pipeline pass.

    Attributes:
        answer:          The generated answer string.
        session_id:      Session identifier.
        query:           Original user query.
        rewritten_query: Query after rewriting.
        sources:         List of source identifiers for cited documents.
        confidence:      Final confidence score from the Reflection Agent.
        evaluation:      Quality metrics from the Evaluation Layer (if enabled).
        reasoning_chain: Ordered reasoning steps used to derive the answer.
        metadata:        Arbitrary per-step diagnostic data.
        elapsed_ms:      Total wall-clock time for the full pipeline.
        error:           Non-empty if a non-recoverable error occurred.
    """

    answer: str
    session_id: str
    query: str
    rewritten_query: str = ""
    sources: List[str] = field(default_factory=list)
    confidence: float = 1.0
    evaluation: Dict[str, float] = field(default_factory=dict)
    reasoning_chain: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    elapsed_ms: float = 0.0
    error: str = ""


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAGPipeline:
    """
    End-to-end Agentic RAG + Self-Improving RAG pipeline.

    All major components are instantiated once and reused across calls,
    enabling warm caches (LLM clients, vector store connections, BM25 index).

    Usage
    -----
    >>> pipeline = RAGPipeline()
    >>> response = pipeline.chat("What is RAG?", session_id="user-123")
    >>> print(response.answer)
    """

    _MAX_RETRIES = 2  # Maximum reflection-driven retries

    def __init__(self) -> None:
        logger.info("Initialising RAGPipeline …")

        # -- Agents --
        self._query_understanding = QueryUnderstandingAgent()
        self._task_planning = TaskPlanningAgent()
        self._retrieval_planning = RetrievalPlanningAgent()
        self._multi_retriever = MultiRetrieverAgent()
        self._reranker = RerankAgent()
        self._context_engineer = ContextEngineeringAgent()
        self._reasoning = ReasoningAgent()
        self._reflection = ReflectionAgent()

        # -- Memory --
        self._conversation_memory = ConversationMemory()
        self._episodic_memory = EpisodicMemory()

        # -- Evaluation & Optimization --
        self._evaluator = Evaluator()
        self._feedback_learning = FeedbackLearning(self._episodic_memory)
        self._system_optimizer = SystemOptimizer()

        # -- LLM client (for final answer generation) --
        self._llm: Optional[Any] = None

        logger.info("RAGPipeline ready.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chat(
        self,
        query: str,
        session_id: Optional[str] = None,
        reference_answer: Optional[str] = None,
    ) -> ChatResponse:
        """
        Process a user query through the full Agentic RAG pipeline.

        Args:
            query:            Raw user query string.
            session_id:       Session identifier.  A new UUID is generated if
                              not provided.
            reference_answer: Optional gold-standard answer for evaluation.

        Returns:
            :class:`ChatResponse` with the final answer and all metadata.
        """
        start = time.perf_counter()
        if not session_id:
            session_id = str(uuid.uuid4())

        logger.info("Pipeline.chat | session=%s | query=%.80s", session_id, query)

        try:
            response = self._run_pipeline(query, session_id, reference_answer)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline error: %s", exc)
            response = ChatResponse(
                answer="I'm sorry, an internal error occurred. Please try again.",
                session_id=session_id,
                query=query,
                error=str(exc),
            )

        response.elapsed_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "Pipeline.chat done | session=%s | elapsed=%.1fms | confidence=%.2f",
            session_id, response.elapsed_ms, response.confidence,
        )
        return response

    def add_documents(self, documents: List[Dict[str, Any]]) -> int:
        """
        Add documents to the knowledge base for retrieval.

        Delegates to the underlying retriever backends.

        Args:
            documents: List of document dicts with ``"content"`` keys.

        Returns:
            Number of document chunks indexed.
        """
        from memory.knowledge_memory import KnowledgeMemory  # noqa: PLC0415
        km = KnowledgeMemory()
        return km.add_documents(documents)

    def get_optimizer_status(self) -> Dict[str, Any]:
        """Return the current system optimizer status for monitoring."""
        return self._system_optimizer.get_status()

    # ------------------------------------------------------------------
    # Private pipeline execution
    # ------------------------------------------------------------------

    def _run_pipeline(
        self, query: str, session_id: str, reference_answer: Optional[str]
    ) -> ChatResponse:
        """
        Execute the full pipeline and return a :class:`ChatResponse`.

        This method implements the pipeline workflow described in the module
        docstring.  It supports up to :attr:`_MAX_RETRIES` reflection-driven
        retries.
        """
        # 1. Retrieve conversation history
        history = self._conversation_memory.get_history(session_id)

        # 2. Query Understanding
        qu_output = self._query_understanding.execute(
            AgentInput(query=query, conversation_history=history, session_id=session_id)
        )
        qu = qu_output.result  # QueryUnderstanding dataclass

        # 3. Task Planning
        tp_output = self._task_planning.execute(
            AgentInput(
                query=qu.rewritten_query,
                session_id=session_id,
                metadata={"query_understanding": qu_output.metadata},
            )
        )
        task_plan = tp_output.result

        # 4. Retrieval Planning
        rp_output = self._retrieval_planning.execute(
            AgentInput(
                query=qu.rewritten_query,
                session_id=session_id,
                metadata={
                    "query_understanding": qu_output.metadata,
                    "task_plan": tp_output.metadata,
                },
            )
        )
        strategies = rp_output.result  # List[RetrievalStrategy]

        # 5. Multi-Retriever (parallel retrieval)
        mr_output = self._multi_retriever.execute(
            AgentInput(
                query=qu.rewritten_query,
                session_id=session_id,
                metadata={
                    "retrieval_strategies": rp_output.metadata.get("strategies", []),
                },
            )
        )
        retrieved_docs = mr_output.result or []

        # Reflection-driven retry loop
        answer = ""
        reflection_result = None
        context_str = ""
        reasoning_chain: List[str] = []

        for attempt in range(self._MAX_RETRIES + 1):
            if attempt > 0:
                logger.info("Retry attempt %d/%d", attempt, self._MAX_RETRIES)

            # 6. Rerank
            rr_output = self._reranker.execute(
                AgentInput(
                    query=qu.rewritten_query,
                    context=retrieved_docs,
                    session_id=session_id,
                )
            )
            reranked_docs = rr_output.result or []

            # 7. Context Engineering
            ce_output = self._context_engineer.execute(
                AgentInput(
                    query=qu.rewritten_query,
                    context=reranked_docs,
                    session_id=session_id,
                )
            )
            context_str = ce_output.result or ""

            # 8. Reasoning
            rsn_output = self._reasoning.execute(
                AgentInput(
                    query=qu.rewritten_query,
                    session_id=session_id,
                    metadata={
                        "context_str": context_str,
                        "task_plan": tp_output.metadata,
                    },
                )
            )
            reasoning_result = rsn_output.result
            reasoning_chain = getattr(reasoning_result, "reasoning_chain", [])

            # 9. Answer Generation
            answer = self._generate_answer(
                query=qu.rewritten_query,
                context_str=context_str,
                reasoning=getattr(reasoning_result, "final_reasoning", ""),
                history=history,
            )

            # 10. Reflection
            ref_output = self._reflection.execute(
                AgentInput(
                    query=query,
                    session_id=session_id,
                    metadata={
                        "candidate_answer": answer,
                        "context_str": context_str,
                    },
                )
            )
            reflection_result = ref_output.result

            if not reflection_result.should_retry:
                break  # Quality gate passed

            logger.info(
                "Reflection triggered retry: %s", reflection_result.retry_reason
            )

        # 11. Update conversation memory
        self._conversation_memory.add_turn(session_id, "user", query)
        self._conversation_memory.add_turn(session_id, "assistant", answer)

        # Collect source documents
        sources = list({
            doc.get("source", "") for doc in reranked_docs if doc.get("source")
        })

        # 12–14. Evaluation + Feedback + Optimization (async-like: non-blocking)
        eval_scores: Dict[str, float] = {}
        if settings.evaluation_enabled and context_str:
            try:
                eval_result = self._evaluator.evaluate(
                    query=query,
                    answer=answer,
                    context=context_str.split("\n\n---\n\n"),
                    reference_answer=reference_answer,
                )
                eval_scores = {
                    "faithfulness": eval_result.faithfulness,
                    "answer_relevance": eval_result.answer_relevance,
                    "context_precision": eval_result.context_precision,
                    "context_recall": eval_result.context_recall,
                    "overall": eval_result.overall_score,
                }

                if settings.feedback_learning_enabled:
                    learning_update = self._feedback_learning.learn(
                        eval_result=eval_result,
                        episode_context={
                            "session_id": session_id,
                            "query": query,
                            "intent": qu.intent,
                            "strategy_used": (
                                strategies[0].mode.value if strategies else "unknown"
                            ),
                        },
                    )
                    self._system_optimizer.apply(learning_update)

            except Exception as exc:  # noqa: BLE001
                logger.warning("Evaluation/feedback failed (non-critical): %s", exc)

        return ChatResponse(
            answer=answer,
            session_id=session_id,
            query=query,
            rewritten_query=qu.rewritten_query,
            sources=sources,
            confidence=getattr(reflection_result, "confidence", 1.0),
            evaluation=eval_scores,
            reasoning_chain=reasoning_chain,
            metadata={
                "intent": qu.intent,
                "complexity": qu.complexity,
                "num_retrieved": len(retrieved_docs),
                "num_reranked": len(reranked_docs),
            },
        )

    # ------------------------------------------------------------------
    # Answer Generation
    # ------------------------------------------------------------------

    def _generate_answer(
        self,
        query: str,
        context_str: str,
        reasoning: str,
        history: List[Dict[str, str]],
    ) -> str:
        """
        Generate the final answer using the LLM.

        The system prompt instructs the LLM to:
        - Answer based only on the provided context.
        - Use the reasoning narrative as an internal guide.
        - Cite sources where relevant.
        - Acknowledge when the context does not contain enough information.

        Args:
            query:       (Rewritten) user query.
            context_str: Formatted context from the Context Engineering Agent.
            reasoning:   Reasoning narrative from the Reasoning Agent.
            history:     Recent conversation history for multi-turn coherence.

        Returns:
            Generated answer string.
        """
        if not is_llm_available():
            return (
                f"[Mock Answer] Based on the retrieved context, here is a response "
                f"to your query: '{query}'. "
                "Please configure an OpenAI or Azure OpenAI API key for real responses."
            )

        if self._llm is None:
            self._llm = build_llm_client()

        context_truncated = truncate_text(context_str, max_tokens=6000)

        # Build conversation messages
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful, accurate, and concise AI assistant. "
                    "Answer the user's question using ONLY the provided context. "
                    "If the context does not contain enough information, say so clearly. "
                    "Do not fabricate facts. Cite specific documents when possible."
                ),
            }
        ]

        # Include recent conversation history (last 4 turns)
        for turn in history[-4:]:
            messages.append({"role": turn["role"], "content": turn["content"]})

        # Add the current context + reasoning + question
        user_content = (
            f"Context:\n{context_truncated}\n\n"
        )
        if reasoning:
            user_content += f"Reasoning:\n{reasoning}\n\n"
        user_content += f"Question: {query}"

        messages.append({"role": "user", "content": user_content})

        try:
            response = self._llm.chat.completions.create(
                model=get_model_name(),
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:  # noqa: BLE001
            logger.error("Answer generation failed: %s", exc)
            return f"An error occurred while generating the answer: {exc}"
