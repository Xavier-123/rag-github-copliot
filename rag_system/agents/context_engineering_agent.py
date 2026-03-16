"""
Context Engineering Agent（上下文工程 Agent）
============================================
对重排序后的检索结果进行上下文工程优化，提高 LLM 输入质量：
  1. Compression（压缩）   : 移除冗余内容，保留核心信息
  2. Filtering（过滤）     : 过滤低相关性或噪声文档
  3. Summarization（摘要） : 对长文档生成简洁摘要

工作流位置：RerankAgent → [ContextEngineeringAgent] → ReasoningAgent

调用关系：
  - 输入 state 字段  : reranked_docs, rewritten_query, rag_strategy
  - 输出 state 字段  : context（最终传给 ReasoningAgent 的上下文字符串）

上下文构建策略：
  - 每篇文档以结构化格式拼接（包含来源标注）
  - 总上下文长度受 max_context_length 限制，超出则截断
  - 文档按重排得分排序，高质量文档优先出现在上下文前部
"""

from __future__ import annotations

from typing import List, Optional

from rag_system.agents.base import AgentState, BaseAgent, RetrievedDocument
from rag_system.config.settings import get_settings


# 上下文最大字符数（约 4000 tokens）
MAX_CONTEXT_CHARS = 16000
# 单文档最大字符数（超出则摘要）
MAX_DOC_CHARS = 3000
# 过滤阈值（低于此分数的文档被过滤）
FILTER_SCORE_THRESHOLD = 0.1

COMPRESSION_SYSTEM_PROMPT = "你是一个专业的信息提炼助手，负责从文档中提取与查询最相关的核心信息。"
COMPRESSION_PROMPT_TEMPLATE = """
请从以下文档中提取与问题最相关的核心信息，保持事实准确性，去除冗余内容。

## 用户问题
{query}

## 原始文档
{content}

## 要求
- 提取与问题直接相关的关键信息
- 保留原文中的数据、事实、引用
- 输出长度控制在原文的 30%-50%
- 直接输出压缩后的内容，不要加任何前缀说明
"""


class ContextEngineeringAgent(BaseAgent):
    """
    上下文工程 Agent
    ================
    通过过滤、压缩、摘要三步优化检索上下文，减少 LLM 幻觉，提升答案质量。

    处理流程：
    1. **过滤**  : 移除相关性分数 < 阈值的文档（噪声过滤）
    2. **压缩**  : 对超长文档使用 LLM 提炼关键信息（可选，配置控制）
    3. **拼接**  : 将文档按结构化格式组装为最终上下文字符串
    4. **截断**  : 控制总上下文不超过 MAX_CONTEXT_CHARS

    注意：LLM 压缩为可选功能（enable_compression=True），默认关闭以节省 API 调用。
    """

    def __init__(self, enable_compression: bool = False, settings=None) -> None:
        super().__init__("ContextEngineeringAgent", settings)
        self._cfg = get_settings()
        self._enable_compression = enable_compression
        self._llm: Optional[object] = None

        if enable_compression:
            self._llm = True  # 标记需要懒加载

    def _get_llm(self):
        """懒加载 LLM 实例"""
        if not isinstance(self._llm, object) or self._llm is True:
            from langchain_openai import ChatOpenAI
            self._llm = ChatOpenAI(
                model=self._cfg.llm.model_name,
                temperature=0.0,
                api_key=self._cfg.llm.api_key,
                base_url=self._cfg.llm.api_base,
            )
        return self._llm

    def _execute(self, state: AgentState) -> AgentState:
        docs: List[RetrievedDocument] = state.get("reranked_docs", [])
        query: str = state.get("rewritten_query", state.get("user_query", ""))

        if not docs:
            self._logger.warning("无文档可处理，上下文为空")
            return {**state, "context": ""}

        # Step 1: 过滤低相关性文档
        filtered_docs = self._filter_docs(docs)
        self._logger.debug(f"过滤后文档数: {len(filtered_docs)}/{len(docs)}")

        if not filtered_docs:
            # 若全部被过滤，保留得分最高的文档
            filtered_docs = [max(docs, key=lambda d: d.score)]

        # Step 2: 压缩超长文档（可选）
        if self._enable_compression and self._llm:
            filtered_docs = self._compress_docs(query, filtered_docs)

        # Step 3: 构建上下文字符串
        context = self._build_context(filtered_docs)

        # Step 4: 长度截断
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS] + "\n...[内容截断]"

        self._logger.info(
            f"上下文工程完成: {len(filtered_docs)} 篇文档, "
            f"上下文长度 {len(context)} 字符"
        )

        return {**state, "context": context}

    def _filter_docs(self, docs: List[RetrievedDocument]) -> List[RetrievedDocument]:
        """过滤相关性得分低于阈值的文档"""
        return [doc for doc in docs if doc.score >= FILTER_SCORE_THRESHOLD]

    def _compress_docs(
        self, query: str, docs: List[RetrievedDocument]
    ) -> List[RetrievedDocument]:
        """
        使用 LLM 对超长文档进行信息压缩。
        仅压缩超过 MAX_DOC_CHARS 的文档，以节省 API 调用。
        """
        from langchain.schema import HumanMessage, SystemMessage
        compressed = []
        for doc in docs:
            if len(doc.content) > MAX_DOC_CHARS:
                try:
                    prompt = COMPRESSION_PROMPT_TEMPLATE.format(
                        query=query, content=doc.content[:MAX_DOC_CHARS * 2]
                    )
                    response = self._get_llm().invoke(  # type: ignore
                        [
                            SystemMessage(content=COMPRESSION_SYSTEM_PROMPT),
                            HumanMessage(content=prompt),
                        ]
                    )
                    doc.content = response.content
                    doc.metadata["compressed"] = True
                    self._logger.debug(f"文档 {doc.doc_id} 已压缩")
                except Exception as e:
                    self._logger.warning(f"文档压缩失败，保留原文: {e}")
            compressed.append(doc)
        return compressed

    def _build_context(self, docs: List[RetrievedDocument]) -> str:
        """
        将文档列表构建为结构化上下文字符串。
        每篇文档标注来源和序号，方便 LLM 引用。
        """
        parts = []
        for i, doc in enumerate(docs, 1):
            source_label = {
                "vector": "知识库",
                "graph": "知识图谱",
                "web": "Web搜索",
            }.get(doc.source, doc.source)

            parts.append(
                f"[文档 {i}] 来源: {source_label} | 相关度: {doc.score:.3f}\n"
                f"{doc.content.strip()}"
            )

        return "\n\n---\n\n".join(parts)
