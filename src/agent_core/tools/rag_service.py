"""RAG 知识库检索服务：`search_knowledge_base` 技能脚本（`skills/core/search-knowledge-base/
scripts/main.py`）依赖的两个服务类。

原样迁移自 `diit-agent-server` 的 `src/service/rag_service.py` + `src/service/
rag_query_rewrite_service.py`（两个文件合并成一个，参照本仓库 `web_search_tool.py`
"一个薄封装 + 一个客户端类同放一个文件"的既有约定），业务逻辑未改动，只替换了三处
依赖入口：`src.env_utils.get_settings` → `src.config.settings.get_settings`；
`LLMFactory.get_llm(LLMPurpose.UTILITY, ...)` → `create_chat_model(...)`；
`src.core.prompts.prompts.RAG_QUERY_REWRITE*`（魔法属性访问）→
`prompt_factory.get/render("RAG_QUERY_REWRITE*", ...)`（显式方法调用）。

`RAGService.format_knowledge_chunks()` 在格式化知识片段后会动态追加一段"引用要求"提示
（`[i]` 行内引用 + 结尾 `<ref_json>{"sources":[...]}}</ref_json>`）——这正是
`chat_pipeline.py` 的 `_push_ref_text`/`_parse_ref_json` 状态机实际解析的格式，原样保留
即可，不需要额外处理。
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from src.agent_core.model import create_chat_model
from src.agent_core.prompts import prompt_factory
from src.config.settings import get_settings

settings = get_settings()


class RAGService:
    """RAG 知识库检索服务：封装对外部 RAG API 的调用。"""

    def __init__(self) -> None:
        self.base_url = settings.RAG_API_URL
        logger.info(f"RAGService initialized | base_url={self.base_url}")

    async def retrieval_simple(
        self,
        query: str,
        work_flow_run_id: str,
        db_id: str = "0",
        top_k: int | None = None,
        score_threshold: float | None = None,
        vector_similarity_weight: float | None = None,
        page: int | None = None,
        page_size: int | None = None,
    ) -> list[dict[str, Any]]:
        """调用 RAG 检索接口，返回知识片段列表（每项含 id/content/score/title）。"""
        top_k = settings.RAG_TOP_K if top_k is None else top_k
        page = settings.RAG_PAGE if page is None else page
        page_size = settings.RAG_PAGE_SIZE if page_size is None else page_size
        score_threshold = settings.RAG_SCORE_THRESHOLD if score_threshold is None else score_threshold
        vector_similarity_weight = (
            settings.RAG_VECTOR_SIMILARITY_WEIGHT if vector_similarity_weight is None else vector_similarity_weight
        )

        logger.info(
            f"RAG retrieval started | query={query[:120]!r} | db_id={db_id} | top_k={top_k} "
            f"| score_threshold={score_threshold} | vector_weight={vector_similarity_weight} "
            f"| page={page} | page_size={page_size} | run_id={work_flow_run_id}"
        )

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{self.base_url}/retrievalSimple",
                    json={
                        "work_flow_run_id": work_flow_run_id,
                        "select_all": True,
                        "query": query,
                        "retrieval_setting": {
                            "top_k": top_k,
                            "score_threshold": score_threshold,
                            "vector_similarity_weight": vector_similarity_weight,
                            "page_size": page_size,
                            "page": page,
                        },
                    },
                )
                response.raise_for_status()
                results = response.json()

            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            logger.info(
                f"RAG retrieval completed | query={query[:120]!r} | results_count={len(results)} "
                f"| elapsed_ms={elapsed_ms} | run_id={work_flow_run_id}"
            )
            return results
        except httpx.HTTPError as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            logger.error(f"RAG API call failed | query={query[:120]!r} | elapsed_ms={elapsed_ms} | error={exc}")
            raise
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            logger.error(
                f"RAG retrieval unexpected error | query={query[:120]!r} | elapsed_ms={elapsed_ms} | error={exc}"
            )
            raise

    def format_knowledge_chunks(self, results: list[dict]) -> str:
        """把检索结果格式化为知识片段列表 + 引用格式提示，供 LLM 理解和引用。"""
        if not results:
            logger.warning("No RAG results to format")
            return "未找到相关知识片段。"

        chunks = []
        for i, item in enumerate(results, 1):
            chunk = (
                f'<知识片段 [{i}] id={item["id"]} title="{item["title"]}" score={item["score"]:.4f}>\n'
                f'{item["content"]}\n</知识片段>'
            )
            chunks.append(chunk)

        formatted_result = "\n\n".join(chunks)
        reminder = (
            f"\n\n---\n【引用要求】基于以上片段回答时，每句话末尾必须紧跟该片段的 [序号]，"
            f"序号范围 [1]–[{len(chunks)}]。"
            "\n正文结束后立即输出（勿加代码块）："
            '\n<ref_json>{"sources":[{"index":1,"name":"对应title"},...]}</ref_json>'
        )
        return formatted_result + reminder

    @staticmethod
    def extract_sources(results: list[dict]) -> list[dict[str, str]]:
        """从检索结果中提取结构化来源，按片段 id 去重（保留顺序）。"""
        seen: set[str] = set()
        sources: list[dict] = []
        for item in results:
            chunk_id = str(item.get("id", ""))
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            sources.append({
                "id": chunk_id, "type": "file", "name": item.get("title", ""),
                "desc": str(item.get("content", ""))[:200].replace("\n", " "), "url": "",
            })
        return sources


@dataclass
class QueryRewriteResult:
    """`RAGQueryRewriteService.rewrite_query` 的归一化返回结果。"""

    query: str
    rewritten: bool
    fallback_used: bool


class RAGQueryRewriteService:
    """把对话式检索问题改写成适合知识库检索的独立查询语句。"""

    async def rewrite_query(
        self,
        *,
        original_query: str,
        user_query: str = "",
        conversation_history: list[dict[str, Any]] | None = None,
    ) -> QueryRewriteResult:
        normalized_original = self._normalize_query(original_query)
        if not normalized_original:
            return QueryRewriteResult(query="", rewritten=False, fallback_used=True)

        if not settings.RAG_QUERY_REWRITE_ENABLED:
            logger.info(f"RAG query rewrite disabled, using original query | original_query={normalized_original[:120]!r}")
            return QueryRewriteResult(query=normalized_original, rewritten=False, fallback_used=False)

        history_window = self._slice_history(conversation_history or [])
        history_text = self._format_history(history_window)
        normalized_user_query = self._normalize_query(user_query) or normalized_original
        history_turns = len(history_window) // 2

        logger.info(
            f"RAG query rewrite started | original_query={normalized_original[:120]!r} "
            f"| user_query={normalized_user_query[:120]!r} | history_turns={history_turns}"
        )

        llm = create_chat_model(
            max_tokens=settings.RAG_QUERY_REWRITE_MAX_TOKENS, timeout=settings.RAG_QUERY_REWRITE_TIMEOUT,
        )
        messages = [
            SystemMessage(content=prompt_factory.get("RAG_QUERY_REWRITE_SYSTEM")),
            HumanMessage(content=prompt_factory.render(
                "RAG_QUERY_REWRITE", user_query=normalized_user_query,
                tool_query=normalized_original, history_text=history_text,
            )),
        ]

        t0 = time.perf_counter()
        try:
            response = await llm.ainvoke(messages)
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
            logger.warning(
                f"RAG query rewrite failed, falling back to original query "
                f"| original_query={normalized_original[:120]!r} "
                f"| error_type={type(exc).__name__} | elapsed_ms={elapsed_ms}"
            )
            return QueryRewriteResult(query=normalized_original, rewritten=False, fallback_used=True)

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        rewritten_query = self._normalize_query(self._content_to_text(getattr(response, "content", "")))
        rewritten_query = self._strip_common_prefix(rewritten_query)

        if not rewritten_query:
            logger.info(
                f"RAG query rewrite returned empty output, falling back to original query "
                f"| original_query={normalized_original[:120]!r} | elapsed_ms={elapsed_ms}"
            )
            return QueryRewriteResult(query=normalized_original, rewritten=False, fallback_used=True)

        rewrite_applied = rewritten_query != normalized_original
        logger.info(
            f"RAG query rewrite completed | original_query={normalized_original[:120]!r} "
            f"| rewritten_query={rewritten_query[:120]!r} | rewrite_applied={rewrite_applied} "
            f"| history_turns={history_turns} | elapsed_ms={elapsed_ms}"
        )
        return QueryRewriteResult(query=rewritten_query, rewritten=rewrite_applied, fallback_used=False)

    @staticmethod
    def _normalize_query(query: str) -> str:
        if not query:
            return ""
        query = query.replace("\n", " ").replace("\t", " ")
        return re.sub(r"\s+", " ", query).strip()

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text") or item.get("content")
                    parts.append(str(text) if text is not None else "")
                else:
                    parts.append(str(item))
            return "".join(parts)
        return str(content)

    @staticmethod
    def _strip_common_prefix(query: str) -> str:
        if not query:
            return ""
        query = query.strip().strip("`").strip()
        for prefix in ("重写后的查询：", "重写后的检索查询：", "检索查询：", "查询："):
            if query.startswith(prefix):
                return query[len(prefix):].strip()
        return query

    @staticmethod
    def _slice_history(conversation_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        non_empty_messages = [
            {"role": str(item.get("role", "")).strip(), "content": str(item.get("content", "")).strip()}
            for item in conversation_history
            if str(item.get("content", "")).strip()
        ]
        max_messages = settings.RAG_QUERY_REWRITE_HISTORY_TURNS * 2
        if max_messages <= 0:
            return []
        return non_empty_messages[-max_messages:]

    @staticmethod
    def _format_history(conversation_history: list[dict[str, Any]]) -> str:
        if not conversation_history:
            return "无历史对话"
        role_names = {"user": "用户", "assistant": "助手"}
        lines = []
        for message in conversation_history:
            role = role_names.get(message.get("role"), message.get("role") or "未知")
            lines.append(f"{role}: {message.get('content', '')}")
        return "\n".join(lines)
