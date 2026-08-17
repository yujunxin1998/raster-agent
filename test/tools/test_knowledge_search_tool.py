"""`search_knowledge_base` 独立 Tool 单元测试（迁移自技能脚本）。"""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.tools.knowledge_search_tool import search_knowledge_base
from src.agent_core.tools.rag_service import QueryRewriteResult

_RUNTIME = SimpleNamespace(context=AgentRuntimeContext(conversation_id="c1"))


async def test_empty_query_returns_error_without_calling_service() -> None:
    result = await search_knowledge_base.coroutine(query="  ", top_k=None, runtime=_RUNTIME)

    assert "不能为空" in result


async def test_returns_formatted_chunks_with_citation_reminder() -> None:
    fake_results = [{"id": "1", "title": "文件.pdf", "content": "内容片段", "score": 0.9}]
    fake_rag_service = SimpleNamespace(
        retrieval_simple=AsyncMock(return_value=fake_results),
        format_knowledge_chunks=lambda results: "<知识片段 [1] id=1 title=\"文件.pdf\">内容片段</知识片段>\n\n---\n<ref_json>",
    )
    fake_rewrite_service = SimpleNamespace(
        rewrite_query=AsyncMock(return_value=QueryRewriteResult(query="改写后的查询", rewritten=True, fallback_used=False)),
    )

    with patch("src.agent_core.tools.knowledge_search_tool.RAGService", return_value=fake_rag_service), \
         patch("src.agent_core.tools.knowledge_search_tool.RAGQueryRewriteService", return_value=fake_rewrite_service):
        result = await search_knowledge_base.coroutine(query="地质灾害如何处理", top_k=5, runtime=_RUNTIME)

    assert "文件.pdf" in result
    assert "<ref_json>" in result
    fake_rag_service.retrieval_simple.assert_awaited_once()
    call_kwargs = fake_rag_service.retrieval_simple.await_args.kwargs
    assert call_kwargs["query"] == "改写后的查询"
    assert call_kwargs["top_k"] == 5
