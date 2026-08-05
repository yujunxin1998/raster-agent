#!/usr/bin/env python3
"""RAG skill main script — executed by SkillTool as a subprocess.

Reads a JSON params object from stdin, performs knowledge-base retrieval
(with optional query rewriting), and writes a structured JSON result to
stdout.  The output format is:

    {
        "content": "<formatted knowledge chunks>",
        "__metadata__": { ... retriever_resources and stats ... }
    }

SkillContentReader.run_script() parses this format and promotes
``__metadata__`` separately from ``content`` so the caller can persist
retriever_resources.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys

# ---------------------------------------------------------------------------
# Ensure the project root is on sys.path so that "src.*" imports work when
# this script is executed as a subprocess from any working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(  # diit-agent-server/
    os.path.dirname(              # skills/
        os.path.dirname(          # core/
            os.path.dirname(      # search-knowledge-base/
                _SCRIPT_DIR       # scripts/
            )
        )
    )
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.agent_core.tools.rag_service import RAGQueryRewriteService, RAGService  # noqa: E402
from src.config.settings import get_settings  # noqa: E402

settings = get_settings()


async def _run(params: dict) -> dict:
    query: str = (params.get("query") or "").strip()
    if not query:
        return {
            "content": "检索 query 不能为空。",
            "__metadata__": {"total_segments": 0, "retriever_resources": []},
        }

    top_k: int = int(params.get("top_k") or settings.RAG_TOP_K)
    workflow_id: str = str(params.get("workflow_id") or "unknown")
    db_id: str = str(params.get("db_id") or "0")
    user_query: str = (params.get("user_query") or query).strip()
    conversation_history: list = params.get("conversation_history") or []

    rag_service = RAGService()
    rewrite_service = RAGQueryRewriteService()

    rewrite_result = await rewrite_service.rewrite_query(
        original_query=query,
        user_query=user_query,
        conversation_history=conversation_history,
    )

    results = await rag_service.retrieval_simple(
        query=rewrite_result.query,
        work_flow_run_id=workflow_id,
        db_id=db_id,
        top_k=top_k,
    )

    formatted_chunks = rag_service.format_knowledge_chunks(results)

    retriever_resources = [
        {
            "position": idx + 1,
            "dataset_id": db_id,
            "dataset_name": "知识库",
            "document_id": item["id"],
            "document_name": item.get("title", "未知文档"),
            "data_source_type": "knowledge_base",
            "segment_id": item["id"],
            "score": item["score"],
            "content": item["content"],
        }
        for idx, item in enumerate(results)
    ]

    return {
        "content": formatted_chunks,
        "__metadata__": {
            "retriever_resources": retriever_resources,
            "total_segments": len(results),
            "query": rewrite_result.query,
            "db_id": db_id,
            "top_k": top_k,
            "top_score": max((item.get("score") or 0 for item in results), default=0),
        },
    }


def main() -> None:
    raw = sys.stdin.read()
    try:
        params = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        error_out = {
            "content": f"[参数解析失败: {exc}]",
            "__metadata__": {"total_segments": 0, "retriever_resources": []},
        }
        print(json.dumps(error_out, ensure_ascii=False))
        sys.exit(1)

    result = asyncio.run(_run(params))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
