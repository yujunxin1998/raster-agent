"""`search_knowledge_base` 独立业务 Tool。

对应 `docs/Skill与Tool完全解耦重构设计.md` 第 10.2 节：把原来
`skills/core/search-knowledge-base/scripts/main.py`（技能脚本，经
`SkillContentReader.run_script()` 子进程执行）里的编排逻辑迁移成一个直接
挂在 Lead Agent 工具集上的 `@tool` 函数。`RAGService`/`RAGQueryRewriteService`
（`rag_service.py`）本身不改动，只是不再经"技能脚本子进程"这层间接调用。

**引用格式兼容**（迁移必须保持的兼容点，重构文档 10.2 节明确要求）：
`RAGService.format_knowledge_chunks()` 已经在格式化文本末尾自带一段"引用要求"
提示（`[i]` 行内引用 + 结尾 `<ref_json>{...}</ref_json>` 格式说明），这段文本
原样作为工具返回值交给模型——`chat_pipeline.py` 的 `_push_ref_text`/
`_parse_ref_json` 状态机解析的是**模型自己的输出**（模型读了这段提示后在自己
的回复里生成 `<ref_json>`），不是直接解析工具返回值，所以迁移到独立 Tool 后
这条链路不需要任何改动，只要工具返回值还是 `format_knowledge_chunks()` 的原样
输出即可。

`workflow_id`/`db_id`/`user_query`/`conversation_history` 原来声明在 SKILL.md
的 `runtime_context_keys` 里，但 `AgentRuntimeContext`（`middlewares/context.py`）
从未定义过这几个字段——`SkillToolFactory._invoke` 的 `getattr(context, key,
_MISSING)` 对它们恒返回 `_MISSING`，也就是说这几个键在生产环境里从来没有被
真正注入过，脚本里的默认值（`workflow_id="unknown"`/`db_id="0"`/
`user_query=query`/`conversation_history=[]`）才是实际生效的行为。这是迁移前
就存在的既有行为（很可能是遗留的未完成接入，不是本次重构的范围——重构文档
"非目标"一节明确排除"重写 RAG 算法"），这里原样保留，不在迁移过程中顺带"修复"。
"""
from __future__ import annotations

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.tools.rag_service import RAGQueryRewriteService, RAGService
from src.config.settings import get_settings

_QUERY_EMPTY_MESSAGE = "检索 query 不能为空。"


@tool
async def search_knowledge_base(
    query: str,
    top_k: int | None,
    runtime: ToolRuntime[AgentRuntimeContext],
) -> str:
    """从知识库检索相关专业知识，适用于需要知识库证据支撑的专业问题。

    Args:
        query: 要检索的查询内容，应该是用户的原始问题或经过提炼的关键问题。
        top_k: 返回的知识片段数量，不传时使用默认值。
        runtime: 注入的运行时（当前未从中取值，见模块文档）。
    """
    query = (query or "").strip()
    if not query:
        return _QUERY_EMPTY_MESSAGE

    settings = get_settings()
    resolved_top_k = top_k if top_k is not None else settings.RAG_TOP_K

    rag_service = RAGService()
    rewrite_service = RAGQueryRewriteService()

    rewrite_result = await rewrite_service.rewrite_query(original_query=query, user_query=query)

    results = await rag_service.retrieval_simple(
        query=rewrite_result.query,
        work_flow_run_id="unknown",
        db_id="0",
        top_k=resolved_top_k,
    )

    return rag_service.format_knowledge_chunks(results)
