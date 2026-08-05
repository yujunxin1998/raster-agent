"""FastAPI 应用入口。

组装顺序（lifespan 内）：数据库连接池 → 各 Store 建表 → workspace/sandbox/
guardrail 三个基础设施 → skill 机制 → memory 机制 → 会话/消息存储 →
Lead Agent checkpointer → 内部工具过滤名单。这个顺序对应各模块间的真实依赖
关系（skill 机制依赖 guardrail + sandbox，guardrail 依赖 storage 层的两个
Store，均需要在此之前完成初始化；Lead Agent 依赖前面全部基础设施都已就绪）。

`/chat`（REST，非流式）、`/conversations`、`/ws/chat`（WebSocket 流式 + 前端
工具双通道回环）三组接口对应设计文档 4.1 节"Lead Agent + 委派工具"的落地。
配置了 `REDIS_URL` 时 `EvalMiddleware` 会把请求级观测数据写入 Redis Streams
（`eval:trace`/`eval:tool`/`eval:http`），未配置时全程静默降级。
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from loguru import logger

from src.agent_core.agents.checkpointer import init_checkpointer
from src.agent_core.eval.emitter import aclose_emitter_redis
from src.agent_core.eval.http_middleware import EvalMiddleware, aclose_middleware_redis
from src.agent_core.guardrail import init_guardrail_provider
from src.agent_core.memory import (
    ElasticsearchMemoryStore,
    MemoryCompressor,
    MemoryExtractor,
    MemoryManager,
    init_memory_manager,
)
from src.agent_core.sandbox import get_sandbox_provider, init_local_sandbox_provider
from src.agent_core.skills import init_skill_manager
from src.agent_core.tools.tool_filter import tool_filter_registry
from src.agent_core.workspace import get_thread_workspace_manager, init_thread_workspace_manager
from src.api.router import (
    chat_router,
    conversation_router,
    health_router,
    memory_router,
    skill_router,
)
from src.api.websocket import chat_ws
from src.config.settings import get_settings
from src.storage.conversation_store import init_conversation_store
from src.storage.database import close_database_pool, init_database_pool
from src.storage.memory_audit_store import init_memory_audit_store
from src.storage.memory_jobs_store import init_memory_jobs_store
from src.storage.message_store import init_message_store
from src.storage.skill_settings_store import init_skill_settings_store
from src.storage.tool_permission_store import init_tool_permission_store

settings = get_settings()

# 内部工具：不推送给前端、不持久化。对应原项目 main.py 里的
# `tool_filter.register("save_memory", "recall_memory")` 调用点。
tool_filter_registry.register("save_memory", "recall_memory")

logger.info(f"当前启用的环境为: {settings.CURRENT_ENV}")


def _build_memory_manager(memory_store: ElasticsearchMemoryStore) -> MemoryManager:
    """按当前配置装配一个 MemoryManager 实例（提取器 + 压缩器 + 各项行为参数）。"""
    extractor = MemoryExtractor(
        model_name=settings.MEMORY_EXTRACT_MODEL or settings.DEFAULT_MODEL,
        provider=settings.PROVIDER,
        api_key=settings.API_KEY,
        base_url=settings.BASE_URL,
        default_importance_threshold=settings.MEMORY_IMPORTANCE_THRESHOLD,
        duplicate_score_threshold=settings.MEMORY_DUPLICATE_SCORE_THRESHOLD,
        conflict_score_threshold=settings.MEMORY_CONFLICT_SCORE_THRESHOLD,
        low_confidence_margin=settings.MEMORY_LOW_CONFIDENCE_MARGIN,
    )
    compressor = MemoryCompressor(
        model_name=settings.MEMORY_COMPRESS_MODEL or settings.DEFAULT_MODEL,
        provider=settings.PROVIDER,
        api_key=settings.API_KEY,
        base_url=settings.BASE_URL,
        default_threshold=settings.MEMORY_COMPRESSION_THRESHOLD,
        default_keep_recent=settings.MEMORY_KEEP_RECENT,
    )
    return MemoryManager(
        store=memory_store,
        extractor=extractor,
        compressor=compressor,
        memory_enabled=settings.MEMORY_ENABLED,
        injection_enabled=settings.MEMORY_INJECTION_ENABLED,
        auto_extract_enabled=settings.MEMORY_AUTO_EXTRACT_ENABLED,
        recall_candidate_k=settings.MEMORY_RECALL_CANDIDATE_K,
        max_recall=settings.MEMORY_MAX_RECALL,
        min_recall_score=settings.MEMORY_MIN_RECALL_SCORE,
        max_context_tokens=settings.MEMORY_MAX_CONTEXT_TOKENS,
        importance_threshold=settings.MEMORY_IMPORTANCE_THRESHOLD,
        sensitive_filter_enabled=settings.MEMORY_SENSITIVE_FILTER_ENABLED,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── 启动 ────────────────────────────────────────────────

    # 1. 数据库连接池
    pool_manager = await init_database_pool(settings.DATABASE_URL)
    pool = pool_manager.pool

    # 2. 各 Store 建表（skill 开关 / 工具权限 / 记忆审计 / 记忆后处理任务 / 会话 / 消息）
    skill_settings_store = await init_skill_settings_store(pool)
    tool_permission_store = await init_tool_permission_store(pool)
    await init_memory_audit_store(pool)
    await init_memory_jobs_store(pool)
    await init_conversation_store(pool)
    await init_message_store(pool)

    # 3. 虚拟工作区（按会话隔离目录）
    init_thread_workspace_manager(settings.WORKSPACE_ROOT)
    workspace_manager = get_thread_workspace_manager()

    # 4. 本地沙箱提供者
    init_local_sandbox_provider(
        workspace_manager,
        default_timeout_seconds=settings.SANDBOX_COMMAND_TIMEOUT_SECONDS,
        max_output_bytes=settings.SANDBOX_MAX_OUTPUT_BYTES,
        max_memory_mb=settings.SANDBOX_MAX_MEMORY_MB,
    )
    sandbox_provider = get_sandbox_provider()

    # 5. 权限控制（组合技能开关表 + 工具权限表）
    guardrail_provider = init_guardrail_provider(skill_settings_store, tool_permission_store)

    # 6. Skill 机制：扫描技能目录，装配 SkillToolFactory（依赖 guardrail + sandbox）
    init_skill_manager(
        settings.SKILLS_DIRS,
        enabled=settings.SKILLS_ENABLED,
        guardrail_provider=guardrail_provider,
        sandbox_provider=sandbox_provider,
        skill_script_timeout_seconds=settings.SKILL_SCRIPT_TIMEOUT_SECONDS,
    )

    # 7. Memory 机制：长期记忆存储 + 提取器 + 压缩器
    memory_store = ElasticsearchMemoryStore(
        es_url=settings.ES_URL,
        es_memory_index=settings.ES_MEMORY_INDEX,
        embedding_base_url=settings.EMBEDDING_BASE_URL,
        embedding_model=settings.EMBEDDING_MODEL,
        reranker_base_url=settings.RERANKER_BASE_URL,
        reranker_model=settings.RERANKER_MODEL,
        sensitive_filter_enabled=settings.MEMORY_SENSITIVE_FILTER_ENABLED,
        es_api_key=settings.ES_API_KEY,
        es_username=settings.ES_USERNAME,
        es_password=settings.ES_PASSWORD,
        es_verify_certs=settings.ES_VERIFY_CERTS,
    )
    await memory_store.setup()
    init_memory_manager(_build_memory_manager(memory_store))
    logger.info("MemoryManager 初始化完成")

    # 8. Lead Agent 会话持久化 checkpointer（连接生命周期绑定在本 async with 块内）
    async with AsyncPostgresSaver.from_conn_string(settings.DATABASE_URL) as checkpointer:
        await checkpointer.setup()
        init_checkpointer(checkpointer)
        logger.info("Checkpointer 初始化完成")

        logger.info("全部基础设施初始化完成，应用已就绪")
        yield

    # ── 关闭 ────────────────────────────────────────────────
    await memory_store.close()
    await aclose_emitter_redis()
    await aclose_middleware_redis()
    await close_database_pool()
    logger.info("所有资源已释放")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Agent Loop 工程：技能(skill)/记忆(memory)/工具(tool)机制 "
                "+ 权限控制(guardrail)/虚拟文件系统(workspace)/沙箱(sandbox) "
                "+ Lead Agent 编排层（委派工具、REST /chat 与 /conversations）。",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(EvalMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _error_body(code: int, msg: str) -> dict:
    """构造与 `src.common.response.ApiResponse` 结构一致的错误响应体。"""
    return {"code": code, "msg": msg, "data": None, "timestamp": int(time.time() * 1000)}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=_error_body(exc.status_code, str(exc.detail)))


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(status_code=422, content=_error_body(422, "参数校验失败"))


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("未捕获异常")
    return JSONResponse(status_code=500, content=_error_body(500, str(exc)))


app.include_router(health_router.router, tags=["Health"])
app.include_router(skill_router.router, tags=["Skills"])
app.include_router(memory_router.router, tags=["Memory"])
app.include_router(chat_router.router, tags=["Chat"])
app.include_router(conversation_router.router, tags=["Conversations"])
app.include_router(chat_ws.router, tags=["WebSocket"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        log_level=None,
    )
