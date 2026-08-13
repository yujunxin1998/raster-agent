"""FastAPI 应用入口。

组装顺序（lifespan 内）：数据库连接池 → 各 Store 建表 → workspace/sandbox/
guardrail 三个基础设施 → skill 机制 → memory 机制 → 会话/消息存储 →
Lead Agent checkpointer → checkpoint 孤儿清理 → 后台维护循环 → 内部工具过滤
名单。这个顺序对应各模块间的真实依赖关系（skill 机制依赖 guardrail + sandbox，
guardrail 依赖 storage 层的两个 Store，均需要在此之前完成初始化；Lead Agent
依赖前面全部基础设施都已就绪；checkpoint 清理依赖 checkpointer 已完成初始化）。

`/chat`（REST，非流式）、`/conversations`、`/ws/chat`（WebSocket 流式 + 前端
工具双通道回环）三组接口对应设计文档 4.1 节"Lead Agent + 委派工具"的落地。
配置了 `REDIS_URL` 时 `EvalMiddleware` 会把请求级观测数据写入 Redis Streams
（`eval:trace`/`eval:tool`/`eval:http`），未配置时全程静默降级。

后台维护循环（`_maintenance_loop`）不引入 APScheduler 等独立调度框架，延续
本工程"进程内简单 asyncio 任务"的风格，承载两件周期性维护工作：记忆 staleness
复核（`MemoryStalenessReviewer`）和 LangGraph checkpoint 孤儿数据清理
（`CheckpointCleanup.cleanup_orphans`），详见设计文档第七节路线图第三期。
"""
from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager

if sys.platform == "win32":
    # psycopg（AsyncPostgresSaver 底层驱动，见 lifespan 第 8 步）不支持
    # Windows 默认的 ProactorEventLoop，asyncio 连接时直接抛
    # `psycopg.InterfaceError`。必须在本进程创建/绑定任何 event loop 之前
    # （包括 uvicorn 内部创建的、以及 reload=True 时 spawn 出的子进程重新
    # import 本模块时）完成策略切换，因此放在模块最顶部，不放在
    # `if __name__ == "__main__":` 块里——后者在 reload 子进程里不保证会
    # 早于 uvicorn 自己的 loop 创建时机执行到。非 Windows 平台上这是空操作
    # （Linux/Docker 部署不受影响）。
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from loguru import logger
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from src.agent_core.agents.checkpointer import init_checkpointer
from src.agent_core.eval.emitter import aclose_emitter_redis
from src.agent_core.eval.http_middleware import EvalMiddleware, aclose_middleware_redis
from src.agent_core.guardrail import init_guardrail_provider
from src.agent_core.memory import (
    ElasticsearchMemoryStore,
    MemoryCompressor,
    MemoryExtractor,
    MemoryManager,
    MemoryStalenessReviewer,
    UserProfileUpdater,
    init_memory_manager,
)
from src.agent_core.sandbox import get_sandbox_provider, init_docker_sandbox_provider, init_local_sandbox_provider
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
from src.storage.checkpoint_cleanup import get_checkpoint_cleanup, init_checkpoint_cleanup
from src.storage.conversation_store import init_conversation_store
from src.storage.database import close_database_pool, init_database_pool
from src.storage.file_store import init_file_store
from src.storage.memory_audit_store import init_memory_audit_store
from src.storage.memory_jobs_store import init_memory_jobs_store
from src.storage.message_store import init_message_store
from src.storage.skill_settings_store import init_skill_settings_store
from src.storage.tool_permission_store import init_tool_permission_store
from src.storage.user_profile_store import get_user_profile_store, init_user_profile_store

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
        trigger_type=settings.MEMORY_COMPRESSION_TRIGGER_TYPE,
        trigger_value=settings.MEMORY_COMPRESSION_TRIGGER_VALUE,
        default_keep_recent=settings.MEMORY_KEEP_RECENT,
        model_max_input_tokens=settings.MEMORY_MODEL_MAX_INPUT_TOKENS,
    )
    # 用户画像/时间线（L1/L2）更新跟 Facts 提取是同一类"看这一轮对话、产出
    # 结构化内容"的任务，复用 MEMORY_EXTRACT_MODEL，不新增模型配置项。
    profile_updater = UserProfileUpdater(
        model_name=settings.MEMORY_EXTRACT_MODEL or settings.DEFAULT_MODEL,
        provider=settings.PROVIDER,
        api_key=settings.API_KEY,
        base_url=settings.BASE_URL,
    )
    return MemoryManager(
        store=memory_store,
        extractor=extractor,
        compressor=compressor,
        profile_store=get_user_profile_store(),
        profile_updater=profile_updater,
        memory_enabled=settings.MEMORY_ENABLED,
        injection_enabled=settings.MEMORY_INJECTION_ENABLED,
        auto_extract_enabled=settings.MEMORY_AUTO_EXTRACT_ENABLED,
        profile_update_enabled=settings.MEMORY_PROFILE_UPDATE_ENABLED,
        recall_candidate_k=settings.MEMORY_RECALL_CANDIDATE_K,
        max_recall=settings.MEMORY_MAX_RECALL,
        min_recall_score=settings.MEMORY_MIN_RECALL_SCORE,
        max_context_tokens=settings.MEMORY_MAX_CONTEXT_TOKENS,
        importance_threshold=settings.MEMORY_IMPORTANCE_THRESHOLD,
        sensitive_filter_enabled=settings.MEMORY_SENSITIVE_FILTER_ENABLED,
    )


async def _maintenance_loop(staleness_reviewer: MemoryStalenessReviewer) -> None:
    """周期性后台维护：记忆 staleness 复核 + checkpoint 孤儿数据清理。

    循环体在 `sleep` 之前，启动后立即先跑一轮，不需要等一整个 interval 才第一次
    生效；两个子任务各自 try/except，其中一个失败不影响另一个继续执行。
    """
    interval_seconds = settings.MAINTENANCE_INTERVAL_HOURS * 3600
    while True:
        if settings.MEMORY_STALENESS_ENABLED:
            await staleness_reviewer.run_once()

        if settings.CHECKPOINT_CLEANUP_ENABLED:
            try:
                cleaned = await get_checkpoint_cleanup().cleanup_orphans()
                if cleaned:
                    logger.info(f"[Maintenance] 清理 {cleaned} 个孤儿 checkpoint thread")
            except Exception as exc:
                logger.warning(f"[Maintenance] checkpoint 孤儿清理失败: {exc}")

        await asyncio.sleep(interval_seconds)


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
    await init_file_store(pool)
    await init_user_profile_store(pool)

    # 3. 虚拟工作区（按会话隔离目录）
    init_thread_workspace_manager(settings.WORKSPACE_ROOT)
    workspace_manager = get_thread_workspace_manager()

    # 4. 沙箱提供者（按 SANDBOX_PROVIDER 选择 Local 或 Docker 实现）
    if settings.SANDBOX_PROVIDER == "docker":
        await init_docker_sandbox_provider(
            workspace_manager,
            image=settings.DOCKER_SANDBOX_IMAGE,
            project_root=settings.PROJECT_ROOT,
            default_timeout_seconds=settings.SANDBOX_COMMAND_TIMEOUT_SECONDS,
            max_output_bytes=settings.SANDBOX_MAX_OUTPUT_BYTES,
            max_memory_mb=settings.SANDBOX_MAX_MEMORY_MB,
            max_pids=settings.SANDBOX_MAX_PIDS,
            network_enabled=settings.SANDBOX_NETWORK_ENABLED,
        )
    else:
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
    #
    # 不用 AsyncPostgresSaver.from_conn_string()：它内部是一条裸 AsyncConnection，
    # 没有任何重连机制——一旦这条连接因空闲超时/网络抖动/数据库重启被服务端或
    # 中间网络设备断开，psycopg 会把它标记为 [BAD]，之后每一次 checkpointer 调用
    # 都会立刻抛 `psycopg.OperationalError: the connection is closed`，且永远
    # 不会自愈，只能重启进程（实际发生过，见 WS 流式响应异常日志）。改用
    # `psycopg_pool.AsyncConnectionPool`：`AsyncPostgresSaver` 原生支持传入连接池
    # （见 `langgraph.checkpoint.postgres._ainternal.Conn` 的类型定义），连接池
    # 自带健康检查（`check=AsyncConnectionPool.check_connection`），借出连接前
    # 会先探活，坏连接自动丢弃重建，不再需要整个进程重启才能恢复。
    checkpoint_pool = AsyncConnectionPool(
        conninfo=settings.DATABASE_URL,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        check=AsyncConnectionPool.check_connection,
        open=False,
    )
    async with checkpoint_pool:
        checkpointer = AsyncPostgresSaver(conn=checkpoint_pool)
        await checkpointer.setup()
        init_checkpointer(checkpointer)
        logger.info("Checkpointer 初始化完成")

        # 9. checkpoint 孤儿数据清理（依赖 checkpointer 已完成初始化）
        init_checkpoint_cleanup(pool, checkpointer)

        # 10. 后台维护循环：记忆 staleness 复核 + checkpoint 孤儿数据清理
        staleness_reviewer = MemoryStalenessReviewer(
            memory_store,
            max_age_days=settings.MEMORY_STALENESS_MAX_AGE_DAYS,
            low_importance_threshold=settings.MEMORY_STALENESS_IMPORTANCE_THRESHOLD,
        )
        maintenance_task = asyncio.create_task(_maintenance_loop(staleness_reviewer))

        logger.info("全部基础设施初始化完成，应用已就绪")
        yield

        maintenance_task.cancel()
        try:
            await maintenance_task
        except asyncio.CancelledError:
            pass

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
    allow_origin_regex=settings.CORS_ALLOW_ORIGIN_REGEX,
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


if settings.CURRENT_ENV == "development":
    # 本地前后端联调用的手工测试页面（上传/发送/下载全流程），生产环境不挂载。
    app.mount("/manual_test", StaticFiles(directory="manual_test"), name="manual_test")

app.include_router(health_router.router, tags=["Health"])
app.include_router(skill_router.router, tags=["Skills"])
app.include_router(memory_router.router, tags=["Memory"])
app.include_router(chat_router.router, tags=["Chat"])
app.include_router(conversation_router.router, tags=["Conversations"])
app.include_router(chat_ws.router, tags=["WebSocket"])


if __name__ == "__main__":
    import uvicorn
    from pathlib import Path

    # 不排除 WORKSPACE_ROOT 的话，沙箱工具（write_file/run_python 等，见
    # agent_core/tools/sandbox_tool.py）往里面写的每一个 .py 文件都会被
    # uvicorn 的 reload watcher 当成"源码变更"，立刻重启进程、强制断开正在
    # 进行中的 WebSocket 连接——现象是"工具调用明明成功了，但整个回复卡在
    # 那不动"，因为连接在 reload 那一刻直接被服务端切断，不是业务逻辑卡死。
    # uvicorn 的 `FileFilter` 只用字面量 `Path(e) in path.parents` 做目录级
    # 排除判断（见 `uvicorn/supervisors/watchfilesreload.py`），不会自动
    # `resolve()`；传相对路径会因为跟 watchfiles 回调的绝对路径对不上而
    # 排除失效，必须传解析后的绝对路径。
    workspace_root_exclude = str(Path(settings.WORKSPACE_ROOT).resolve())

    uvicorn.run(
        "src.main:app",
        host="0.0.0.0",
        port=settings.APP_PORT,
        reload=True,
        reload_excludes=[workspace_root_exclude],
        log_level=None,
    )
