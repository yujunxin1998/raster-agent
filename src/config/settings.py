"""系统配置：基于 pydantic-settings 的强类型配置类。

设计说明：
    - 全局仅通过 get_settings() 单例访问，禁止在业务代码中直接 os.getenv()，
      使配置读取入口收敛到一处，便于审计有哪些环境变量在起作用。
    - 字段全部大写、按功能分组注释，与原 diit-agent-server 项目的 env_utils.py
      保持同名同语义，降低迁移期间的心智负担；新增分组（WORKSPACE_*/SANDBOX_*/
      GUARDRAIL_*）对应本工程新增的三个基础设施模块。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv(override=True)


class SystemConfiguration(BaseSettings):
    """进程级配置项集合。

    每个字段对应一个环境变量，默认值与 diit-agent-server 保持一致，
    确保沿用相同的 .env 时行为不发生变化。
    """

    # ================ 服务基本配置 =============================
    APP_NAME: str = os.getenv("APP_NAME", "raster-agent-server")
    CURRENT_ENV: str = os.getenv("ENVIRONMENT", "development")
    APP_VERSION: str = os.getenv("APP_VERSION", "0.1.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8080"))
    # 浏览器能直接访问到本服务的地址（如 http://localhost:8080 或反代后的
    # https://xxx.com/agent-api），供拼装回复正文里的下载链接（`save_output_file`
    # 等）使用。留空时这些链接退化为相对路径——只有在前端和本服务同源部署
    # （反代把两者挂在同一个 origin 下）时相对路径才能正确解析，否则浏览器会
    # 把它解析成"当前页面（前端 SPA）自己的地址 + 这段路径"而不是本服务的地址，
    # 点击后打不开。
    PUBLIC_BASE_URL: str = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")
    CORS_ALLOW_ORIGIN_REGEX: str = os.getenv(
        "CORS_ALLOW_ORIGIN_REGEX", r"http://(localhost|127\.0\.0\.1)(:\d+)?$"
    )

    # ================ 大语言模型环境依赖 ========================
    API_KEY: str = os.getenv("OPENAI_API_KEY", os.getenv("API_KEY", ""))
    BASE_URL: str = os.getenv("OPENAI_BASE_URL", os.getenv("BASE_URL", ""))
    DEFAULT_MODEL: str = os.getenv("OPENAI_MODEL", os.getenv("DEFAULT_MODEL", ""))
    PROVIDER: str = os.getenv(
        "PROVIDER", "vllm" if os.getenv("OPENAI_BASE_URL") else "deepseek"
    )

    # ================ OpenAI-compatible / vLLM ==================
    # DeerFlow-style model configuration.  These variables are preferred when
    # present, while the legacy names above remain supported.
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "")
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "")
    OPENAI_MODEL_THINKING: str = os.getenv("OPENAI_MODEL_THINKING", "")

    # ================ 数据库（storage 层） ======================
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")

    # ================ 搜索工具 ==================================
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

    # ================ Elasticsearch ============================
    ES_URL: str = os.getenv("ES_URL", "")
    ES_API_KEY: str = os.getenv("ES_API_KEY", "")
    ES_USERNAME: str = os.getenv("ES_USERNAME", "")
    ES_PASSWORD: str = os.getenv("ES_PASSWORD", "")
    ES_VERIFY_CERTS: bool = os.getenv("ES_VERIFY_CERTS", "true").lower() == "true"
    ES_MEMORY_INDEX: str = os.getenv("ES_MEMORY_INDEX", "agent_memories")

    # ================ Embedding（Xinference BGE）===============
    EMBEDDING_BASE_URL: str = os.getenv("EMBEDDING_BASE_URL", "")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "bge-m3")

    # ================ Reranker（Xinference BGE）================
    RERANKER_BASE_URL: str = os.getenv("RERANKER_BASE_URL", "")
    RERANKER_MODEL: str = os.getenv("RERANKER_MODEL", "bge-reranker-v2-m3")

    # ================ 记忆管理行为参数 ============================
    # 压缩触发条件：type 决定看哪个指标，value 是对应阈值。
    #   messages: 消息条数超过 value（整数）时触发
    #   tokens:   近似 token 数（count_tokens_approximately）超过 value（整数）时触发
    #   fraction: 近似 token 数占 MEMORY_MODEL_MAX_INPUT_TOKENS 的比例超过
    #             value（0~1 小数）时触发
    MEMORY_COMPRESSION_TRIGGER_TYPE: str = os.getenv("MEMORY_COMPRESSION_TRIGGER_TYPE", "messages")
    MEMORY_COMPRESSION_TRIGGER_VALUE: float = float(os.getenv("MEMORY_COMPRESSION_TRIGGER_VALUE", "30"))
    # 仅 fraction 触发类型需要：自部署的 Qwen/vLLM 等模型没有 LangChain 的
    # profile 机制可以自动探测最大输入 token 数，只能显式配置。
    MEMORY_MODEL_MAX_INPUT_TOKENS: int = int(os.getenv("MEMORY_MODEL_MAX_INPUT_TOKENS", "32000"))
    MEMORY_KEEP_RECENT: int = int(os.getenv("MEMORY_KEEP_RECENT", "10"))
    MEMORY_RECALL_CANDIDATE_K: int = int(os.getenv("MEMORY_RECALL_CANDIDATE_K", "20"))
    MEMORY_MAX_RECALL: int = int(os.getenv("MEMORY_MAX_RECALL", "5"))
    MEMORY_IMPORTANCE_THRESHOLD: int = int(os.getenv("MEMORY_IMPORTANCE_THRESHOLD", "5"))
    MEMORY_DUPLICATE_SCORE_THRESHOLD: float = float(os.getenv("MEMORY_DUPLICATE_SCORE_THRESHOLD", "0.92"))
    MEMORY_CONFLICT_SCORE_THRESHOLD: float = float(os.getenv("MEMORY_CONFLICT_SCORE_THRESHOLD", "0.75"))
    MEMORY_LOW_CONFIDENCE_MARGIN: int = int(os.getenv("MEMORY_LOW_CONFIDENCE_MARGIN", "2"))
    MEMORY_MIN_RECALL_SCORE: float = float(os.getenv("MEMORY_MIN_RECALL_SCORE", "0.3"))
    MEMORY_MAX_CONTEXT_TOKENS: int = int(os.getenv("MEMORY_MAX_CONTEXT_TOKENS", "1000"))
    MEMORY_JOB_MAX_ATTEMPTS: int = int(os.getenv("MEMORY_JOB_MAX_ATTEMPTS", "3"))
    MEMORY_EXTRACT_MODEL: str = os.getenv("MEMORY_EXTRACT_MODEL", "")
    MEMORY_COMPRESS_MODEL: str = os.getenv("MEMORY_COMPRESS_MODEL", "")

    # ================ 记忆功能开关 ================================
    MEMORY_ENABLED: bool = os.getenv("MEMORY_ENABLED", "true").lower() == "true"
    MEMORY_AUTO_EXTRACT_ENABLED: bool = os.getenv("MEMORY_AUTO_EXTRACT_ENABLED", "true").lower() == "true"
    MEMORY_INJECTION_ENABLED: bool = os.getenv("MEMORY_INJECTION_ENABLED", "true").lower() == "true"
    MEMORY_TOOL_ENABLED: bool = os.getenv("MEMORY_TOOL_ENABLED", "true").lower() == "true"
    MEMORY_SENSITIVE_FILTER_ENABLED: bool = os.getenv("MEMORY_SENSITIVE_FILTER_ENABLED", "true").lower() == "true"
    # 用户画像/时间线（L1/L2）实时更新开关，跟 Facts 提取（L3，
    # MEMORY_AUTO_EXTRACT_ENABLED）各自独立的 LLM 调用，可单独关闭控制成本。
    MEMORY_PROFILE_UPDATE_ENABLED: bool = os.getenv("MEMORY_PROFILE_UPDATE_ENABLED", "true").lower() == "true"

    # ================ Skill 机制 ================================
    SKILLS_DIRS: str = os.getenv("SKILLS_DIRS", "skills/core,skills/public")
    SKILLS_ENABLED: bool = os.getenv("SKILLS_ENABLED", "true").lower() == "true"
    SKILL_SCRIPT_TIMEOUT_SECONDS: int = int(os.getenv("SKILL_SCRIPT_TIMEOUT_SECONDS", "60"))
    SKILL_SCRIPT_MAX_OUTPUT_BYTES: int = int(os.getenv("SKILL_SCRIPT_MAX_OUTPUT_BYTES", str(2 * 1024 * 1024)))
    SKILL_SCRIPT_MAX_MEMORY_MB: int = int(os.getenv("SKILL_SCRIPT_MAX_MEMORY_MB", "512"))

    # ================ Workspace（虚拟文件系统，新增） =============
    # 每个 conversation_id 对应一个隔离目录 {WORKSPACE_ROOT}/users/{user_id}/threads/{conversation_id}/，
    # 详见 src/agent_core/workspace。
    WORKSPACE_ROOT: str = os.getenv("WORKSPACE_ROOT", "./data/workspaces")

    # ================ Upload（会话附件上传，新增） =================
    UPLOAD_MAX_FILE_BYTES: int = int(os.getenv("UPLOAD_MAX_FILE_BYTES", str(20 * 1024 * 1024)))
    UPLOAD_ALLOWED_EXTENSIONS: str = os.getenv(
        "UPLOAD_ALLOWED_EXTENSIONS",
        ".txt,.md,.csv,.json,.pdf,.doc,.docx,.xls,.xlsx,.png,.jpg,.jpeg,.gif,.webp",
    )
    # 附件是否自动离线转换为 Markdown（.docx/.xlsx/.pdf，见
    # agent_core/ingestion/document_extractor.py），命名/语义对齐
    # bytedance/deer-flow 的 uploads.auto_convert_documents 配置项。
    # 关闭后附件只复制进 workspace/，模型拿到原始二进制文件，多数格式
    # read_file 会读出乱码——仅用于需要绕开转换开销/排查转换问题的场景。
    ATTACHMENT_AUTO_CONVERT_DOCUMENTS: bool = (
        os.getenv("ATTACHMENT_AUTO_CONVERT_DOCUMENTS", "true").lower() == "true"
    )

    # ================ Sandbox（沙箱执行，新增） ===================
    SANDBOX_PROVIDER: str = os.getenv("SANDBOX_PROVIDER", "local")
    SANDBOX_COMMAND_TIMEOUT_SECONDS: int = int(os.getenv("SANDBOX_COMMAND_TIMEOUT_SECONDS", "60"))
    SANDBOX_MAX_OUTPUT_BYTES: int = int(os.getenv("SANDBOX_MAX_OUTPUT_BYTES", str(2 * 1024 * 1024)))
    SANDBOX_MAX_MEMORY_MB: int = int(os.getenv("SANDBOX_MAX_MEMORY_MB", "512"))
    SANDBOX_MAX_PIDS: int = int(os.getenv("SANDBOX_MAX_PIDS", "64"))
    SANDBOX_NETWORK_ENABLED: bool = os.getenv("SANDBOX_NETWORK_ENABLED", "false").lower() == "true"
    # run_python/run_command 的 stdout/stderr 超过这个行数时，不再把完整内容塞进
    # 模型上下文——只给一份头尾预览，完整内容落盘到会话 workspace 里，模型需要时
    # 自己用 read_file 读取（用磁盘 IO 换 token/上下文压力）。
    SANDBOX_INLINE_OUTPUT_MAX_LINES: int = int(os.getenv("SANDBOX_INLINE_OUTPUT_MAX_LINES", "50"))
    # SANDBOX_PROVIDER=docker 时使用：默认镜像不含 Node.js，涉及 .js 脚本的技能
    # 需要换成自带 node 的镜像；PROJECT_ROOT 用于技能脚本绝对路径在容器内的翻译，
    # 默认取本文件所在项目的根目录，一般不需要覆盖。
    DOCKER_SANDBOX_IMAGE: str = os.getenv("DOCKER_SANDBOX_IMAGE", "python:3.11-slim")
    PROJECT_ROOT: str = os.getenv("PROJECT_ROOT", str(Path(__file__).resolve().parents[2]))

    # ================ 后台维护任务（新增） =========================
    # 复用同一个 asyncio 周期循环承载 staleness 复核 + checkpoint 孤儿清理，
    # 不引入 APScheduler 等独立调度框架。
    MAINTENANCE_INTERVAL_HOURS: int = int(os.getenv("MAINTENANCE_INTERVAL_HOURS", "24"))
    MEMORY_STALENESS_ENABLED: bool = os.getenv("MEMORY_STALENESS_ENABLED", "true").lower() == "true"
    MEMORY_STALENESS_MAX_AGE_DAYS: int = int(os.getenv("MEMORY_STALENESS_MAX_AGE_DAYS", "90"))
    MEMORY_STALENESS_IMPORTANCE_THRESHOLD: int = int(os.getenv("MEMORY_STALENESS_IMPORTANCE_THRESHOLD", "3"))
    CHECKPOINT_CLEANUP_ENABLED: bool = os.getenv("CHECKPOINT_CLEANUP_ENABLED", "true").lower() == "true"

    # ================ Guardrail（权限控制，新增） =================
    GUARDRAIL_ENABLED: bool = os.getenv("GUARDRAIL_ENABLED", "true").lower() == "true"

    # ================ 数据库自然语言查询（database 委派，新增） =====
    DB_QUERY_API_URL: str = os.getenv("DB_QUERY_API_URL", "http://ask-db-service:8084/api/ask-db/unified-query")
    DB_QUERY_API_TIMEOUT: float = float(os.getenv("DB_QUERY_API_TIMEOUT", "60.0"))
    ECHART_API_URL: str = os.getenv("ECHART_API_URL", "http://192.168.80.60:8888/api/ask-data/echarts/generate")
    ECHART_API_TIMEOUT: int = int(os.getenv("ECHART_API_TIMEOUT", "30"))

    # ================ RAG 检索（search_knowledge_base 技能，新增） ===
    RAG_API_URL: str = os.getenv("RAG_API_URL", "")
    RAG_DB_ID: str = os.getenv("RAG_DB_ID", "0")
    RAG_TOP_K: int = int(os.getenv("RAG_TOP_K", "1024"))
    RAG_PAGE: int = int(os.getenv("RAG_PAGE", "1"))
    RAG_PAGE_SIZE: int = int(os.getenv("RAG_PAGE_SIZE", "10"))
    RAG_SCORE_THRESHOLD: float = float(os.getenv("RAG_SCORE_THRESHOLD", "0.0"))
    RAG_VECTOR_SIMILARITY_WEIGHT: float = float(os.getenv("RAG_VECTOR_SIMILARITY_WEIGHT", "0.8"))
    RAG_QUERY_REWRITE_ENABLED: bool = os.getenv("RAG_QUERY_REWRITE_ENABLED", "true").lower() == "true"
    RAG_QUERY_REWRITE_MAX_TOKENS: int = int(os.getenv("RAG_QUERY_REWRITE_MAX_TOKENS", "256"))
    RAG_QUERY_REWRITE_TIMEOUT: float = float(os.getenv("RAG_QUERY_REWRITE_TIMEOUT", "15.0"))
    RAG_QUERY_REWRITE_HISTORY_TURNS: int = int(os.getenv("RAG_QUERY_REWRITE_HISTORY_TURNS", "3"))

    # ================ Redis（Eval 遥测，新增） =====================
    # 留空即禁用 Eval 遥测（emitter/middleware 内部静默降级，不影响主流程）。
    REDIS_URL: str = os.getenv("REDIS_URL", "")


@lru_cache(maxsize=1)
def get_settings() -> "SystemConfiguration":
    """返回全局唯一的配置单例。

    使用 lru_cache 而非模块级变量，是为了让"首次调用时才真正实例化"这件事
    显式可见（第一次调用即完成一次 .env 校验），同时避免测试场景下重复
    实例化触发多次 pydantic-settings 校验开销。

    Returns:
        进程内唯一的 SystemConfiguration 实例。
    """
    return SystemConfiguration()
