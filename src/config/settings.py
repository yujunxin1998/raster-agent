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
    MEMORY_COMPRESSION_THRESHOLD: int = int(os.getenv("MEMORY_COMPRESSION_THRESHOLD", "30"))
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

    # ================ Sandbox（沙箱执行，新增） ===================
    SANDBOX_PROVIDER: str = os.getenv("SANDBOX_PROVIDER", "local")
    SANDBOX_COMMAND_TIMEOUT_SECONDS: int = int(os.getenv("SANDBOX_COMMAND_TIMEOUT_SECONDS", "60"))
    SANDBOX_MAX_OUTPUT_BYTES: int = int(os.getenv("SANDBOX_MAX_OUTPUT_BYTES", str(2 * 1024 * 1024)))
    SANDBOX_MAX_MEMORY_MB: int = int(os.getenv("SANDBOX_MAX_MEMORY_MB", "512"))

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
