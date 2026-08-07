"""集中定义的枚举常量。

遵循"禁止魔法值"原则：业务代码中不应出现裸字符串/裸数字来表达状态、分类、
动作这类有限取值集合，一律通过本模块的枚举引用。枚举继承 str，使其可以
直接参与 Pydantic 模型字段、JSON 序列化、Elasticsearch 文档写入，无需
额外的 .value 转换（但拼 SQL/发 HTTP 请求体时仍建议显式 .value，避免
依赖 str 隐式转换带来的可读性问题）。
"""
from __future__ import annotations

from enum import Enum


class SkillCategory(str, Enum):
    """技能分类，决定挂载到哪个 Agent（对应原 supervisor 路由中的子 Agent）。"""

    GENERAL = "general"
    RAG = "rag"
    WEB_SEARCH = "web_search"
    DATABASE = "database"
    TOOL = "tool"

    @classmethod
    def default(cls) -> "SkillCategory":
        """SKILL.md frontmatter 未声明或声明了非法 category 时的兜底值。"""
        return cls.GENERAL


class MemoryType(str, Enum):
    """长期记忆的语义分类（Facts 层的五分类，对齐三层记忆架构设计）。"""

    PREFERENCE = "preference"
    KNOWLEDGE = "knowledge"
    CONTEXT = "context"
    BEHAVIOR = "behavior"
    GOAL = "goal"
    SUMMARY = "summary"  # 压缩摘要专用（MemoryCompressor 写入），不属于 Facts 五分类


class MemoryStatus(str, Enum):
    """长期记忆的生命周期状态。"""

    ACTIVE = "active"
    PENDING = "pending"
    ARCHIVED = "archived"
    EXPIRED = "expired"  # 仅在读取时按 expires_at 派生展示，不物理写入存储


class MemorySource(str, Enum):
    """记忆的产生来源，用于审计追溯。"""

    EXTRACTOR = "extractor"
    TOOL = "tool"
    API = "api"
    COMPRESSOR = "compressor"


class MemoryAuditAction(str, Enum):
    """记忆审计日志的动作类型。"""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RECALL = "recall"
    INJECT = "inject"
    COMPRESS = "compress"
    SAVE_REJECTED = "save_rejected"
    UPDATE_REJECTED = "update_rejected"


class MemoryJobStatus(str, Enum):
    """记忆后处理任务（提取/压缩）的执行状态。"""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class GuardrailDecisionType(str, Enum):
    """权限校验结果类型。"""

    ALLOW = "allow"
    DENY = "deny"


class WorkspaceDirectory(str, Enum):
    """每个会话隔离目录下的三个固定子目录。"""

    WORKSPACE = "workspace"
    UPLOADS = "uploads"
    OUTPUTS = "outputs"


class SandboxCommandStatus(str, Enum):
    """沙箱命令执行结果状态。"""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    OUTPUT_TRUNCATED = "output_truncated"
