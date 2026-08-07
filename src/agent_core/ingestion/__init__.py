"""附件文本摄取模块：把办公文档格式离线转换为纯文本，供沙箱工具消费。"""
from src.agent_core.ingestion.document_extractor import (
    ExtractionResult,
    extract_text,
    is_extractable,
    is_known_unsupported,
)

__all__ = [
    "ExtractionResult",
    "extract_text",
    "is_extractable",
    "is_known_unsupported",
]
