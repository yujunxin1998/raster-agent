"""提示词模板值对象。

原样迁移自 diit-agent-server 的 `src/core/prompts/registry.py::PromptTemplate`：
一条提示词的轻量实体，只携带名称、正文、来源路径三个字段。是否含有
`{variable}` 占位符（如 `rag_query_rewrite.md`）由正文内容本身决定，
模板对象不区分"静态模板"和"待渲染模板"两种类型——统一交给
`PromptFactory.render()` 处理，没有占位符的模板 `.format()` 是无操作的。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    """一条提示词模板。

    Attributes:
        name: 模板标识，全大写，与文件名对应（如 TITLE_GENERATION、
            RAG_QUERY_REWRITE）。
        content: 模板正文，可能包含 `{variable}` 形式的占位符。
        source: 相对项目根目录的来源路径，便于排查"这条提示词到底改的哪个文件"。
    """

    name: str
    content: str
    source: str
