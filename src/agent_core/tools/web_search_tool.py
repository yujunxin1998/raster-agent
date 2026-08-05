"""网络搜索工具，基于 Tavily。

原样迁移自 `src/core/tools/search_tool.py`，把原来的"模块级单例客户端 +
裸函数"改造为 `TavilySearchClient` 类（延迟初始化、职责单一：只负责调用
Tavily API 并格式化结果），`web_search` 仍然是一个 `@tool` 装饰的薄函数，
委托给该类的模块级单例——LangChain 的 `@tool` 装饰器要求被装饰对象是
普通函数，因此工具入口保留函数形态，具体实现下沉到类里，兼顾"面向对象
封装"与"LangChain 工具协议"两方面的要求。
"""
from __future__ import annotations

from langchain_core.tools import tool
from loguru import logger
from tavily import TavilyClient

from src.common.exceptions import ConfigurationError
from src.config.settings import get_settings

_MAX_RESULTS = 5


class TavilySearchClient:
    """对 Tavily 搜索 API 的薄封装，负责调用与结果格式化。"""

    def __init__(self, api_key: str) -> None:
        """初始化搜索客户端。

        Args:
            api_key: Tavily API Key。

        Raises:
            ConfigurationError: api_key 为空。
        """
        if not api_key:
            raise ConfigurationError("TAVILY_API_KEY 未配置，请在 .env 中设置")
        self._client = TavilyClient(api_key=api_key)

    def search(self, query: str) -> str:
        """执行一次搜索并把结果格式化为易读文本。

        Args:
            query: 搜索语句。

        Returns:
            格式化后的搜索结果文本；调用失败时返回错误说明文本，不抛出异常。
        """
        try:
            results = self._client.search(
                query=query, max_results=_MAX_RESULTS,
                include_answer=True, include_raw_content=False,
            )
        except Exception as exc:
            logger.error(f"[TavilySearchClient] 搜索失败: {exc}")
            return f"搜索失败：{exc}"

        return self._format_results(results)

    def _format_results(self, results: dict) -> str:
        answer = results.get("answer", "")
        sources = results.get("results", [])

        lines: list[str] = []
        if answer:
            lines.append(f"【摘要】{answer}\n")

        lines.append("【来源】")
        for index, item in enumerate(sources, 1):
            title = item.get("title", "无标题")
            url = item.get("url", "")
            content = item.get("content", "").strip()
            lines.append(f"{index}. {title}\n   链接：{url}\n   摘要：{content[:300]}")

        return "\n".join(lines)


_client: TavilySearchClient | None = None


def _get_client() -> TavilySearchClient:
    """延迟初始化模块级单例，避免未配置 TAVILY_API_KEY 时在导入阶段就报错。"""
    global _client
    if _client is None:
        _client = TavilySearchClient(get_settings().TAVILY_API_KEY)
    return _client


@tool
def web_search(query: str) -> str:
    """搜索互联网获取最新信息，适用于需要查询实时数据、新闻、当前事件的问题。"""
    logger.info(f"[web_search] query={query!r}")
    try:
        client = _get_client()
    except ConfigurationError as exc:
        return str(exc)
    return client.search(query)
