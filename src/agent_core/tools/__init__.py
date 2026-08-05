"""Tool 机制模块：后端工具 + 前端工具（client-side tool）双通道。

对应设计文档「三、3.3 工具机制」：后端工具在服务端进程内直接执行
（`web_search`/`save_memory`/`recall_memory`），前端工具通过 WebSocket
转发给客户端执行（`custom_tool_converter.py`）。`tool_filter_registry`
标记不需要展示/持久化的内部工具。
"""
from src.agent_core.tools.custom_tool_converter import CustomToolConverter, FrontendTool
from src.agent_core.tools.memory_tools import recall_memory, save_memory
from src.agent_core.tools.tool_filter import ToolFilterRegistry, tool_filter_registry
from src.agent_core.tools.web_search_tool import TavilySearchClient, web_search

__all__ = [
    "CustomToolConverter",
    "FrontendTool",
    "save_memory",
    "recall_memory",
    "ToolFilterRegistry",
    "tool_filter_registry",
    "TavilySearchClient",
    "web_search",
]
