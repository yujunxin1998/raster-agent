"""前端工具的请求级投影：不进全局 ToolRegistry（设计文档 4.2 节）。

复用现有 `CustomToolConverter` 做真正的协议解析/BaseTool 转换，本模块只是
在此基础上多包一层 `ToolDefinition` 元数据，供 `resolve_tools()` 按 4.3 节
的冲突策略与 application 级定义合并。前端工具本来就是 per-message 的
（`chat_ws.py` 里 `tool_list` 挂在每条 `content` 上），不是常驻 session，
因此没有必要像 GPT 原方案那样维护一个"注册-注销"的 Session Overlay
Registry，现算现用即可。
"""
from __future__ import annotations

from loguru import logger

from src.agent_core.tools.custom_tool_converter import CustomToolConverter
from src.agent_core.tools.registry.tool_definition import ToolDefinition
from src.common.exceptions import ToolConversionError

_PERMISSIONS_SCOPE = "frontend"


def discover_frontend_tools(
    *,
    tool_list: list[dict] | None,
    manager,
    client_id: str,
    conversation_id: str,
) -> list[ToolDefinition]:
    """把本次消息携带的 `tool_list` 转成 request-scope 的 `ToolDefinition` 列表。

    Args:
        tool_list: 前端传入的原始工具定义列表，可为空。
        manager: 承载 WebSocket 连接、负责"发指令-等结果"往返的连接管理器
            （`ConnectionManager`，避免循环 import 用 Any 语义传入）。
        client_id: 目标客户端连接 ID。
        conversation_id: 归属会话 ID。

    Returns:
        本次请求的前端工具定义列表；`tool_list` 解析失败时记 error 日志并
        返回空列表，不中断当前对话——与 `CustomToolConverter.merge_tools`
        原有的容错语义保持一致。
    """
    if not tool_list:
        return []

    source_id = f"frontend:{client_id}:{conversation_id}"
    try:
        frontend_tools = CustomToolConverter.parse_tool_list(tool_list)
    except ToolConversionError as exc:
        logger.error(f"[FrontendToolProvider] 工具解析失败: {exc}")
        return []

    definitions: list[ToolDefinition] = []
    for frontend_tool in frontend_tools:
        base_tool = CustomToolConverter.convert_to_basetool(frontend_tool, manager, client_id, conversation_id)
        definitions.append(
            ToolDefinition(
                canonical_name=f"frontend.{frontend_tool.name}",
                model_name=frontend_tool.name,
                description=frontend_tool.description or f"前端工具: {frontend_tool.name}",
                source_type="frontend",
                source_id=source_id,
                scope="request",
                permissions_key=frontend_tool.name,
                permissions_scope=_PERMISSIONS_SCOPE,
                build_tool=(lambda t=base_tool: t),
            )
        )
        logger.info(f"[FrontendToolProvider] 注册工具: {frontend_tool.name}")
    return definitions
