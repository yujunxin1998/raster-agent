"""前端自定义工具（client-side tool）转换机制。

原样迁移自 `src/core/tools/custom_tool_converter.py`：把前端通过 WebSocket
声明的 MCP JSON Schema 工具列表转换为 LangChain `BaseTool`，执行时经
WebSocket 往返客户端获取结果。协议、超时、异常处理行为均未改动，
确保 `diit-agent-web` 的前端工具机制无需任何改动即可对接。
"""
from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from langchain_core.tools import BaseTool, StructuredTool
from loguru import logger
from pydantic import BaseModel, Field, create_model

from src.common.exceptions import ToolConversionError

_FRONTEND_TOOL_TIMEOUT_MESSAGE = "前端工具响应超时（>15s）"
_MCP_CONTEXT_FIELD_NAME = "ctx"  # MCP 协议保留字段，不暴露给 LLM

# JSON Schema type -> Python type 映射
_TYPE_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class FrontendTool(BaseModel):
    """前端传入的单个工具定义（MCP 标准格式）。

    字段名刻意采用与命名规范不一致的 mixedCase（`inputSchema`/`outputSchema`），
    因为这两个字段名由 MCP JSON Schema 协议规定、由前端直接按此拼写传入，
    改成 snake_case 会导致 Pydantic 反序列化失败——协议兼容性优先于内部命名统一。
    """

    name: str
    description: Optional[str] = ""
    inputSchema: dict[str, Any]  # noqa: N815 - MCP 协议字段名，不可重命名
    outputSchema: Optional[dict[str, Any]] = None  # noqa: N815 - 同上
    meta: Optional[dict[str, Any]] = None


class CustomToolConverter:
    """将前端传入的 JSON Schema 工具列表转换为 LangChain BaseTool。"""

    @staticmethod
    def parse_tool_list(tool_list: list[dict]) -> list[FrontendTool]:
        """解析并校验前端工具列表。

        Args:
            tool_list: 前端传入的原始工具定义列表。

        Returns:
            解析后的 FrontendTool 列表。

        Raises:
            ToolConversionError: tool_list 不是列表，或某个工具缺少必填字段。
        """
        if not isinstance(tool_list, list):
            raise ToolConversionError("tool_list 必须是列表")

        tools = []
        for item in tool_list:
            if "name" not in item or "inputSchema" not in item:
                raise ToolConversionError(f"工具缺少必填字段 name/inputSchema: {item}")
            if "properties" not in item.get("inputSchema", {}):
                raise ToolConversionError(f"工具 {item['name']} 的 inputSchema 缺少 properties")
            tools.append(FrontendTool(**item))
        return tools

    @staticmethod
    def _schema_to_pydantic(tool_name: str, schema: dict) -> type[BaseModel]:
        """将 JSON Schema 的 properties 动态生成 Pydantic 参数模型。"""
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        fields: dict[str, Any] = {}

        for field_name, field_info in properties.items():
            if field_name == _MCP_CONTEXT_FIELD_NAME:
                continue
            python_type = _TYPE_MAP.get(field_info.get("type", "string"), str)
            description = field_info.get("description", "")
            if field_name in required:
                fields[field_name] = (python_type, Field(..., description=description))
            else:
                fields[field_name] = (Optional[python_type], Field(None, description=description))

        model_name = f"ToolInput_{tool_name}_{uuid.uuid4().hex[:6]}"
        return create_model(model_name, **fields)

    @staticmethod
    def convert_to_basetool(
        tool: FrontendTool,
        manager,          # ConnectionManager，避免循环 import 用 Any
        client_id: str,
        conversation_id: str,
    ) -> BaseTool:
        """将 FrontendTool 包装成 LangChain StructuredTool。

        工具被 Agent 调用时，通过 WebSocket 发送指令给前端执行，然后异步
        等待前端返回结果。

        Args:
            tool: 前端工具定义。
            manager: 承载 WebSocket 连接、负责"发指令-等结果"往返的连接管理器。
            client_id: 目标客户端连接 ID。
            conversation_id: 归属会话 ID。

        Returns:
            对应的 StructuredTool 实例。
        """
        args_schema = CustomToolConverter._schema_to_pydantic(tool.name, tool.inputSchema)

        async def tool_execute(**kwargs) -> str:
            logger.info(f"[FrontendTool] 调用 name={tool.name} args={kwargs}")
            try:
                response = await manager.send_tool_call_and_wait_response(
                    client_id=client_id, tool_name=tool.name, tool_args=kwargs,
                    conversation_id=conversation_id,
                )
                if not response.get("isError", False):
                    result = {"status": "success", "tool_name": tool.name, "result": response.get("result", response)}
                else:
                    result = {"status": "error", "tool_name": tool.name, "message": response.get("message", "前端工具执行失败")}
            except TimeoutError:
                result = {"status": "error", "tool_name": tool.name, "message": _FRONTEND_TOOL_TIMEOUT_MESSAGE}
            except Exception as exc:
                result = {"status": "error", "tool_name": tool.name, "message": str(exc)}

            return json.dumps(result, ensure_ascii=False)

        return StructuredTool(
            name=tool.name,
            description=tool.description or f"前端工具: {tool.name}",
            args_schema=args_schema,
            coroutine=tool_execute,
        )

    @staticmethod
    def merge_tools(
        original_tools: list[BaseTool],
        tool_list: list[dict] | None,
        manager,
        client_id: str,
        conversation_id: str,
    ) -> list[BaseTool]:
        """将前端工具列表与服务端原有工具合并，同名工具以前端定义为准（覆盖）。

        Args:
            original_tools: 服务端已有的工具列表。
            tool_list: 前端传入的原始工具定义列表，可为空。
            manager: 连接管理器。
            client_id: 目标客户端连接 ID。
            conversation_id: 归属会话 ID。

        Returns:
            合并后的工具列表；tool_list 解析失败时记 error 日志并原样返回
            original_tools，不中断当前对话。
        """
        if not tool_list:
            return original_tools

        tool_map = {tool.name: tool for tool in original_tools}

        try:
            frontend_tools = CustomToolConverter.parse_tool_list(tool_list)
            for frontend_tool in frontend_tools:
                base_tool = CustomToolConverter.convert_to_basetool(frontend_tool, manager, client_id, conversation_id)
                tool_map[base_tool.name] = base_tool
                logger.info(f"[FrontendTool] 注册工具: {frontend_tool.name}")
        except Exception as exc:
            logger.error(f"[FrontendTool] 工具解析失败: {exc}")

        return list(tool_map.values())
