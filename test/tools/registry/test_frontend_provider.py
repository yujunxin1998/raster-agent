"""discover_frontend_tools() 单元测试（设计文档 4.2 节 FrontendToolProvider）。"""
from __future__ import annotations

from src.agent_core.tools.registry.frontend_provider import discover_frontend_tools


def test_discover_frontend_tools_returns_empty_list_when_tool_list_is_none() -> None:
    assert discover_frontend_tools(tool_list=None, manager=None, client_id="c1", conversation_id="conv1") == []


def test_discover_frontend_tools_returns_empty_list_when_tool_list_is_empty() -> None:
    assert discover_frontend_tools(tool_list=[], manager=None, client_id="c1", conversation_id="conv1") == []


def test_discover_frontend_tools_builds_request_scope_definitions() -> None:
    tool_list = [
        {
            "name": "show_map",
            "description": "在地图上高亮一个区域",
            "inputSchema": {"type": "object", "properties": {"region": {"type": "string"}}, "required": ["region"]},
        }
    ]

    definitions = discover_frontend_tools(
        tool_list=tool_list, manager=object(), client_id="client-1", conversation_id="conv-1",
    )

    assert len(definitions) == 1
    definition = definitions[0]
    assert definition.model_name == "show_map"
    assert definition.canonical_name == "frontend.show_map"
    assert definition.source_type == "frontend"
    assert definition.scope == "request"
    assert definition.source_id == "frontend:client-1:conv-1"
    assert definition.permissions_scope == "frontend"
    built = definition.build_tool()
    assert built.name == "show_map"


def test_discover_frontend_tools_returns_empty_list_on_malformed_tool_list() -> None:
    """`tool_list` 解析失败时记 error 并返回空列表，不中断当前对话
    （与 `CustomToolConverter.merge_tools` 原有的容错语义保持一致）。"""
    malformed_tool_list = [{"name": "broken"}]  # 缺少必填的 inputSchema

    definitions = discover_frontend_tools(
        tool_list=malformed_tool_list, manager=object(), client_id="c1", conversation_id="conv1",
    )

    assert definitions == []
