"""MCPToolProvider/MCPConnectionManager 单元/故障测试（设计文档 4.4 节 + 第八节）。"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.agent_core.tools.registry.mcp_provider import MCPConnectionManager, MCPServerConfig, parse_mcp_servers
from src.agent_core.tools.registry.tool_registry import ToolRegistry


def test_parse_mcp_servers_empty_setting_returns_empty_list() -> None:
    assert parse_mcp_servers("[]") == []
    assert parse_mcp_servers("") == []
    assert parse_mcp_servers("   ") == []


def test_parse_mcp_servers_invalid_json_returns_empty_list() -> None:
    assert parse_mcp_servers("not json") == []


def test_parse_mcp_servers_parses_stdio_and_streamable_http() -> None:
    raw = (
        '[{"id": "github", "transport": "stdio", "command": "npx", "args": ["-y", "github-mcp"]},'
        ' {"id": "search", "transport": "streamable_http", "url": "https://example.com/mcp"}]'
    )

    configs = parse_mcp_servers(raw)

    assert [c.id for c in configs] == ["github", "search"]
    assert configs[0].to_connection() == {
        "transport": "stdio", "command": "npx", "args": ["-y", "github-mcp"],
    }
    assert configs[1].to_connection() == {"transport": "streamable_http", "url": "https://example.com/mcp"}


def test_parse_mcp_servers_skips_entry_with_duplicate_id() -> None:
    raw = (
        '[{"id": "github", "transport": "stdio", "command": "npx"},'
        ' {"id": "github", "transport": "stdio", "command": "npx2"}]'
    )

    configs = parse_mcp_servers(raw)

    assert len(configs) == 1
    assert configs[0].command == "npx"


def test_parse_mcp_servers_skips_entry_missing_required_field() -> None:
    raw = '[{"id": "broken"}]'  # 缺少 transport

    assert parse_mcp_servers(raw) == []


def test_to_connection_raises_when_stdio_missing_command() -> None:
    cfg = MCPServerConfig(id="x", transport="stdio")
    with pytest.raises(ValueError, match="command"):
        cfg.to_connection()


def test_empty_server_configs_produce_noop_manager() -> None:
    """`MCP_SERVERS` 为空数组（默认值）时是纯空操作，不影响其余来源
    （设计文档 4.4 节"配置"一段）。"""
    manager = MCPConnectionManager(ToolRegistry(), [], poll_interval_seconds=60, reconnect_backoff_max_seconds=60)

    manager.start()  # 不应该抛异常，也不应该起任何后台任务

    assert manager.status == {}


def _fake_tool(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, description=f"{name} description")


async def test_poll_loop_publishes_tools_then_evicts_on_disconnect() -> None:
    """MCP Server 连接中途断开：验证该 Server 的工具从快照里被摘掉
    （断线立即摘除，不留死工具，设计文档 4.4 节"断线处理"）。"""
    tool_registry = ToolRegistry()
    cfg = MCPServerConfig(id="flaky", transport="stdio", command="npx")
    manager = MCPConnectionManager(tool_registry, [cfg], poll_interval_seconds=0.01, reconnect_backoff_max_seconds=1)

    call_count = 0

    async def fake_get_tools(*, server_name: str):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return [_fake_tool("list_issues")]
        raise ConnectionError("server unreachable")

    manager._client = SimpleNamespace(get_tools=fake_get_tools)

    task = asyncio.create_task(manager._poll_loop(cfg))
    try:
        # 第一次轮询成功：工具应该出现在快照里。
        for _ in range(200):
            if "list_issues" in tool_registry.current_snapshot().by_model_name:
                break
            await asyncio.sleep(0.01)
        assert "list_issues" in tool_registry.current_snapshot().by_model_name
        assert manager.status["flaky"] == "connected"

        # 第二次轮询失败：工具必须被摘除，不是继续留着一份调用必超时的死工具。
        for _ in range(200):
            if "list_issues" not in tool_registry.current_snapshot().by_model_name:
                break
            await asyncio.sleep(0.01)
        assert "list_issues" not in tool_registry.current_snapshot().by_model_name
        assert manager.status["flaky"] == "reconnecting"
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
