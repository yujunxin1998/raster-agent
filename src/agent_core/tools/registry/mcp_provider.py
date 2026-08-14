"""MCPToolProvider + MCPConnectionManager：真实的 MCP 一等来源。

对应设计文档《工具注册中心与热重载设计.md》4.4 节。技术选型
`langchain-mcp-adapters`（LangChain 官方维护，包装 `mcp` Python SDK）而不是
像 `custom_tool_converter.py` 那样手写一遍 JSON Schema -> Pydantic 转换——
MCP 场景有现成客户端库可用，没必要重复造轮子。

`MultiServerMCPClient.get_tools()` 按库自身设计"为每次工具调用现开一个新
session"（stdio 场景等价于现拉起一个子进程），因此本模块不持有长期存活的
`ClientSession`：`MCPConnectionManager` 只是周期性调用 `get_tools()`，既
验证了 Server 可达性，又直接拿到可执行的 `BaseTool`，比手写
`ClientSession` 生命周期管理简单可靠得多。这是相对设计文档草稿里"持有
长连接 + 订阅 notifications/tools/list_changed"的一处务实简化，用轮询
替代推送，详见文档"与原方案的偏差"说明。
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Literal

from langchain_mcp_adapters.client import MultiServerMCPClient
from loguru import logger

from src.agent_core.tools.registry.tool_definition import ToolDefinition
from src.agent_core.tools.registry.tool_registry import ToolRegistry

_STDIO = "stdio"
_STREAMABLE_HTTP = "streamable_http"


@dataclass(frozen=True)
class MCPServerConfig:
    """一个 MCP Server 的连接配置，解析自 `settings.MCP_SERVERS`。"""

    id: str
    transport: Literal["stdio", "streamable_http"]
    command: str | None = None
    args: tuple[str, ...] = ()
    env: dict[str, str] | None = None
    url: str | None = None
    headers: dict[str, str] | None = None

    def to_connection(self) -> dict:
        """转换成 `langchain_mcp_adapters` 期望的 Connection 字典（TypedDict）。

        Raises:
            ValueError: transport 对应的必填字段缺失，或 transport 本身不受支持。
        """
        if self.transport == _STDIO:
            if not self.command:
                raise ValueError(f"MCP Server {self.id!r} transport=stdio 缺少 command")
            connection: dict = {"transport": _STDIO, "command": self.command, "args": list(self.args)}
            if self.env:
                connection["env"] = self.env
            return connection
        if self.transport == _STREAMABLE_HTTP:
            if not self.url:
                raise ValueError(f"MCP Server {self.id!r} transport=streamable_http 缺少 url")
            connection = {"transport": _STREAMABLE_HTTP, "url": self.url}
            if self.headers:
                connection["headers"] = self.headers
            return connection
        raise ValueError(f"MCP Server {self.id!r} 不支持的 transport={self.transport!r}")


def parse_mcp_servers(mcp_servers_setting: str) -> list[MCPServerConfig]:
    """解析 `MCP_SERVERS` 配置字符串（JSON 数组）。

    单条配置格式错误时跳过并记 error，不阻断其余 Server 的解析——与
    `SkillLoader` 一致的容错风格。

    Args:
        mcp_servers_setting: `settings.MCP_SERVERS` 原始 JSON 字符串。

    Returns:
        解析成功的 `MCPServerConfig` 列表；配置为空/非法 JSON 时返回空列表。
    """
    raw = (mcp_servers_setting or "").strip()
    if not raw:
        return []
    try:
        entries = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error(f"[MCPToolProvider] MCP_SERVERS 不是合法 JSON，已忽略: {exc}")
        return []
    if not isinstance(entries, list):
        logger.error("[MCPToolProvider] MCP_SERVERS 必须是 JSON 数组，已忽略")
        return []

    configs: list[MCPServerConfig] = []
    seen_ids: set[str] = set()
    for entry in entries:
        try:
            if not isinstance(entry, dict) or not entry.get("id") or not entry.get("transport"):
                raise ValueError(f"缺少必填字段 id/transport: {entry!r}")
            server_id = str(entry["id"])
            if server_id in seen_ids:
                raise ValueError(f"重复的 id: {server_id!r}")
            seen_ids.add(server_id)
            configs.append(
                MCPServerConfig(
                    id=server_id,
                    transport=entry["transport"],
                    command=entry.get("command"),
                    args=tuple(entry.get("args") or ()),
                    env=entry.get("env"),
                    url=entry.get("url"),
                    headers=entry.get("headers"),
                )
            )
        except (ValueError, TypeError) as exc:
            logger.error(f"[MCPToolProvider] 跳过非法 MCP Server 配置: {exc}")
    return configs


class MCPConnectionManager:
    """管理配置的全部 MCP Server：周期轮询发现工具、发布进 `ToolRegistry`、
    断线时摘除、指数退避重连（设计文档 4.4 节）。
    """

    def __init__(
        self,
        tool_registry: ToolRegistry,
        server_configs: list[MCPServerConfig],
        poll_interval_seconds: int,
        reconnect_backoff_max_seconds: int,
    ) -> None:
        """初始化管理器（不发起任何连接，`start()` 才真正启动轮询）。

        Args:
            tool_registry: 全局 ToolRegistry 单例，轮询结果发布到这里。
            server_configs: 已解析的 MCP Server 配置列表，可为空。
            poll_interval_seconds: 健康 Server 两次刷新之间的间隔。
            reconnect_backoff_max_seconds: 不可达 Server 的重试退避上限。
        """
        self._tool_registry = tool_registry
        self._server_configs = server_configs
        self._poll_interval_seconds = poll_interval_seconds
        self._reconnect_backoff_max_seconds = reconnect_backoff_max_seconds
        self._client = (
            MultiServerMCPClient({cfg.id: cfg.to_connection() for cfg in server_configs})
            if server_configs
            else None
        )
        self._status: dict[str, str] = {cfg.id: "pending" for cfg in server_configs}
        self._tasks: list[asyncio.Task] = []

    @property
    def status(self) -> dict[str, str]:
        """各 Server 当前状态快照（"pending"/"connected"/"reconnecting"），供排障查看。"""
        return dict(self._status)

    def start(self) -> None:
        """为每个配置的 Server 各起一个后台轮询任务。配置为空时是空操作。"""
        for cfg in self._server_configs:
            task = asyncio.create_task(self._poll_loop(cfg), name=f"mcp-poll-{cfg.id}")
            self._tasks.append(task)
        if self._server_configs:
            logger.info(f"[MCPConnectionManager] 已启动 {len(self._server_configs)} 个 MCP Server 轮询任务")

    async def stop(self) -> None:
        """取消全部轮询任务（应用关闭时调用）。"""
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

    async def _poll_loop(self, cfg: MCPServerConfig) -> None:
        backoff = 1.0
        while True:
            try:
                tools = await self._client.get_tools(server_name=cfg.id)
                await self._publish_tools(cfg, tools)
                self._status[cfg.id] = "connected"
                backoff = 1.0
                await asyncio.sleep(self._poll_interval_seconds)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._status[cfg.id] = "reconnecting"
                logger.warning(f"[MCPConnectionManager] {cfg.id} 不可达，{backoff:.0f}s 后重试: {exc}")
                # 断线立即摘除该 Server 的工具，不留"看起来存在、实际调用
                # 必超时"的死工具（设计文档 4.4 节"断线处理"）。
                await self._evict(cfg)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, self._reconnect_backoff_max_seconds)

    async def _publish_tools(self, cfg: MCPServerConfig, tools: list) -> None:
        source_id = f"mcp:{cfg.id}"
        permissions_scope = f"mcp:{cfg.id}"
        definitions = [
            ToolDefinition(
                canonical_name=f"mcp.{cfg.id}.{tool.name}",
                model_name=tool.name,
                description=tool.description or "",
                source_type="mcp",
                source_id=source_id,
                scope="application",
                permissions_key=tool.name,
                permissions_scope=permissions_scope,
                build_tool=(lambda t=tool: t),
            )
            for tool in tools
        ]
        try:
            await self._tool_registry.publish(source_id=source_id, definitions=definitions)
        except ValueError as exc:
            # 与其它来源命名冲突：保留旧快照，记 error，不让一个配错的 MCP
            # Server 拖垮其它来源（与设计文档 5.4 节"发布与失败处理"同一哲学）。
            logger.error(f"[MCPConnectionManager] {cfg.id} 工具发布被拒绝（命名冲突）: {exc}")

    async def _evict(self, cfg: MCPServerConfig) -> None:
        await self._tool_registry.publish(source_id=f"mcp:{cfg.id}", definitions=[])


_manager: MCPConnectionManager | None = None


async def init_mcp_connection_manager(
    tool_registry: ToolRegistry,
    mcp_servers_setting: str,
    poll_interval_seconds: int,
    reconnect_backoff_max_seconds: int,
) -> MCPConnectionManager:
    """应用启动时调用一次：解析配置、构造管理器、启动全部轮询任务。

    `mcp_servers_setting` 为空数组（默认值）时构造出的管理器不持有任何
    Server、`start()` 是空操作——这就是"兼容 MCP"和"现在没有真实 MCP
    Server 也不影响其余部分"能同时成立的原因。
    """
    global _manager
    configs = parse_mcp_servers(mcp_servers_setting)
    _manager = MCPConnectionManager(tool_registry, configs, poll_interval_seconds, reconnect_backoff_max_seconds)
    _manager.start()
    logger.info(f"[MCPConnectionManager] 初始化完成，共 {len(configs)} 个 MCP Server")
    return _manager


async def shutdown_mcp_connection_manager() -> None:
    """应用关闭时调用一次，取消全部轮询任务。未初始化时是空操作。"""
    if _manager is not None:
        await _manager.stop()


def get_mcp_connection_manager() -> MCPConnectionManager:
    """返回全局唯一的 MCPConnectionManager 实例。

    Raises:
        RuntimeError: init_mcp_connection_manager() 尚未被调用。
    """
    if _manager is None:
        raise RuntimeError("MCPConnectionManager 尚未初始化，请确认应用已完成启动")
    return _manager
