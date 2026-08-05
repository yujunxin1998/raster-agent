"""挂到 Lead Agent 上的通用沙箱执行工具：write_file / read_file / run_python / run_command。

对应设计文档 5.3 节"未来的代码执行类工具统一经过 Sandbox 接口执行"——
`SandboxMiddleware` 当时只是提前打了地基（每轮开始前拿一个 Sandbox 实例存成
自己的私有属性），本文件是第一批真正的消费者，但没有复用那个缓存实例：中间件
实例的私有属性对工具协程不可见，改用 `skill_tool_factory.py` 已经验证过的
方式——按 `conversation_id`/`user_id` 直接 `sandbox_provider.acquire()` 现取
一个。这不是权宜之计：`LocalSandboxProvider.acquire()` 本身就是无状态的（只是
构造一个绑定到同一会话工作区目录的新 `LocalSandbox` 对象，不持有子进程/连接等
需要"预留"的资源），每次调用成本可以忽略，跟"复用一个缓存实例"在 LocalSandbox
这一期没有实质差别。

每个工具调用都会经过 `GuardrailMiddleware`（工具级权限，可用
`user_tool_permissions` 表按用户禁用）、`LoopDetectionMiddleware`（连续 3 次
完全相同的调用会被短路）、`ToolErrorHandlingMiddleware`（异常统一包裹成 error
ToolMessage 而不是中断整轮对话，因此本文件不需要自己 try/except 业务异常）
——这是把这几个工具直接挂在 Lead Agent 自己的 `base_tools`（而不是新开一个
`delegate_to_code_agent` 委派子 Agent）的核心原因：`sub_agent_factory.py`
现造的子 Agent 不挂载任何中间件（见其 `build_delegate_tool._invoke`
实现），如果把这几个工具塞进子 Agent 的工具集，子 Agent 内部"写→跑→改→再跑"
的循环会完全跑在死循环检测和权限校验之外。

安全边界：`LocalSandbox`（见其类文档）不做容器级隔离，只有 cwd 限定在会话
工作区 + 环境变量清洗 + 命令超时/输出截断/POSIX 下的内存上限，跟现有技能脚本
执行的信任级别完全一致——这几个工具不会引入比技能脚本更强的攻击面，但也没有
更强；如果之后要接不受信任的任意代码，需要先做 5.3 节提到的容器化沙箱。
"""
from __future__ import annotations

import sys
from typing import Awaitable, Callable

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool

from src.agent_core.sandbox import get_sandbox_provider
from src.agent_core.sandbox.sandbox import CommandResult, Sandbox

_SANDBOX_UNAVAILABLE_MESSAGE = "当前调用缺少会话上下文（thread_id），无法使用沙箱工具。"


def _resolve_ids(config: RunnableConfig) -> tuple[str | None, str | None]:
    configurable = config.get("configurable", {}) if config else {}
    return configurable.get("thread_id"), configurable.get("user_id")


def _format_command_result(result: CommandResult) -> str:
    lines = [f"status={result.status.value} return_code={result.return_code}"]
    if result.truncated:
        lines.append("[警告] stdout 超过输出上限，已截断")
    stdout = result.stdout_text().strip()
    stderr = result.stderr_text().strip()
    if stdout:
        lines.append(f"--- stdout ---\n{stdout}")
    if stderr:
        lines.append(f"--- stderr ---\n{stderr}")
    return "\n".join(lines)


async def _with_sandbox(config: RunnableConfig, fn: Callable[[Sandbox], Awaitable[str]]) -> str:
    """acquire → 执行 → release 的公共骨架。"""
    conversation_id, user_id = _resolve_ids(config)
    if not conversation_id:
        return _SANDBOX_UNAVAILABLE_MESSAGE

    provider = get_sandbox_provider()
    sandbox = await provider.acquire(conversation_id, user_id)
    try:
        return await fn(sandbox)
    finally:
        await provider.release(sandbox)


@tool
async def write_file(path: str, content: str, config: RunnableConfig, overwrite: bool = True) -> str:
    """向本次会话的沙箱工作区写入一个文件（相对路径）。文件已存在时默认覆盖，overwrite=False 时若已存在会报错。"""

    async def _run(sandbox: Sandbox) -> str:
        await sandbox.write_file(path, content, overwrite=overwrite)
        return f"已写入 {path}（{len(content)} 字符）"

    return await _with_sandbox(config, _run)


@tool
async def read_file(path: str, config: RunnableConfig) -> str:
    """读取本次会话沙箱工作区内某个文件（相对路径）的文本内容。"""

    async def _run(sandbox: Sandbox) -> str:
        return await sandbox.read_file(path)

    return await _with_sandbox(config, _run)


@tool
async def run_python(
    config: RunnableConfig,
    code: str | None = None,
    file_path: str | None = None,
    timeout: float | None = None,
) -> str:
    """在沙箱里执行 Python：传 code 直接跑一段代码，或传 file_path 运行工作区内已写好的脚本，二者必须且只能提供一个。"""
    if bool(code) == bool(file_path):
        return "code 和 file_path 必须且只能提供一个"

    async def _run(sandbox: Sandbox) -> str:
        command = [sys.executable, "-c", code] if code else [sys.executable, file_path]
        result = await sandbox.execute_command(command, timeout=timeout)
        return _format_command_result(result)

    return await _with_sandbox(config, _run)


@tool
async def run_command(command: list[str], config: RunnableConfig, timeout: float | None = None) -> str:
    """在沙箱工作区内执行一条命令（如 pip install、git、node 等）。command 是可执行文件+参数的列表，不支持管道/重定向等 shell 语法。"""
    if not command:
        return "command 不能为空"

    async def _run(sandbox: Sandbox) -> str:
        result = await sandbox.execute_command(command, timeout=timeout)
        return _format_command_result(result)

    return await _with_sandbox(config, _run)
