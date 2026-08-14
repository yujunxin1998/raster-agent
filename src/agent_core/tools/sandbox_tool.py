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

`run_python`/`run_command` 的 stdout/stderr 超过 `SANDBOX_INLINE_OUTPUT_MAX_LINES`
（默认 50 行）时不再整段塞进模型上下文——只给一份头尾预览，完整内容落盘到
`tool_output/` 目录，模型需要细节时自己用 `read_file` 按需读取（用磁盘 IO 换
token/上下文压力，见 `_inline_or_offload`）。这和 `SANDBOX_MAX_OUTPUT_BYTES`
是两层不同的控制：字节级截断是硬上限（超过部分真的丢了，永远拿不回来），
行数级落盘是软处理（内容完整保留，只是不默认塞进上下文）。
"""
from __future__ import annotations

import json
import sys
import uuid
from pathlib import PurePosixPath
from typing import Awaitable, Callable

from langgraph.prebuilt import ToolRuntime
from langchain_core.tools import tool

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.sandbox import get_sandbox_provider
from src.agent_core.sandbox.sandbox import CommandResult, Sandbox
from src.common.constants import WorkspaceDirectory
from src.config.settings import get_settings

_SANDBOX_UNAVAILABLE_MESSAGE = "当前调用缺少会话上下文（thread_id），无法使用沙箱工具。"
_TOOL_OUTPUT_DIR = "tool_output"

# 产物后缀是这几种时，返回结果里额外带 <image> 标签（前端 ImageBubble 识别
# 这个标签渲染成图片预览+下载卡片，见 useChatStore.js::_extractImageBlocks）；
# 非图片产物（PDF/Excel/zip 等）没有预览的意义，维持原来的纯文本下载链接。
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg"}


def _resolve_ids(runtime: ToolRuntime[AgentRuntimeContext]) -> tuple[str | None, str | None]:
    return runtime.context.conversation_id, runtime.context.user_id


async def _inline_or_offload(sandbox: Sandbox, stream_name: str, text: str) -> str:
    """长输出落盘 + 头尾预览，短输出原样内联返回。

    用磁盘 IO 换模型上下文/token 压力：一次跑飞的循环打印几千行不应该把上下文
    直接撑爆，但完整内容也不能真的丢掉——模型确实需要看细节时，可以自己用
    `read_file` 按需读取落盘的完整内容，比"要么全塞进上下文、要么整体截断丢弃"
    这两个极端都更合理。

    Args:
        sandbox: 当前会话的沙箱实例，落盘用。
        stream_name: "stdout" 或 "stderr"，用于落盘文件名和提示文案。
        text: 该输出流的完整文本（已经过 `SANDBOX_MAX_OUTPUT_BYTES` 字节级截断，
            这里只处理行数）。

    Returns:
        未超过行数阈值时原样返回；超过时返回"头尾预览 + 落盘路径提示"文本。
    """
    max_lines = get_settings().SANDBOX_INLINE_OUTPUT_MAX_LINES
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return text

    half = max_lines // 2
    head, tail = lines[:half], lines[-half:]
    omitted = len(lines) - len(head) - len(tail)
    path = f"{_TOOL_OUTPUT_DIR}/{uuid.uuid4().hex[:8]}_{stream_name}.log"
    await sandbox.write_file(path, text)

    preview = "\n".join(head) + f"\n...(中间省略 {omitted} 行)...\n" + "\n".join(tail)
    return (
        f"[{stream_name} 共 {len(lines)} 行，超过 {max_lines} 行预览上限，"
        f"仅展示头尾，完整内容已保存到 {path}，需要时用 read_file 读取]\n{preview}"
    )


async def _format_command_result(sandbox: Sandbox, result: CommandResult) -> str:
    lines = [f"status={result.status.value} return_code={result.return_code}"]
    if result.truncated:
        lines.append("[警告] stdout 超过输出上限，已截断")
    stdout = result.stdout_text().strip()
    stderr = result.stderr_text().strip()
    if stdout:
        lines.append(f"--- stdout ---\n{await _inline_or_offload(sandbox, 'stdout', stdout)}")
    if stderr:
        lines.append(f"--- stderr ---\n{await _inline_or_offload(sandbox, 'stderr', stderr)}")
    return "\n".join(lines)


async def _with_sandbox(
    runtime: ToolRuntime[AgentRuntimeContext], fn: Callable[[Sandbox], Awaitable[str]],
) -> str:
    """acquire → 执行 → release 的公共骨架。"""
    conversation_id, user_id = _resolve_ids(runtime)
    if not conversation_id:
        return _SANDBOX_UNAVAILABLE_MESSAGE

    provider = get_sandbox_provider()
    sandbox = await provider.acquire(conversation_id, user_id)
    try:
        return await fn(sandbox)
    finally:
        await provider.release(sandbox)


@tool
async def write_file(
    path: str, content: str, runtime: ToolRuntime[AgentRuntimeContext], overwrite: bool = True,
) -> str:
    """向本次会话的沙箱工作区写入一个文件（相对路径）。文件已存在时默认覆盖，overwrite=False 时若已存在会报错。"""

    async def _run(sandbox: Sandbox) -> str:
        await sandbox.write_file(path, content, overwrite=overwrite)
        return f"已写入 {path}（{len(content)} 字符）"

    return await _with_sandbox(runtime, _run)


@tool
async def read_file(path: str, runtime: ToolRuntime[AgentRuntimeContext]) -> str:
    """读取本次会话沙箱工作区内某个文件（相对路径）的文本内容。"""

    async def _run(sandbox: Sandbox) -> str:
        return await sandbox.read_file(path)

    return await _with_sandbox(runtime, _run)


@tool
async def save_output_file(path: str, content: str, runtime: ToolRuntime[AgentRuntimeContext]) -> str:
    """把最终产物（图表、生成的文档等，不是中间过程文件）保存到本次会话的产物目录，返回可下载链接。

    产物是图片格式（.png/.jpg/.jpeg/.gif/.webp/.svg）时，返回结果里会带一个
    `<image>` 标签——回复正文里原样保留这个标签（不要拆开转述成普通 Markdown
    链接），前端会识别并渲染成图片预览卡片。
    """

    async def _run(sandbox: Sandbox) -> str:
        await sandbox.write_file(path, content, directory=WorkspaceDirectory.OUTPUTS)
        conversation_id, _ = _resolve_ids(runtime)
        base_url = get_settings().PUBLIC_BASE_URL
        link = f"{base_url}/conversations/{conversation_id}/outputs/{path}"

        extension = PurePosixPath(path).suffix.lower()
        if extension in _IMAGE_EXTENSIONS:
            image_tag = json.dumps({"url": link, "title": PurePosixPath(path).name}, ensure_ascii=False)
            return f"已保存产物 {path}\n<image>{image_tag}</image>"
        return f"已保存产物 {path}，下载链接：{link}"

    return await _with_sandbox(runtime, _run)


@tool
async def run_python(
    runtime: ToolRuntime[AgentRuntimeContext],
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
        return await _format_command_result(sandbox, result)

    return await _with_sandbox(runtime, _run)


@tool
async def run_command(
    command: list[str], runtime: ToolRuntime[AgentRuntimeContext], timeout: float | None = None,
) -> str:
    """在沙箱工作区内执行一条命令（如 pip install、git、node 等）。command 是可执行文件+参数的列表，不支持管道/重定向等 shell 语法。"""
    if not command:
        return "command 不能为空"

    async def _run(sandbox: Sandbox) -> str:
        result = await sandbox.execute_command(command, timeout=timeout)
        return await _format_command_result(sandbox, result)

    return await _with_sandbox(runtime, _run)
