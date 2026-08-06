"""挂到 Lead Agent 上的沙箱工具（write_file/read_file/run_python/run_command）单元测试。

只验证工具自身的编排逻辑（缺 thread_id 时的兜底、acquire/release 配对、
run_python 的 code/file_path 互斥校验、CommandResult 格式化），不依赖真实
LocalSandbox 子进程执行。
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from src.agent_core.sandbox.sandbox import CommandResult
from src.agent_core.tools.sandbox_tool import read_file, run_command, run_python, save_output_file, write_file
from src.common.constants import SandboxCommandStatus, WorkspaceDirectory


def _config(thread_id: str | None = "c1", user_id: str | None = "u1") -> dict:
    configurable: dict = {}
    if thread_id is not None:
        configurable["thread_id"] = thread_id
    if user_id is not None:
        configurable["user_id"] = user_id
    return {"configurable": configurable}


def _fake_provider(sandbox: MagicMock) -> MagicMock:
    provider = MagicMock()
    provider.acquire = AsyncMock(return_value=sandbox)
    provider.release = AsyncMock()
    return provider


async def test_write_file_missing_thread_id_skips_sandbox_acquire() -> None:
    with patch("src.agent_core.tools.sandbox_tool.get_sandbox_provider") as get_provider:
        result = await write_file.coroutine(path="a.py", content="print(1)", config=_config(thread_id=None))

    get_provider.assert_not_called()
    assert "thread_id" in result


async def test_write_file_acquires_and_releases_sandbox() -> None:
    sandbox = MagicMock()
    sandbox.write_file = AsyncMock()
    provider = _fake_provider(sandbox)

    with patch("src.agent_core.tools.sandbox_tool.get_sandbox_provider", return_value=provider):
        result = await write_file.coroutine(path="a.py", content="print(1)", config=_config())

    provider.acquire.assert_awaited_once_with("c1", "u1")
    sandbox.write_file.assert_awaited_once_with("a.py", "print(1)", overwrite=True)
    provider.release.assert_awaited_once_with(sandbox)
    assert "a.py" in result


async def test_read_file_releases_sandbox_even_on_exception() -> None:
    sandbox = MagicMock()
    sandbox.read_file = AsyncMock(side_effect=FileNotFoundError("文件不存在: a.py"))
    provider = _fake_provider(sandbox)

    with patch("src.agent_core.tools.sandbox_tool.get_sandbox_provider", return_value=provider):
        try:
            await read_file.coroutine(path="a.py", config=_config())
            raised = False
        except FileNotFoundError:
            raised = True

    assert raised
    provider.release.assert_awaited_once_with(sandbox)


async def test_save_output_file_writes_to_outputs_directory_and_returns_link() -> None:
    sandbox = MagicMock()
    sandbox.write_file = AsyncMock()
    provider = _fake_provider(sandbox)

    with patch("src.agent_core.tools.sandbox_tool.get_sandbox_provider", return_value=provider):
        result = await save_output_file.coroutine(path="report.md", content="# 结果", config=_config())

    sandbox.write_file.assert_awaited_once_with("report.md", "# 结果", directory=WorkspaceDirectory.OUTPUTS)
    assert "/conversations/c1/outputs/report.md" in result


async def test_run_python_rejects_when_both_code_and_file_path_given() -> None:
    with patch("src.agent_core.tools.sandbox_tool.get_sandbox_provider") as get_provider:
        result = await run_python.coroutine(code="print(1)", file_path="a.py", config=_config())

    get_provider.assert_not_called()
    assert "只能提供一个" in result


async def test_run_python_rejects_when_neither_code_nor_file_path_given() -> None:
    result = await run_python.coroutine(config=_config())
    assert "只能提供一个" in result


async def test_run_python_with_code_uses_dash_c_command() -> None:
    sandbox = MagicMock()
    sandbox.execute_command = AsyncMock(
        return_value=CommandResult(status=SandboxCommandStatus.SUCCESS, return_code=0, stdout=b"1\n")
    )
    provider = _fake_provider(sandbox)

    with patch("src.agent_core.tools.sandbox_tool.get_sandbox_provider", return_value=provider), \
         patch("src.agent_core.tools.sandbox_tool.sys.executable", "python3"):
        result = await run_python.coroutine(code="print(1)", config=_config())

    sandbox.execute_command.assert_awaited_once_with(["python3", "-c", "print(1)"], timeout=None)
    assert "status=success" in result
    assert "1" in result


async def test_run_python_with_file_path_runs_interpreter_on_file() -> None:
    sandbox = MagicMock()
    sandbox.execute_command = AsyncMock(
        return_value=CommandResult(status=SandboxCommandStatus.FAILED, return_code=1, stderr=b"boom")
    )
    provider = _fake_provider(sandbox)

    with patch("src.agent_core.tools.sandbox_tool.get_sandbox_provider", return_value=provider), \
         patch("src.agent_core.tools.sandbox_tool.sys.executable", "python3"):
        result = await run_python.coroutine(file_path="quicksort.py", config=_config())

    sandbox.execute_command.assert_awaited_once_with(["python3", "quicksort.py"], timeout=None)
    assert "status=failed" in result
    assert "boom" in result


async def test_run_command_rejects_empty_command() -> None:
    with patch("src.agent_core.tools.sandbox_tool.get_sandbox_provider") as get_provider:
        result = await run_command.coroutine(command=[], config=_config())

    get_provider.assert_not_called()
    assert "不能为空" in result


async def test_run_command_reports_truncated_output() -> None:
    sandbox = MagicMock()
    sandbox.execute_command = AsyncMock(
        return_value=CommandResult(
            status=SandboxCommandStatus.OUTPUT_TRUNCATED, return_code=0, stdout=b"x" * 10, truncated=True
        )
    )
    provider = _fake_provider(sandbox)

    with patch("src.agent_core.tools.sandbox_tool.get_sandbox_provider", return_value=provider):
        result = await run_command.coroutine(command=["echo", "hi"], config=_config())

    assert "已截断" in result
