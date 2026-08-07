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
    settings = MagicMock(PUBLIC_BASE_URL="")

    with patch("src.agent_core.tools.sandbox_tool.get_sandbox_provider", return_value=provider), patch(
        "src.agent_core.tools.sandbox_tool.get_settings", return_value=settings
    ):
        result = await save_output_file.coroutine(path="report.md", content="# 结果", config=_config())

    sandbox.write_file.assert_awaited_once_with("report.md", "# 结果", directory=WorkspaceDirectory.OUTPUTS)
    assert "/conversations/c1/outputs/report.md" in result


async def test_save_output_file_wraps_image_extension_in_image_tag() -> None:
    """图片后缀返回 `<image>` 标签而不是纯文本下载链接，供前端渲染成预览卡片。"""
    sandbox = MagicMock()
    sandbox.write_file = AsyncMock()
    provider = _fake_provider(sandbox)
    settings = MagicMock(PUBLIC_BASE_URL="")

    with patch("src.agent_core.tools.sandbox_tool.get_sandbox_provider", return_value=provider), patch(
        "src.agent_core.tools.sandbox_tool.get_settings", return_value=settings
    ):
        result = await save_output_file.coroutine(path="chart.png", content="fake-bytes", config=_config())

    assert "<image>" in result and "</image>" in result
    assert "/conversations/c1/outputs/chart.png" in result
    assert "下载链接" not in result


async def test_save_output_file_non_image_extension_keeps_plain_link() -> None:
    sandbox = MagicMock()
    sandbox.write_file = AsyncMock()
    provider = _fake_provider(sandbox)
    settings = MagicMock(PUBLIC_BASE_URL="")

    with patch("src.agent_core.tools.sandbox_tool.get_sandbox_provider", return_value=provider), patch(
        "src.agent_core.tools.sandbox_tool.get_settings", return_value=settings
    ):
        result = await save_output_file.coroutine(path="report.pdf", content="fake-bytes", config=_config())

    assert "<image>" not in result
    assert "下载链接" in result


async def test_save_output_file_prefixes_public_base_url_when_configured() -> None:
    """`PUBLIC_BASE_URL` 配置后返回绝对链接，避免前端 SPA 把相对路径解析成自己的地址。"""
    sandbox = MagicMock()
    sandbox.write_file = AsyncMock()
    provider = _fake_provider(sandbox)
    settings = MagicMock(PUBLIC_BASE_URL="http://localhost:8080")

    with patch("src.agent_core.tools.sandbox_tool.get_sandbox_provider", return_value=provider), patch(
        "src.agent_core.tools.sandbox_tool.get_settings", return_value=settings
    ):
        result = await save_output_file.coroutine(path="report.md", content="# 结果", config=_config())

    assert "http://localhost:8080/conversations/c1/outputs/report.md" in result


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


async def test_short_stdout_stays_inline_without_disk_write() -> None:
    sandbox = MagicMock()
    stdout = "\n".join(f"line{i}" for i in range(10)).encode()
    sandbox.execute_command = AsyncMock(
        return_value=CommandResult(status=SandboxCommandStatus.SUCCESS, return_code=0, stdout=stdout)
    )
    sandbox.write_file = AsyncMock()
    provider = _fake_provider(sandbox)

    with patch("src.agent_core.tools.sandbox_tool.get_sandbox_provider", return_value=provider):
        result = await run_command.coroutine(command=["echo", "hi"], config=_config())

    sandbox.write_file.assert_not_called()
    assert "line0" in result and "line9" in result
    assert "tool_output" not in result


async def test_long_stdout_gets_offloaded_to_disk_with_head_tail_preview() -> None:
    sandbox = MagicMock()
    stdout = "\n".join(f"line{i}" for i in range(200)).encode()
    sandbox.execute_command = AsyncMock(
        return_value=CommandResult(status=SandboxCommandStatus.SUCCESS, return_code=0, stdout=stdout)
    )
    sandbox.write_file = AsyncMock()
    provider = _fake_provider(sandbox)

    with patch("src.agent_core.tools.sandbox_tool.get_sandbox_provider", return_value=provider):
        result = await run_command.coroutine(command=["echo", "hi"], config=_config())

    sandbox.write_file.assert_awaited_once()
    written_path, written_content = sandbox.write_file.await_args.args
    assert written_path.startswith("tool_output/") and written_path.endswith("_stdout.log")
    assert written_content.count("\n") == 199  # 完整 200 行内容原样落盘，一行不少

    assert "line0" in result  # 头部预览
    assert "line199" in result  # 尾部预览
    assert "line100" not in result  # 中间被省略
    assert "中间省略" in result
    assert written_path in result  # 提示文本里带落盘路径
