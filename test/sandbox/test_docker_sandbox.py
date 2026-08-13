"""`DockerSandbox` 单元测试：全部 mock `docker_client`，不依赖真实 Docker daemon。

文件类方法（read_file/write_file/list_dir）委托给内部真实的 `LocalSandbox`，
用 `tmp_path` 构造真实 `ThreadWorkspace` 直接验证；只有 `execute_command`
才需要 mock 容器交互。
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

from src.agent_core.sandbox.docker_sandbox import DockerSandbox
from src.agent_core.workspace.thread_workspace import ThreadWorkspace
from src.common.constants import SandboxCommandStatus, WorkspaceDirectory


def _make_sandbox(tmp_path: Path, docker_client, *, max_output_bytes: int = 2 * 1024 * 1024) -> DockerSandbox:
    workspace = ThreadWorkspace(conversation_id="c1", user_id="u1", root=tmp_path / "thread")
    workspace.ensure_directories()
    return DockerSandbox(
        workspace,
        docker_client,
        image="python:3.11-slim",
        project_root=str(tmp_path / "project"),
        default_timeout_seconds=30,
        max_output_bytes=max_output_bytes,
        max_memory_mb=256,
        max_pids=64,
        network_enabled=False,
    )


def _fake_container(*, exit_code: int = 0, stdout: bytes = b"", stderr: bytes = b"") -> MagicMock:
    container = MagicMock()
    container.id = "fake-container-id"
    container.wait = MagicMock(return_value={"StatusCode": exit_code})

    def _logs(stdout=False, stderr=False):
        if stdout:
            return _fake_container_stdout
        if stderr:
            return _fake_container_stderr
        return b""

    _fake_container_stdout = stdout
    _fake_container_stderr = stderr
    container.logs = MagicMock(side_effect=_logs)
    return container


async def test_read_write_list_dir_delegate_to_local_sandbox(tmp_path: Path) -> None:
    docker_client = MagicMock()
    sandbox = _make_sandbox(tmp_path, docker_client)

    await sandbox.write_file("a.txt", "hello")
    content = await sandbox.read_file("a.txt")
    entries = await sandbox.list_dir()

    assert content == "hello"
    assert entries == [{"name": "a.txt", "path": "a.txt", "type": "file"}]
    docker_client.containers.run.assert_not_called()


async def test_write_file_to_outputs_directory(tmp_path: Path) -> None:
    docker_client = MagicMock()
    sandbox = _make_sandbox(tmp_path, docker_client)

    await sandbox.write_file("report.md", "# 结果", directory=WorkspaceDirectory.OUTPUTS)
    content = await sandbox.read_file("report.md", directory=WorkspaceDirectory.OUTPUTS)

    assert content == "# 结果"


async def test_execute_command_forwards_stdin_via_workspace_file(tmp_path: Path) -> None:
    docker_client = MagicMock()
    docker_client.containers.run.return_value = _fake_container(exit_code=0)
    sandbox = _make_sandbox(tmp_path, docker_client)

    result = await sandbox.execute_command(["echo", "hi"], stdin=b"data")

    assert result.status == SandboxCommandStatus.SUCCESS
    called_command = docker_client.containers.run.call_args.args[1]
    assert called_command[:2] == ["sh", "-c"]
    assert called_command[-2:] == ["echo", "hi"]
    assert not list((tmp_path / "thread" / "workspace").glob(".sandbox-stdin-*"))


async def test_execute_command_translates_sys_executable_to_python3(tmp_path: Path) -> None:
    docker_client = MagicMock()
    docker_client.containers.run.return_value = _fake_container(exit_code=0, stdout=b"1\n")
    sandbox = _make_sandbox(tmp_path, docker_client)

    result = await sandbox.execute_command([sys.executable, "-c", "print(1)"])

    called_command = docker_client.containers.run.call_args.args[1]
    assert called_command == ["python3", "-c", "print(1)"]
    assert result.status == SandboxCommandStatus.SUCCESS
    assert result.stdout == b"1\n"


async def test_execute_command_translates_project_root_absolute_path(tmp_path: Path) -> None:
    docker_client = MagicMock()
    docker_client.containers.run.return_value = _fake_container(exit_code=0)
    sandbox = _make_sandbox(tmp_path, docker_client)

    project_root = str(tmp_path / "project")
    script_path = f"{project_root}/skills/core/demo/scripts/main.py"

    await sandbox.execute_command([sys.executable, script_path])

    called_command = docker_client.containers.run.call_args.args[1]
    assert called_command == ["python3", "/app/skills/core/demo/scripts/main.py"]


async def test_execute_command_mounts_workspace_and_project_root(tmp_path: Path) -> None:
    docker_client = MagicMock()
    docker_client.containers.run.return_value = _fake_container(exit_code=0)
    sandbox = _make_sandbox(tmp_path, docker_client)

    await sandbox.execute_command(["python3", "-c", "print(1)"])

    _, kwargs = docker_client.containers.run.call_args
    volumes = kwargs["volumes"]
    assert volumes[str(tmp_path / "thread" / "workspace")]["bind"] == "/workspace"
    assert volumes[str(tmp_path / "project")] == {"bind": "/app", "mode": "ro"}
    assert kwargs["working_dir"] == "/workspace"
    assert kwargs["mem_limit"] == "256m"
    assert kwargs["pids_limit"] == 64
    assert kwargs["network_disabled"] is True
    assert kwargs["read_only"] is True
    assert kwargs["cap_drop"] == ["ALL"]
    assert kwargs["security_opt"] == ["no-new-privileges"]


async def test_execute_command_failed_status(tmp_path: Path) -> None:
    docker_client = MagicMock()
    docker_client.containers.run.return_value = _fake_container(exit_code=1, stderr=b"boom")
    sandbox = _make_sandbox(tmp_path, docker_client)

    result = await sandbox.execute_command(["python3", "-c", "raise SystemExit(1)"])

    assert result.status == SandboxCommandStatus.FAILED
    assert result.return_code == 1
    assert result.stderr == b"boom"


async def test_execute_command_timeout_kills_and_removes_container(tmp_path: Path) -> None:
    docker_client = MagicMock()
    container = _fake_container()
    container.wait = MagicMock(side_effect=Exception("timed out"))
    docker_client.containers.run.return_value = container
    sandbox = _make_sandbox(tmp_path, docker_client)

    result = await sandbox.execute_command(["python3", "-c", "import time; time.sleep(999)"], timeout=1)

    assert result.status == SandboxCommandStatus.TIMEOUT
    container.kill.assert_called_once()
    container.remove.assert_called_once_with(force=True)


async def test_execute_command_truncates_output(tmp_path: Path) -> None:
    docker_client = MagicMock()
    docker_client.containers.run.return_value = _fake_container(exit_code=0, stdout=b"x" * 20)
    sandbox = _make_sandbox(tmp_path, docker_client, max_output_bytes=10)

    result = await sandbox.execute_command(["python3", "-c", "print('x' * 20)"])

    assert result.status == SandboxCommandStatus.OUTPUT_TRUNCATED
    assert result.truncated is True
    assert len(result.stdout) == 10


async def test_execute_command_container_always_removed(tmp_path: Path) -> None:
    docker_client = MagicMock()
    container = _fake_container(exit_code=0)
    docker_client.containers.run.return_value = container
    sandbox = _make_sandbox(tmp_path, docker_client)

    await sandbox.execute_command(["python3", "-c", "print(1)"])

    container.remove.assert_called_once_with(force=True)


async def test_execute_command_rejects_empty_command(tmp_path: Path) -> None:
    docker_client = MagicMock()
    sandbox = _make_sandbox(tmp_path, docker_client)

    try:
        await sandbox.execute_command([])
        raised = False
    except ValueError:
        raised = True

    assert raised
