from pathlib import Path

from src.agent_core.sandbox.local_sandbox import LocalSandbox
from src.agent_core.workspace.thread_workspace import ThreadWorkspace
from src.common.constants import SandboxCommandStatus


async def test_local_sandbox_truncates_stderr(tmp_path: Path) -> None:
    workspace = ThreadWorkspace("thread", "user", tmp_path / "thread")
    workspace.ensure_directories()
    sandbox = LocalSandbox(
        workspace,
        default_timeout_seconds=10,
        max_output_bytes=10,
        max_memory_mb=256,
    )

    result = await sandbox.execute_command(
        ["python", "-c", "import sys; sys.stderr.write('e' * 20); raise SystemExit(1)"]
    )

    assert result.status == SandboxCommandStatus.OUTPUT_TRUNCATED
    assert result.truncated is True
    assert len(result.stderr) == 10
