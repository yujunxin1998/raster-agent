from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.agent_core.sandbox.docker_sandbox_provider import init_docker_sandbox_provider
from src.agent_core.workspace.thread_workspace_manager import ThreadWorkspaceManager


async def _initialize(tmp_path: Path) -> None:
    await init_docker_sandbox_provider(
        ThreadWorkspaceManager(str(tmp_path)),
        image="registry.example/sandbox@sha256:abc",
        project_root=str(tmp_path),
        default_timeout_seconds=30,
        max_output_bytes=1024,
        max_memory_mb=256,
    )


async def test_init_checks_daemon_and_preloaded_image(tmp_path: Path) -> None:
    client = MagicMock()
    with patch("docker.from_env", return_value=client):
        await _initialize(tmp_path)

    client.ping.assert_called_once_with()
    client.images.get.assert_called_once_with("registry.example/sandbox@sha256:abc")


async def test_init_fails_closed_when_daemon_is_unavailable(tmp_path: Path) -> None:
    client = MagicMock()
    client.ping.side_effect = OSError("daemon down")

    with patch("docker.from_env", return_value=client):
        with pytest.raises(RuntimeError, match="禁止降级到 Local"):
            await _initialize(tmp_path)


async def test_init_fails_closed_when_image_is_missing(tmp_path: Path) -> None:
    client = MagicMock()
    client.images.get.side_effect = LookupError("image missing")

    with patch("docker.from_env", return_value=client):
        with pytest.raises(RuntimeError, match="镜像不可用"):
            await _initialize(tmp_path)
