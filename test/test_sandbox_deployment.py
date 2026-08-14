import pytest

from src.main import validate_sandbox_deployment


@pytest.mark.parametrize("environment", ["test", "staging", "production"])
def test_online_environments_require_docker(environment: str) -> None:
    with pytest.raises(RuntimeError, match="必须使用 Docker"):
        validate_sandbox_deployment(environment, "local")


def test_development_allows_local() -> None:
    validate_sandbox_deployment("development", "local")


def test_online_environment_allows_docker() -> None:
    validate_sandbox_deployment("production", "docker")


def test_unknown_provider_is_rejected() -> None:
    with pytest.raises(RuntimeError, match="不支持"):
        validate_sandbox_deployment("development", "remote")
