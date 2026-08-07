"""`build_sandbox_env` 单元测试：密钥过滤 + 强制 UTF-8 编码。"""
from __future__ import annotations

from src.agent_core.sandbox.env_policy import build_sandbox_env


def test_forces_utf8_io_encoding_even_if_host_has_none() -> None:
    env = build_sandbox_env()

    assert env["PYTHONIOENCODING"] == "utf-8"


def test_extra_env_cannot_override_utf8_io_encoding() -> None:
    env = build_sandbox_env({"PYTHONIOENCODING": "gbk"})

    assert env["PYTHONIOENCODING"] == "utf-8"


def test_filters_secret_like_names_from_host_environ(monkeypatch) -> None:
    monkeypatch.setenv("SOME_API_KEY", "sk-should-not-leak")
    monkeypatch.setenv("DATABASE_URL", "postgres://user:pass@host/db")
    monkeypatch.setenv("HARMLESS_VAR", "keep-me")

    env = build_sandbox_env()

    assert "SOME_API_KEY" not in env
    assert "DATABASE_URL" not in env
    assert env["HARMLESS_VAR"] == "keep-me"
