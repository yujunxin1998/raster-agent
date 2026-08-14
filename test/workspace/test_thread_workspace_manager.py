from pathlib import Path

import pytest

from src.agent_core.workspace.thread_workspace_manager import ThreadWorkspaceManager


def test_workspace_is_scoped_by_valid_user_and_conversation(tmp_path: Path) -> None:
    manager = ThreadWorkspaceManager(str(tmp_path))

    workspace = manager.get_or_create("thread_01", "user-01")

    expected = tmp_path.resolve() / "users" / "user-01" / "threads" / "thread_01"
    assert workspace.root == expected
    assert workspace.workspace_dir.is_dir()
    assert workspace.uploads_dir.is_dir()
    assert workspace.outputs_dir.is_dir()


@pytest.mark.parametrize(
    ("conversation_id", "user_id"),
    [
        ("../escape", "user"),
        ("thread", "../escape"),
        ("/absolute", "user"),
        ("thread", "user/name"),
        ("thread", "x" * 65),
    ],
)
def test_workspace_rejects_unsafe_directory_ids(
    tmp_path: Path, conversation_id: str, user_id: str
) -> None:
    manager = ThreadWorkspaceManager(str(tmp_path))

    with pytest.raises(ValueError):
        manager.get_or_create(conversation_id, user_id)
