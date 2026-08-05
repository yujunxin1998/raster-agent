"""技能目录的文件浏览：给前端"展开看 scripts/references 等子目录"用。

原样迁移自 `src/core/skills/file_tree.py`，路径穿越校验改为复用通用的
`PathGuard`（原实现里的 `_resolve()` 私有方法与 PathGuard 的逻辑完全一致，
现在两处共用同一份代码，不再各自维护）。
"""
from __future__ import annotations

from src.agent_core.skills.skill_definition import SkillDefinition
from src.agent_core.workspace.path_guard import PathGuard

_IGNORED_NAMES = frozenset({"__pycache__", ".DS_Store", ".git"})
_MAX_FILE_BYTES = 200_000  # 超过这个大小只读取前 200KB，避免一次性把超大文件灌进内存/前端


class SkillFileTreeReader:
    """围绕单个 SkillDefinition 提供"列子目录 / 读文件原文"两个操作。

    只在用户点开技能/文件夹时才列目录、读文件——延续渐进式披露原则，
    也只暴露 skill_dir 内部内容（带路径越界校验）。
    """

    def __init__(self, skill: SkillDefinition) -> None:
        self._skill = skill
        self._path_guard = PathGuard(root=skill.skill_dir)

    def list_children(self, relative_path: str = "") -> list[dict]:
        """列出 relative_path 目录下的直接子节点（不递归），目录在前、按名称排序。

        Args:
            relative_path: 相对技能根目录的路径，空字符串表示技能根目录本身。

        Returns:
            形如 `[{"name": ..., "path": ..., "type": "dir" | "file"}, ...]` 的列表。

        Raises:
            ValueError: relative_path 不是一个目录。
            PathTraversalError: relative_path 越出了技能根目录范围。
        """
        target_dir = self._path_guard.resolve(relative_path)
        if not target_dir.is_dir():
            raise ValueError(f"'{relative_path}' 不是一个目录")

        entries = [entry for entry in target_dir.iterdir() if entry.name not in _IGNORED_NAMES]
        entries.sort(key=lambda entry: (entry.is_file(), entry.name.lower()))

        return [
            {
                "name": entry.name,
                "path": self._path_guard.to_relative(entry),
                "type": "dir" if entry.is_dir() else "file",
            }
            for entry in entries
        ]

    def read_file(self, relative_path: str) -> dict:
        """读取 relative_path 对应文件的原文，超大/二进制文件做降级处理。

        Args:
            relative_path: 相对技能根目录的文件路径。

        Returns:
            形如 `{"content": ..., "truncated": bool}` 的字典。

        Raises:
            ValueError: relative_path 不是一个文件。
            PathTraversalError: relative_path 越出了技能根目录范围。
        """
        target_file = self._path_guard.resolve(relative_path)
        if not target_file.is_file():
            raise ValueError(f"'{relative_path}' 不是一个文件")

        raw = target_file.read_bytes()
        truncated = len(raw) > _MAX_FILE_BYTES
        raw = raw[:_MAX_FILE_BYTES]

        try:
            content = raw.decode("utf-8")
        except UnicodeDecodeError:
            return {"content": "[二进制文件，无法预览]", "truncated": False}

        return {"content": content, "truncated": truncated}
