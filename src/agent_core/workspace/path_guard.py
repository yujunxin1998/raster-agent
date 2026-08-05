"""路径穿越防护。

提取自原项目 `src/core/skills/file_tree.py::SkillFileTreeReader._resolve()`
"解析绝对路径后校验没有越出根目录"的思路，独立成通用类，供虚拟工作区
（ThreadWorkspace）和技能文件浏览（SkillFileTreeReader）共用同一套校验逻辑，
不再各自维护一份容易出现校验遗漏的路径拼接代码。
"""
from __future__ import annotations

from pathlib import Path

from src.common.exceptions import PathTraversalError


class PathGuard:
    """围绕单个根目录提供"相对路径 → 校验过的绝对路径"转换。

    典型用法::

        guard = PathGuard(root=workspace_dir)
        real_path = guard.resolve("reports/summary.md")  # 越界会抛 PathTraversalError
    """

    def __init__(self, root: Path) -> None:
        """初始化路径守卫。

        Args:
            root: 允许访问的根目录，必须是已存在的目录路径。

        Raises:
            ValueError: root 为空或不是一个目录路径的字符串表示。
        """
        if root is None:
            raise ValueError("root 不能为空")
        self._root = root.resolve()

    @property
    def root(self) -> Path:
        """返回校验过的根目录绝对路径。"""
        return self._root

    def resolve(self, relative_path: str) -> Path:
        """将相对路径解析为绝对路径，并校验没有越出根目录。

        Args:
            relative_path: 相对根目录的路径，允许为空字符串（表示根目录本身）。

        Returns:
            校验通过的绝对路径（不保证目标一定存在，是否存在由调用方自行判断）。

        Raises:
            PathTraversalError: 解析后的绝对路径越出了根目录范围（如包含 ".."
                导致跳出根目录、或传入了绝对路径试图直接指向根目录之外的位置）。
        """
        candidate = (self._root / (relative_path or "")).resolve()
        if candidate != self._root and self._root not in candidate.parents:
            raise PathTraversalError(
                f"非法路径: {relative_path!r} 越出根目录 {self._root}"
            )
        return candidate

    def to_relative(self, absolute_path: Path) -> str:
        """将根目录下的绝对路径转换为对外展示用的相对路径（统一用 "/" 分隔）。

        Args:
            absolute_path: 必须位于 root 之内的绝对路径。

        Returns:
            以 "/" 分隔的相对路径字符串。
        """
        return str(absolute_path.relative_to(self._root)).replace("\\", "/")
