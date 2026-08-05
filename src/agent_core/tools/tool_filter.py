"""工具过滤名单：标记哪些工具调用不推送给前端展示、不持久化落库。

原样迁移自 `src/core/tools/tool_filter.py`，改造为 `ToolFilterRegistry` 类
（原实现是模块级 `set[str]` + 裸函数），支持运行时动态注册/注销，无需重启服务。
"""
from __future__ import annotations


class ToolFilterRegistry:
    """进程内的工具名黑名单。

    典型场景：`save_memory`/`recall_memory` 这类内部记忆工具，调用过程
    不应该作为 `tool_call`/`tool_response` 事件推送给前端，也不应该写入
    对话消息的持久化记录。
    """

    def __init__(self) -> None:
        self._filtered: set[str] = set()

    def register(self, *tool_names: str) -> None:
        """将一个或多个工具名加入过滤名单。

        Args:
            tool_names: 待过滤的工具名，可变参数。
        """
        self._filtered.update(tool_names)

    def unregister(self, *tool_names: str) -> None:
        """从过滤名单中移除一个或多个工具名。

        Args:
            tool_names: 待移除的工具名，可变参数。
        """
        self._filtered.difference_update(tool_names)

    def is_filtered(self, tool_name: str) -> bool:
        """判断某个工具是否在过滤名单中。"""
        return tool_name in self._filtered

    def snapshot(self) -> frozenset[str]:
        """返回当前名单的只读快照，用于查询和调试。"""
        return frozenset(self._filtered)


# 进程内唯一的过滤名单实例。工具过滤是纯内存态的运行时开关，不涉及持久化，
# 不需要走 init_xxx()/get_xxx() 那一套需要显式初始化的单例模式，模块导入时
# 直接构造一个空实例即可安全使用。
tool_filter_registry = ToolFilterRegistry()
