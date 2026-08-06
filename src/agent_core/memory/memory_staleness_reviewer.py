"""记忆 staleness 定期复核（设计文档第二节表格"记忆模块的 staleness pass"）。

对应 DeerFlow"老记忆定期复核是否该淘汰"的思路：过期或长期无人问津的低重要度
记忆会被批量归档为 `archived`（不是物理删除，仍可在管理 API 里看到并手动恢复）。
不写审计日志——`MemoryAuditStore` 现有的动作类型都是围绕单条记忆 + 某个 user_id
的操作，staleness 是跨用户的批量扫描，硬套审计表语义不通顺，失败也只记 warning
静默降级，和 `MemoryJobsStore` 现有"辅助能力失败不影响主流程"的容错哲学一致。
"""
from __future__ import annotations

from loguru import logger

from src.agent_core.memory.base_memory_store import BaseMemoryStore


class MemoryStalenessReviewer:
    """记忆 staleness 复核器，供后台维护循环周期性调用。"""

    def __init__(self, store: BaseMemoryStore, *, max_age_days: int, low_importance_threshold: int) -> None:
        """初始化复核器。

        Args:
            store: 长期记忆存储实现。
            max_age_days: 低重要度记忆的"创建多久以后才算陈旧"天数阈值。
            low_importance_threshold: 参与陈旧判定的重要度上限（<=）。
        """
        self._store = store
        self._max_age_days = max_age_days
        self._low_importance_threshold = low_importance_threshold

    async def run_once(self) -> int:
        """执行一轮 staleness 扫描，异常不向上抛出。

        Returns:
            本轮归档的记忆条数；扫描失败时返回 0。
        """
        try:
            archived = await self._store.sweep_stale(
                max_age_days=self._max_age_days,
                low_importance_threshold=self._low_importance_threshold,
            )
            logger.info(f"[MemoryStalenessReviewer] 归档 {archived} 条陈旧记忆")
            return archived
        except Exception as exc:
            logger.warning(f"[MemoryStalenessReviewer] 复核失败，已跳过: {exc}")
            return 0
