"""一次性把旧版本（Memory v1）ES 索引里的 Facts 灌入 `user_memory_fact`。

Memory v2 把 PostgreSQL 变成 Facts 的规范化主存，ES 降级为纯检索投影
（`ElasticsearchMemoryStore`）。对已经在跑 v1 的部署，旧数据只存在于 ES
索引里，需要先跑这个脚本把它们迁移进 `user_memory_fact`，之后
`FactProjector` 才能基于 PostgreSQL 重新构建/维护 ES 投影。

用法::

    python -m src.storage.migrations.backfill_facts_from_es

- 保留原文档 ID（原 ES `_id` 就是 `uuid4()` 字符串）、`status`、时间戳。
- 旧文档没有 `confidence` 字段（v1 没有这个概念），迁移时统一给 1.0——
  这些 Fact 已经在生产环境验证过，没有理由降级为 pending。
- `source`/`scope_type`/`scope_id` 是 v1 的展示性字段，新表结构里没有对应
  列，直接丢弃（不影响功能，历史来源仍能从 `memory_audit_logs` 查到）。
- 与新写入的唯一去重键冲突时跳过（`ON CONFLICT ... DO NOTHING`），脚本可以
  安全地重复执行——不会在 Facts 已经开始通过 Memory v2 正常写入之后产生
  重复数据。
- 只处理 PostgreSQL 里还没有的 Fact；已经迁移过的记录不会重复写入
  （幂等，可以在迁移中途失败后直接重跑）。
- 不在 `main.py::lifespan` 里自动触发，需要运维手动执行一次。
"""
from __future__ import annotations

import asyncio
import json

from elasticsearch import AsyncElasticsearch
from elasticsearch.helpers import async_scan
from loguru import logger

from src.config.settings import get_settings
from src.storage.database import init_database_pool
from src.storage.user_memory_fact_store import init_user_memory_fact_store, normalize_fact

_BACKFILL_CONFIDENCE = 1.0


async def backfill(es: AsyncElasticsearch, index: str, pool) -> tuple[int, int]:
    """执行一次全量扫描 + 灌入。

    Args:
        es: 已连接的 ES 客户端。
        index: 旧记忆索引名。
        pool: PostgreSQL 连接池。

    Returns:
        `(migrated, skipped)`：成功写入条数、因唯一键冲突跳过的条数。
    """
    migrated = 0
    skipped = 0

    async for hit in async_scan(es, index=index, query={"query": {"match_all": {}}}):
        source = hit["_source"]
        fact_id = hit["_id"]
        content = source.get("content")
        category = source.get("memory_type")
        if not content or not category:
            logger.warning(f"[Backfill] 跳过缺少 content/memory_type 的文档 id={fact_id}")
            continue

        # 幂等性分两步处理，因为一条 INSERT 只能有一个 ON CONFLICT 仲裁索引：
        # 1) fact_id 是 PK 且直接复用 ES 文档 id，重跑脚本时用它判断"是否已迁移过"；
        # 2) 剩下交给 user_memory_fact 的确定性去重唯一索引处理——v1 的相似度去重
        #    不精确，ES 里可能本来就存在多条 normalized_content 相同的旧文档。
        already_migrated = await pool.fetchval(
            "SELECT 1 FROM user_memory_fact WHERE fact_id = $1", fact_id,
        )
        if already_migrated:
            skipped += 1
            continue

        normalized = normalize_fact(content)
        row = await pool.fetchrow(
            """
            INSERT INTO user_memory_fact (
                fact_id, user_id, agent_name, content, normalized_content, category,
                importance, confidence, status, source_conversation_id,
                evidence_message_ids, superseded_by_fact_id, created_at, expires_at,
                last_accessed_at, access_count
            )
            VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb,$12,$13,$14,$15,$16)
            ON CONFLICT (user_id, COALESCE(agent_name, ''), category, normalized_content)
            WHERE status IN ('active', 'pending')
            DO NOTHING
            RETURNING fact_id
            """,
            fact_id, source.get("user_id", "default"), source.get("agent_name"),
            content, normalized, category,
            int(source.get("importance", 5)), _BACKFILL_CONFIDENCE,
            source.get("status", "active"), source.get("conversation_id"),
            json.dumps([]), source.get("superseded_by"),
            source.get("created_at"), source.get("expires_at"),
            source.get("last_accessed_at"), int(source.get("access_count") or 0),
        )
        if row is None:
            skipped += 1
        else:
            migrated += 1

    return migrated, skipped


async def main() -> None:
    settings = get_settings()
    if not settings.ES_URL:
        logger.error("[Backfill] ES_URL 未配置，无法读取旧数据，终止")
        return

    pool_manager = await init_database_pool(settings.DATABASE_URL)
    fact_store = await init_user_memory_fact_store(pool_manager.pool)  # 确保目标表已存在
    del fact_store  # 只需要建表副作用，本脚本直接用 pool 执行批量 SQL

    client_kwargs: dict = {"hosts": [settings.ES_URL], "verify_certs": settings.ES_VERIFY_CERTS}
    if settings.ES_API_KEY:
        client_kwargs["api_key"] = settings.ES_API_KEY
    elif settings.ES_USERNAME:
        client_kwargs["basic_auth"] = (settings.ES_USERNAME, settings.ES_PASSWORD)

    es = AsyncElasticsearch(**client_kwargs)
    try:
        migrated, skipped = await backfill(es, settings.ES_MEMORY_INDEX, pool_manager.pool)
    finally:
        await es.close()
        await pool_manager.close()

    logger.info(f"[Backfill] 完成：迁移 {migrated} 条，跳过（已存在/冲突）{skipped} 条")


if __name__ == "__main__":
    asyncio.run(main())
