"""一次性数据迁移脚本集合。

跟 `src/storage/*_store.py` 的 `CREATE TABLE IF NOT EXISTS` 幂等建表不同，
这里的脚本处理"把旧数据灌进新表结构"这类只需要跑一次的操作，由运维手动
执行（`python -m src.storage.migrations.<script>`），不在应用启动路径
（`main.py::lifespan`）里自动触发。
"""
from __future__ import annotations
