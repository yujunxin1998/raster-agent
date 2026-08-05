"""委派工具 + Lead Agent 组装（设计文档 4.1 节"Supervisor 多图路由 → Lead Agent + 委派工具"）。

对应第二期第一轮：`rag_agent`/`web_search_agent` 从原项目"Supervisor 图上的固定
节点"改造成 Lead Agent 可以按需调用的 `delegate_to_xxx` 工具；`general_agent`/
`tool_agent` 的工具直接并入 Lead Agent 自己的基础工具集，不再单独委派；
`database_agent` 依赖的外部查询服务本轮未迁移，`delegate_to_database_agent`
推迟到下一轮。
"""
from __future__ import annotations
