"""中间件流水线骨架（对应设计文档 4.2 节）。

本包内的每个类都是 `langchain.agents.middleware.AgentMiddleware` 的子类，
可以独立导入、独立单测；`src/agent_core/loop.py::build_middlewares()` 按
文档顺序把它们组装成一个列表。

边界说明：本工程本轮不包含编排层（见仓库根目录 README「本工程范围」），
这里还没有真正的 Lead Agent 把这份列表传给 `create_agent(middleware=...)`
去跑一条活的请求链路——那是第二期的工作。
"""
from __future__ import annotations
