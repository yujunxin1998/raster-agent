---
name: database-analysis
description: >-
  指导何时以及如何使用 query_database 工具处理数据统计、筛选、排行、对比、
  占比、图表类问题（明确指向数据库数据），以及如何解读返回结果。
category: database
required_tools:
  - query_database
---

## 工具使用规则（强制）

- 仅在用户问题涉及明确的数据统计、筛选、排序、对比、占比等查询需求时调用 `query_database`
- 需要当前请求已配置数据源（datasource_id），否则 `query_database` 无法执行
- 不得凭记忆编造数据库查询结果，所有数据必须来自 `query_database` 返回内容
- 若返回"未配置数据源"或查询失败的提示，必须如实告知用户，不得编造数据掩盖失败

## 回答规则

- 返回内容可能包含 `<analysis>` 查询过程说明、Markdown 表格、`<echart>` 图表配置块
- 直接基于返回的表格、图表配置和数据解读组织最终回答，保留表格和 `<echart>` 块的原始格式，不要省略或转述图表配置 JSON
- 用简洁的语言概括查询条件和关键结论，避免重复输出整段 `<analysis>` 原文
