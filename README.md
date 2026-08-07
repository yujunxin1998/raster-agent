# raster-agent-server

从 `diit-agent-server` 迁移而来的 Agent Loop 基础设施工程。设计背景与整体思路见
[`docs/新一代AgentLoop工程设计方案.md`](docs/新一代AgentLoop工程设计方案.md)。

## 本工程范围

第一期基础设施（skill/memory/tool/prompt 机制 + guardrail/sandbox/workspace/
middlewares）、第二期（**Lead Agent + 委派工具**）均已完成。不再是
`diit-agent-server` 的"Supervisor 多图路由 + 结构化路由决策"，而是一个
`langchain.agents.create_agent` 组装的 Lead Agent，工具集里既有可以直接执行的
基础工具，也有 `delegate_to_rag_agent`/`delegate_to_web_search_agent`/
`delegate_to_database_agent` 三个委派工具；`datasource_id` 强制路由有代码层面
的硬校验（`DatasourceRoutingMiddleware`）。`POST /chat/`（非流式）、
`/conversations/*` CRUD、WebSocket `/ws/chat`（流式 + 前端工具双通道回环）均
已可用；Redis-based Eval 遥测（`eval:trace`/`eval:tool`/`eval:http`）已迁移，
`REDIS_URL` 未配置时全程静默降级。

已迁移 / 新增的能力：

| 模块 | 说明 | 状态 |
|---|---|---|
| `src/agent_core/skills/` | 技能机制（渐进式披露），原样迁移自 `src/core/skills/`，脚本执行改为经由 sandbox 层，新增 `required_secrets` 密钥按需注入 | 迁移 + 改造 |
| `src/agent_core/memory/` | 三层记忆机制（短期/长期/压缩），原样迁移自 `src/core/memory/`，新增 graph-free 的 `compress_messages()` 供中间件调用 | 迁移 + 改造 |
| `src/agent_core/tools/` | 前后端工具双通道机制 + `database_tool.py`（数据库自然语言查询），原样迁移自 `src/core/tools/`/`src/service/db_query_service.py` | 迁移 |
| `src/agent_core/prompts/` | 提示词模板（17 条，含 `LEAD_AGENT`/`RAG_AGENT`/`TITLE_GENERATION` 等），原样迁移自 `src/core/prompts/`，新增 `PromptFactory` 封装 | 迁移 + 封装 |
| `src/agent_core/guardrail/` | 权限控制，工具调用前置授权 | 新增 |
| `src/agent_core/sandbox/` | 沙箱执行环境，技能脚本统一经此执行 | 新增 |
| `src/agent_core/workspace/` | 按会话隔离的虚拟工作区 | 新增 |
| `src/agent_core/middlewares/` + `src/agent_core/loop.py` | 中间件流水线，12 个 `AgentMiddleware` 子类 + `build_middlewares()`，已接入 `create_agent(...)`（见 `agent_core/agents/lead_agent.py`） | 新增 |
| `src/agent_core/model/` | LLM 工厂，参考 DeerFlow 封装：`create_chat_model()` + `PatchedChatDeepSeek`（reasoning_content 多轮续接修复） | 新增 |
| `src/agent_core/agents/` | Lead Agent 组装 + 委派工具：`lead_agent.py`/`delegation_tools.py`/`sub_agent_factory.py`/`datasource_routing_middleware.py`/`checkpointer.py`/`chat_pipeline.py`（REST/WS 共用的对话轮次驱动逻辑） | 新增 |
| `src/agent_core/eval/` | Redis Streams 可观测性埋点（`emitter.py`/`http_middleware.py`），原样迁移自 `src/core/evaluation/`，简化了旧 Supervisor 多节点图特有的"agent 切换检测"逻辑 | 迁移 + 简化 |
| `src/storage/` | 持久化层（DAO），asyncpg 原生 SQL，新增 `conversation_store`/`message_store` | 迁移 + 新增 |
| `src/api/router/` | `/skills`、`/memories`、`/health`、`/chat`（非流式）、`/conversations` REST 接口 | 迁移 + 新增 |
| `src/api/websocket/` | `/ws/chat` 流式协议 + `connection_manager.py`（前端工具双通道回环） | 新增 |

## 目录结构

```
src/
├── main.py                # FastAPI 入口，lifespan 组装全部基础设施
├── config/                 # 配置（pydantic-settings）
├── common/                 # 异常体系 / 统一响应 / 枚举常量
├── agent_core/              # Harness 层：skills / memory / tools / prompts / guardrail / sandbox
│                             #   / workspace / middlewares / loop.py / model / agents
├── storage/                # DAO 层：数据库连接池 + 各类 Store（含 conversation/message）
├── schema/                  # DTO 层：Pydantic 请求/响应模型
├── api/
│   ├── router/              # Controller 层：REST 路由（含 chat/conversation）
│   └── websocket/            # WebSocket 路由（/ws/chat + ConnectionManager）
└── utils/                   # 通用工具函数
skills/                      # 技能内容目录（core/ + public/），从原项目拷贝
test/                        # pytest 单测（skills / middlewares / memory / model / agents / eval / websocket）
```

## 代码规范

命名遵循 Python 惯用写法（PEP8：类 `UpperCamelCase`，函数/变量 `snake_case`），
设计原则遵循《阿里巴巴Java开发手册》中与具体语言无关的部分：

- 分层架构：`api`（Controller）→ `agent_core` / `storage`（Service/Manager + DAO）→ `schema`（DTO），依赖方向单向，不允许下层反向依赖上层
- 单一职责：每个类只负责一件事，文件名即职责（如 `skill_loader.py` 只做加载解析，不掺杂执行逻辑）
- 禁止魔法值：`src/common/constants.py` 集中定义枚举常量，业务代码不出现裸字符串/裸数字
- 完整异常分层：`src/common/exceptions.py` 定义统一异常基类及各模块子类，禁止裸 `except Exception: pass`
- 防御式编程：公共方法入口做参数校验，避免 `None`/空值向下游传播
- 规范注释：类/公共方法使用 Google 风格 docstring，说明 `Args`/`Returns`/`Raises`

## 快速开始

```bash
cp .env.example .env   # 按需填写 DATABASE_URL 等配置
pip install -r requirements.txt
python -m src.main
# 或
uvicorn src.main:app --reload --host 0.0.0.0 --port 8080
```

运行 `/chat`/`/conversations`/`/ws/chat` 需要 `.env` 里的 `DATABASE_URL` 指向一个
真实可用的 Postgres（用于 asyncpg 连接池 + LangGraph `AsyncPostgresSaver`
checkpointer），仓库自带的 `.env.example` 占位值连不上，需要替换成真实凭据。
`delegate_to_database_agent` 需要额外配置 `DB_QUERY_API_URL`/`ECHART_API_URL`
指向的外部服务（默认值是内网地址，本机开发环境大概率不可达——这是已知限制，
不可达时该委派工具会返回错误说明文本，不会中断对话）；`REDIS_URL` 留空即禁用
Eval 遥测，不影响其余功能。

## 与 diit-agent-web 的兼容性

`/skills/*`、`/memories/*`、`/chat/`（非流式）、`/conversations/*`、`/ws/chat`
的路径、参数、响应结构与 `diit-agent-server` 保持一致，前端相关 api 模块理论上
无需改动即可对接。
