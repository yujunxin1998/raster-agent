# Skill 与 Tool 完全解耦重构设计

> 状态：已实现（预路由匹配方式范围收窄为仅规则/元数据，见下方说明）  
> 目标仓库：raster-agent  
> 编写基线：main@5bc8c3f  
> 相关旧文档：Skill注入与Load-Skill重构设计.md

## 实现范围说明（补充）

第 7.2 节把预路由匹配方式留为开放项（规则匹配 / Embedding 召回 / 轻量 LLM
重排）。实现时与用户确认：第一阶段只做规则/元数据匹配（显式指定 + `activation`
声明），不引入 Embedding 或额外 LLM 调用——原因见 `src/agent_core/skills/
skill_router.py` 模块文档。`auto_activate` 目前恒等于 `forced`，语义相关但
未显式声明 `required` 的技能不会被自动激活，仍走 `catalog_candidates` +
`load_skill` 路径。这不违反本文档的 Definition of Done（第 19 节），后续如需
接入语义路由，扩展点在 `skill_router.py::route()`。

## 1. 执行摘要

当前系统虽然已经让部分工作流 Skill 通过 SkillMiddleware 和 load_skill 加载，但仍保留完整的 Skill 到 Tool 投影链：

~~~text
SkillDefinition
  -> SkillKind
  -> SkillToolFactory
  -> StructuredTool
  -> SkillToolProvider
  -> ToolRegistry
~~~

这导致 Skill 同时承担工作流知识、Tool Schema、运行时上下文、密钥注入和脚本执行等职责。目标架构必须建立以下边界：

~~~text
Skill = 模型侧的专业知识、操作流程和按需资源注入机制
Tool  = 系统侧的可执行能力、副作用边界和权限边界
~~~

重构完成后：

- 不再存在 Tool 型 Skill 和 Workflow 型 Skill；
- 所有 Skill 都是按需加载的指令与资源包；
- 任意 Skill 都不会生成独立 StructuredTool；
- Skill 不进入 ToolRegistry；
- SkillMiddleware 是 Skill 发现、激活和资源读取的唯一运行入口；
- query_database 和 search_knowledge_base 迁移为独立业务 Tool；
- Skill 可以指导模型调用 Tool，但不拥有、生成、包装或绕过 Tool。

## 2. 实现前必须处理的工作树边界

实现者开始修改前必须执行：

~~~powershell
git status --short
git diff -- src/agent_core/skills/skill_load_tool.py
git diff -- src/agent_core/skills/skill_tool_factory.py
~~~

编写本文档时工作树已有未提交修改，其中包括 skill_load_tool.py 和 skill_tool_factory.py。禁止使用 git checkout、git reset --hard 或整文件粗暴覆盖。必须先识别现有修改意图，再决定吸收、迁移或废弃；无法判断时先向用户确认。

## 3. 设计原则

本方案遵循 Agent Skills 的渐进式披露思路：

1. Discovery：只暴露 Skill 名称和描述；
2. Activation：任务命中后才加载完整 SKILL.md；
3. Resource access：仅在 Skill 指令要求时读取指定 reference、script 或 template；
4. Execution：模型按照 Skill 指令调用已有 Tool；
5. Skill 中的 script 是资源，不是框架自动发现和执行的固定入口。

强解耦不等于底层不能存在任何 Tool Schema。load_skill 可以作为 SkillMiddleware 自带的控制面协议，但不能进入业务 ToolRegistry，也不能形成一个 Skill 对应一个 Tool 的映射。

## 4. 当前设计问题

### 4.1 SkillKind 是错误的领域边界

当前代码使用 parameters 是否为空推导：

~~~python
SkillKind.TOOL if self.parameters else SkillKind.WORKFLOW
~~~

这个判断不成立：

- Workflow 也可能有报告格式、时间范围等配置参数；
- Tool 也可能没有模型侧参数；
- parameters 描述调用契约，能力形态描述执行语义，二者正交；
- 增加一个可选参数不应导致注册、权限、审计和执行链发生架构级变化。

本次不是把 kind 改成显式字段，而是删除“Skill 可以是 Tool”这一概念。

### 4.2 SkillToolFactory 职责过载

当前 SkillToolFactory 同时负责：

- 生成 Pydantic Schema；
- 创建 StructuredTool；
- Guardrail 校验；
- 注入 AgentRuntimeContext；
- 解析 required_secrets；
- 获取和释放 Sandbox；
- 执行 Skill Script；
- 拼装指令、references 和脚本结果。

这些职责属于 Tool 执行层、安全层和上下文层，不应由 Skill 模块承担。

### 4.3 Skill 热重载不应触发 Tool 发布

修改一段工作流说明不应：

- 增加 ToolRegistry revision；
- 重新计算 Tool 命名冲突；
- 改变 Tool 权限快照；
- 触发 SkillRegistry 与 ToolRegistry 两阶段切换。

## 5. 目标架构与依赖方向

### 5.1 三个独立平面

~~~text
Skill Plane
  SkillLoader
  SkillRegistry
  SkillManager
  SkillContentRepository
  SkillActivationService
  SkillMiddleware

Tool Plane
  BuiltinToolProvider / BusinessToolProvider / MCPProvider
  ToolRegistry
  ToolResolver
  GuardrailMiddleware
  Sandbox Tool / MCP / Remote API

Agent Plane
  Model
  Middleware chain
  Tool execution loop
  Checkpointer / telemetry
~~~

### 5.2 允许的依赖

~~~text
SkillMiddleware -> SkillManager
SkillManager -> SkillRegistry
SkillRegistry -> SkillDefinition
SkillActivationService -> SkillContentRepository

Agent -> ToolRegistry
ToolRegistry -> ToolProvider
ToolProvider -> standalone business Tool
~~~

### 5.3 禁止的依赖

~~~text
SkillDefinition -> StructuredTool
SkillManager -> ToolRegistry
SkillManager -> SandboxProvider
ToolProvider -> SkillRegistry
SkillHotReloader -> ToolRegistry.publish()
SkillContentRepository -> execute_command()
Skill -> Tool permissions bypass
~~~

## 6. 目标 Skill 数据模型

建议 SkillDefinition 收敛为：

~~~python
@dataclass(frozen=True)
class SkillDefinition:
    name: str
    description: str
    category: SkillCategory
    skill_dir: Path
    version: str | None = None
    tags: tuple[str, ...] = ()
    allowed_agents: tuple[str, ...] = ()
    required_tools: tuple[str, ...] = ()
~~~

其中 required_tools 只是依赖声明，不表示 Skill 拥有或注册 Tool。

删除：

~~~text
SkillKind
tool_name
parameters
runtime_context_keys
required_secrets
script_path
has_script()
~~~

如前端 API 暂时依赖 tool_name，可提供带弃用告警的只读别名，但新代码不得继续使用。

目标 SKILL.md：

~~~yaml
---
name: data-analysis
description: >-
  分析结构化数据，适用于字段画像、统计分析、异常检测、
  图表生成和数据结论验证。
category: general
required_tools:
  - read_file
  - run_python
  - save_output_file
---
~~~

废弃字段 tool_name、parameters、runtime_context_keys、required_secrets 在迁移期只记录 warning，不再影响执行语义。

## 7. SkillMiddleware 设计

SkillMiddleware 是唯一 Skill 运行入口，负责：

1. 在主模型第一次调用前完成 Skill 预路由；
2. 自动激活显式指定、Agent 必需和高置信匹配的 Skill；
3. 只将模糊候选以轻量 Catalog 形式提供给模型；
4. 提供 Middleware 自身的控制面操作；
5. 拦截并处理补充 Skill 激活与资源读取请求；
6. 按具体 skill_name 校验可见性和激活权限；
7. 维护本次运行已经激活的 Skill 集合，防止重复注入；
8. 记录 Skill discovery、routing、activation 和 resource-read 事件。

它不得负责数据库查询、知识库检索、业务参数 Schema、业务 Tool 权限、密钥注入、脚本执行、Sandbox 生命周期和 ToolRegistry 发布。

### 7.1 激活策略：自动路由为主，load_skill 为补充

生产路径不得只依赖主模型主动调用 load_skill。单纯把 Skill Catalog 放进 System Prompt，可能出现任务已经命中某个 Skill，但模型认为自己能够直接完成，因而跳过 load_skill 的情况。

最终激活优先级必须是：

~~~text
SkillMiddleware 自动路由
  = 主路径

用户/API 显式指定 Skill
  = 强制路径

Agent required_skills
  = 强制路径

load_skill
  = 任务执行中的动态补充路径
~~~

完整流程：

~~~text
用户请求
  ↓
SkillMiddleware 预路由
  ├─ 显式指定 Skill：强制激活
  ├─ required Skill：强制激活
  ├─ 高置信 Skill：自动激活
  └─ 模糊候选：只暴露 Catalog
  ↓
主模型第一次调用
  ↓
已自动加载的 Skill 直接生效
  ↓
执行过程中发现新需求
  ↓
模型调用 load_skill 补充激活
~~~

这套路径同时解决：

- 主模型没有调用 load_skill 导致漏加载；
- 全量加载所有 Skill 导致上下文膨胀；
- 任务执行过程中动态发现新 Skill；
- 强制业务流程必须生效；
- 多 Skill 组合与去重；
- Skill 路由、自动激活和补充激活的独立审计。

### 7.2 路由输入与决策结果

Middleware 预路由至少综合以下信号：

1. 请求或 API 中显式传入的 Skill；
2. Agent Profile 的 required_skills；
3. 当前用户消息；
4. 必要的会话摘要或最近若干轮消息；
5. 当前 Agent 可见 Skill 的 name、description 和 tags；
6. 可选的规则匹配、Embedding 召回和轻量 LLM 重排。

路由器应返回结构化决策，而不是直接返回一段 Prompt：

~~~python
@dataclass(frozen=True)
class SkillRoutingDecision:
    forced: tuple[str, ...]
    auto_activate: tuple[str, ...]
    catalog_candidates: tuple[str, ...]
    rejected: tuple[str, ...]
    scores: Mapping[str, float]
    reasons: Mapping[str, str]
~~~

建议决策语义：

~~~text
forced
  用户显式指定或 Agent 必需，权限通过后必须加载

auto_activate
  路由器判断完成本任务需要该 Skill，达到自动激活阈值

catalog_candidates
  可能相关但置信度不足，只向主模型展示轻量目录

rejected
  不相关、无权限、依赖缺失或激活策略不允许
~~~

阈值不得硬编码为未经验证的经验值。实现时可提供初始配置，但上线阈值必须由 Skill 路由评测集校准。

### 7.3 激活策略元数据

可为 Skill 增加纯路由语义的 activation 配置：

~~~yaml
activation:
  mode: automatic
~~~

支持：

~~~text
automatic
  Middleware 可高置信自动激活，主模型也可通过 load_skill 补充激活

explicit_only
  只有用户/API 或上层 Agent 显式指定时才能激活

required
  绑定到指定 Agent，在该 Agent 中始终激活
~~~

activation 只控制 Skill 指令是否进入上下文，不授予任何 Tool 权限。

### 7.4 激活状态与重复注入

Middleware 必须维护本次 Agent 运行的激活状态，例如：

~~~python
activated_skills: dict[str, ActivatedSkill]
~~~

每条记录至少包含：

~~~text
skill_name
skill_version 或内容哈希
activation_source: explicit / required / auto / load_skill
activated_at
routing_score
~~~

同一版本 Skill 已激活时，再次调用 load_skill 应返回“已激活”的轻量结果，不得重复把完整正文写入消息。Skill 热重载后是否在当前运行中切换版本必须有明确策略；默认建议单次运行固定版本，新请求再使用新版本。

建议利用 AgentMiddleware.tools 提供：

~~~python
class SkillMiddleware(AgentMiddleware[Any, AgentRuntimeContext]):
    tools = [load_skill, read_skill_resource]
~~~

### 7.5 load_skill

输入：

~~~text
load_skill(skill_name)
~~~

返回：

- 完整 SKILL.md 正文；
- 可用资源索引；
- 必要的可移植路径说明；
- 缺失 required_tools 的提示。

不得自动读取全部 references。

load_skill 的定位是动态补充，而不是主路由。它主要处理：

- 主模型执行中发现新的任务子目标；
- 预路由只把某个 Skill 放入模糊候选 Catalog；
- 自动激活后又需要组合第二个 Skill；
- 用户在任务执行中追加新的 Skill 要求。

对于已在预路由阶段自动激活的 Skill，主模型无需再调用 load_skill。

### 7.6 read_skill_resource

输入：

~~~text
read_skill_resource(skill_name, resource_path)
~~~

支持按需读取 references、scripts、templates 和允许的文本资源。必须实现：

- 路径归一化；
- 拒绝绝对路径和 .. 越界；
- 防止符号链接逃逸；
- 限制文件类型和大小；
- 二进制文件只返回元数据，不注入原始字节。

## 8. 三级渐进式披露

### Level 1：Discovery

模型只看到：

~~~text
- data-analysis: 适用于结构化数据分析、异常检测和图表生成
- knowledge-base-answering: 适用于需要知识库证据的回答
~~~

### Level 2：Activation

~~~text
load_skill("data-analysis")
~~~

只加载完整 SKILL.md 和资源索引。

### Level 3：Resource access

~~~text
read_skill_resource("data-analysis", "references/chart-selection.md")
read_skill_resource("data-analysis", "scripts/profile_dataset.py")
~~~

不允许激活时默认拼接整个 references 目录。

## 9. Skill Scripts 的新语义

旧语义：

~~~text
存在 scripts/main.py
  -> SkillDefinition.has_script()
  -> SkillToolFactory 获取 Sandbox
  -> SkillContentReader.run_script()
~~~

新语义：

~~~text
SKILL.md 指导模型何时使用某个脚本
  -> 模型按需读取脚本或获得安全路径
  -> 模型调用独立 run_python/run_command Tool
  -> SandboxMiddleware/Tool 层负责执行安全
~~~

任何需要密钥、用户身份、业务权限、稳定 SLA、重试限流或远程数据访问的脚本，都应升级为独立 Tool、MCP Tool 或后端服务。

## 10. 业务 Tool 迁移

### 10.1 query_database

当前混合体：

~~~text
skills/core/query-database/SKILL.md
skills/core/query-database/scripts/main.py
src/agent_core/tools/database_tool.py
~~~

目标拆分：

~~~text
Tool: query_database
  建议位置：src/agent_core/tools/database_query_tool.py
  责任：参数校验、datasource_id 注入、数据库调用、图表、错误和遥测

Skill: database-analysis
  建议位置：skills/core/database-analysis/SKILL.md
  责任：何时查库、如何构造查询、如何解释数据、禁止编造结果
~~~

业务 Tool 应由 BuiltinToolProvider 或新建 BusinessToolProvider 注册：

~~~python
@tool
async def query_database(
    query_text: str,
    runtime: ToolRuntime[AgentRuntimeContext],
) -> str:
    datasource_id = runtime.context.datasource_id
    ...
~~~

### 10.2 search_knowledge_base

当前混合体：

~~~text
skills/core/search-knowledge-base/SKILL.md
skills/core/search-knowledge-base/scripts/main.py
src/agent_core/tools/rag_service.py
~~~

目标拆分：

~~~text
Tool: search_knowledge_base
  建议位置：src/agent_core/tools/knowledge_search_tool.py
  责任：Query Rewrite、RAG 检索、top_k、db_id、retriever_resources、Tool Result

Skill: knowledge-base-answering
  建议位置：skills/core/knowledge-base-answering/SKILL.md
  责任：检索时机、证据组合、引用格式、证据不足和冲突处理
~~~

迁移必须保持 chat_pipeline.py 当前引用解析兼容，重点核验 retriever_resources、ref_json 和 ToolMessage 原始输出。

## 11. 权限与密钥

重构后存在两套正交权限：

### Skill 激活权限

控制用户或 Agent 能否发现和加载某个 Skill，由 SkillMiddleware 按具体 skill_name 校验。可复用 GuardrailProvider，但应使用独立 resource_type 或 Skill 语义接口，不得伪装成业务 Tool 权限。

### Tool 执行权限

控制用户能否查询数据库、检索、写文件和执行命令，继续由 Tool Resolver 和 GuardrailMiddleware 管理。

安全不变式：

~~~text
允许加载 Skill
  != 自动拥有 Skill 中提到的 Tool
  != 绕过 Tool 的运行时 Guardrail
~~~

Skill Frontmatter 不再声明或接收密钥。密钥只属于独立 Tool、MCP/Connector、后端服务或 Sandbox Tool 的受控环境。

## 12. 主 Agent 与 Sub-Agent

当前主 Agent 走 SkillMiddleware，部分 Sub-Agent 依赖 create_load_skill_tool 的自包含执行逻辑。重构后必须消除双路径：

~~~text
任何需要 Skill 的 Agent
  -> 显式挂载 SkillMiddleware
  -> 使用同一 Skill Catalog
  -> 使用同一 ActivationService
  -> 使用同一 ContentRepository
~~~

如果某个 Sub-Agent 出于安全原因不允许 Middleware，则它不应获得 Skill，而不是保留一套 Tool 形态的 Skill 激活实现。

## 13. Skill 热重载

旧流程：

~~~text
watch SKILL.md
  -> SkillLoader
  -> candidate SkillRegistry
  -> SkillToolProvider.discover()
  -> ToolRegistry.publish()
  -> replace SkillManager.registry
~~~

新流程：

~~~text
watch Skill directory
  -> debounce
  -> SkillLoader
  -> candidate SkillRegistry
  -> validate names/resources
  -> atomically replace SkillManager.registry
~~~

应监听 SKILL.md、references、scripts 和 templates 的有效变化。候选 Registry 校验失败时保留旧 Registry。Skill 变化不得增加 ToolRegistry revision。

## 14. 模块改动清单

### 删除

~~~text
src/agent_core/skills/skill_tool_factory.py
SkillKind
RequiredSecret 的 Skill 执行语义
SkillToolProvider
SkillManager.tool_factory
SkillManager.get_tools()
source_type="skill"
canonical_name="skill.*"
~~~

### 重写

~~~text
src/agent_core/skills/skill_definition.py
src/agent_core/skills/skill_loader.py
src/agent_core/skills/skill_manager.py
src/agent_core/skills/skill_content_reader.py
src/agent_core/skills/skill_activation_service.py
src/agent_core/agents/skill_middleware.py
src/agent_core/tools/registry/skill_hot_reload.py
src/agent_core/agents/subagent_profiles.py
~~~

建议将 skill_content_reader.py 重命名为 skill_content_repository.py，体现纯读取职责。

### 调整

~~~text
src/main.py
src/agent_core/agents/lead_agent.py
src/agent_core/tools/registry/providers.py
src/agent_core/tools/registry/__init__.py
src/agent_core/skills/__init__.py
src/agent_core/tools/database_tool.py
src/agent_core/tools/rag_service.py
src/agent_core/agents/chat_pipeline.py
~~~

### 候选新增模块

~~~text
src/agent_core/tools/database_query_tool.py
src/agent_core/tools/knowledge_search_tool.py
src/agent_core/skills/skill_content_repository.py
~~~

## 15. 分阶段实施

### 阶段 0：固定基线

1. 审阅当前未提交修改；
2. 运行现有 Skill、ToolRegistry、Lead Agent、Sub-Agent、RAG 和 Database 测试；
3. 记录工具列表、Registry revision 和两个业务 Tool 的输出契约。

### 阶段 1：建立纯 Skill 模型

1. 收敛 SkillDefinition；
2. Loader 对废弃字段告警而不解释；
3. ActivationService 移除 SkillKind 分支；
4. ContentRepository 只读指令和指定资源；
5. 增加路径越界、大文件和二进制资源测试。

### 阶段 2：Middleware 接管

1. SkillMiddleware.tools 提供 load_skill 和 read_skill_resource；
2. 从 ToolRegistry 删除 skill.load_skill；
3. Lead Agent 继续通过 Middleware 获得 Skill；
4. 需要 Skill 的 Sub-Agent 显式挂载同一 Middleware；
5. 删除 create_load_skill_tool 的自包含执行分支。

### 阶段 3：迁移业务 Tool

1. 迁移 query_database；
2. 迁移 search_knowledge_base；
3. 独立注册 ToolDefinition；
4. 验证 Runtime Context；
5. 验证 RAG 引用和 metadata；
6. 删除旧 scripts/main.py 业务入口。

### 阶段 4：拆分混合 Skill

1. query-database 迁移为 database-analysis Skill；
2. search-knowledge-base 迁移为 knowledge-base-answering Skill；
3. Skill 正文只描述何时和如何使用对应 Tool；
4. Skill 名与 Tool 名进入不同命名空间。

### 阶段 5：简化热重载

1. 移除 SkillHotReloader 到 ToolRegistry 的依赖；
2. 实现候选 Registry 校验和原子替换；
3. 监听 Skill 资源目录；
4. 验证 Skill 变更不改变 ToolRegistry revision。

### 阶段 6：删除旧链路

1. 删除 SkillToolFactory；
2. 删除 SkillToolProvider；
3. 删除 SkillKind；
4. 删除 Skill Script 自动执行协议；
5. 删除 required_secrets 和 runtime_context_keys；
6. 清理过时测试、注释和设计文档；
7. 全量回归。

## 16. 测试矩阵

### Skill Loader / Registry

- 合法 Skill 只从纯 Skill 元数据构建；
- 废弃字段不改变执行语义；
- 单 Skill YAML 错误不影响其他 Skill；
- 重名 Skill 显式拒绝；
- Skill 名与 Tool 名相同不触发 ToolRegistry 冲突。

### Middleware Catalog

- 只注入当前 Agent 可见 Skill；
- Catalog 只包含 name 和 description；
- 已自动激活的 Skill 不再重复出现在待选择 Catalog 中；
- 模糊候选才进入 Catalog，高置信匹配直接激活；
- 无 Skill 时不注入空块；
- 热重载后下一次调用看到新 Catalog；
- Catalog 顺序稳定。

### Skill Router

- 用户/API 显式指定 Skill 时强制进入 forced；
- Agent required_skills 在第一次主模型调用前完成激活；
- 高置信必需 Skill 自动进入 auto_activate；
- 仅语义相关但任务不需要的 Skill 不得自动激活；
- 模糊候选只进入 catalog_candidates；
- explicit_only Skill 不得由语义路由自动激活；
- 无权限 Skill 不得出现在自动激活结果和可见 Catalog；
- 多 Skill 任务可以稳定返回多个激活项并保持确定顺序；
- 路由器失败时采用可诊断降级策略，不得默认全量加载；
- 路由结果包含 score、reason 和 activation_source；
- 同一 Skill 在一次运行中不会重复注入正文。

### Activation

- forced 和 auto_activate Skill 在第一次主模型调用前已经生效；
- load_skill 返回正文和资源索引；
- load_skill 可以补充激活 catalog_candidates 中的 Skill；
- load_skill 激活已激活 Skill 时不重复注入正文；
- 不自动拼接全部 references；
- 不存在和无权访问使用不泄露措辞；
- Guardrail 拒绝时不读取正文；
- 不调用 ToolRegistry 或 SandboxProvider。

### 路由评测

每个 Skill 必须维护 should_activate、should_not_activate 和 ambiguous 三类路由样本。至少统计：

~~~text
Recall
Precision
False Negative
False Positive
平均自动激活 Skill 数量
平均 Catalog 候选数量
平均新增上下文 Token
路由延迟
load_skill 补充激活率
重复激活率
~~~

对于必须遵循固定流程的 Skill，优先保证 Recall；对于正文或资源很大的 Skill，同时设置 Precision 和上下文预算门槛。阈值调整必须基于评测结果，不能仅修改 Prompt 后主观判断。

### Resource access

- 按需读取 reference、script 和 template；
- 拒绝绝对路径、父目录和符号链接逃逸；
- 限制超大文件；
- 二进制文件不直接注入上下文；
- 读取资源不会执行 script。

### Business Tools

- query_database 使用可信 runtime.context.datasource_id；
- search_knowledge_base 使用正确的工作流和数据库上下文；
- 两个 Tool 在 ToolRegistry 各只有一条；
- 两个 Tool 都不依赖 SkillDefinition；
- RAG 引用、数据库图表和失败降级契约保持不变。

### ToolRegistry 解耦

- 不存在 source_type="skill"；
- 不存在 canonical_name="skill.*"；
- 修改或增删 Skill 不改变 ToolRegistry revision；
- Skill 热重载与 Tool 热重载互不影响。

### Lead/Sub-Agent

- 需要 Skill 的 Agent 统一挂载 Middleware；
- 不再存在 Sub-Agent 专用 load_skill 执行分支；
- 不允许 Skill 的 Agent 不能通过其他入口激活 Skill；
- Skill Catalog 按 Agent 范围隔离。

## 17. 观测与审计

Skill 事件与 Tool 事件必须分开。

建议 Skill 事件：

~~~text
skill.catalog.injected
skill.routing.started
skill.routing.completed
skill.routing.failed
skill.activation.requested
skill.activation.allowed
skill.activation.denied
skill.activation.failed
skill.activation.auto
skill.activation.explicit
skill.activation.required
skill.activation.supplemental
skill.activation.duplicate_skipped
skill.resource.read
skill.resource.denied
skill.registry.reloaded
~~~

Tool 事件继续使用：

~~~text
tool.call.started
tool.call.completed
tool.call.failed
tool.call.denied
~~~

load_skill 不应被统计为数据库查询或普通业务 Tool 调用。

## 18. 非目标

本次不应顺带：

- 重写 RAG 算法；
- 重写数据库问数服务；
- 更换模型或 Provider；
- 重写其他 ToolRegistry Provider；
- 改变 REST/WebSocket 对外协议；
- 改变引用格式或图表格式；
- 修复与本次无关的工作树修改。

## 19. Definition of Done

只有同时满足以下条件才算完成：

1. 任意 Skill 都不生成独立 StructuredTool；
2. SkillToolFactory 已删除；
3. SkillToolProvider 已删除；
4. SkillKind 已删除；
5. SkillDefinition 不含业务参数、密钥和运行时上下文；
6. Skill 系统不依赖 SandboxProvider；
7. Skill 系统不依赖 ToolRegistry；
8. SkillMiddleware 是唯一 Skill 激活入口；
9. query_database 是独立业务 Tool；
10. search_knowledge_base 是独立业务 Tool；
11. Skill scripts 不会被 Skill 框架自动执行；
12. Skill references 支持按需加载；
13. 加载 Skill 不会赋予额外 Tool 权限；
14. 修改 Skill 不改变 ToolRegistry revision；
15. Lead Agent 和需要 Skill 的 Sub-Agent 使用同一 Middleware 链路；
16. RAG 引用、数据库查询、图表和流式输出契约未回归；
17. 所有新增和相关回归测试通过；
18. 代码库中不再存在“参数化 Skill = Tool”的过时说明。

## 20. 给实现者的执行要求

1. 先阅读本文档、旧 Skill 重构文档和相关测试；
2. 先保护当前未提交修改；
3. 按阶段实施，每阶段运行相关测试；
4. 不允许只做类名重命名或把 SkillToolFactory 换个位置；
5. 不允许保留“parameters 非空时仍生成 Tool”的隐式兼容分支；
6. 不允许为了通过测试把 Skill 执行逻辑迁移到另一个未命名模块；
7. 业务 Tool 迁移必须保持当前外部契约；
8. 完成后提交改动清单、架构对比、测试结果、剩余风险和未实现项。

## 21. 一句话结论

~~~text
Skill 只负责告诉模型“如何完成任务”；
Tool 只负责在受控边界内“执行一个能力”。
Skill 可以引用 Tool，但永远不生成、包装或绕过 Tool。
~~~
