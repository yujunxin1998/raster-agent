# Agent Loop 工程设计方案

> 讨论稿 · 2026-08-04
> 范围：仅涉及 `diit-agent-server` 的重构；`diit-agent-web` 保持不变，本方案的一切改动以"前端零改动"为硬约束。
> 参考：ByteDance `deer-flow`（DeerFlow 2.0）开源 harness 的架构设计思路。
> 落地状态：第一期（skill / memory / tool / prompt 机制迁移 + guardrail / sandbox / workspace / 中间件流水线
> 骨架 / `required_secrets`）已在本仓库（`raster-agent-server`）全部完成。第二期——4.1 节"Lead Agent +
> 委派工具"——也已全部完成：`agent_core/agents/lead_agent.py` 用 `langchain.agents.create_agent` 组装 Lead
> Agent，`agent_core/loop.py::build_middlewares()` 的 11 个中间件真正接入了这个 Agent；路由方式后来又从
> "三个固定委派工具"演进成了通用 `task(subagent_type, task)` 分发（`agent_core/agents/delegation_tools.py` +
> `subagent_profiles.py` 注册表，取代早期的 `delegate_to_rag_agent`/`delegate_to_web_search_agent`/
> `delegate_to_database_agent`）——目前只有 `web-researcher` 一种 subagent 走这套委派机制，
> `search_knowledge_base`/`query_database` 落地成了 Lead Agent 自己的技能（不再是委派工具），
> `general_agent`/`tool_agent` 的工具同样直接并入 Lead Agent 自己的基础工具集（详见 4.1 节"落地说明"）；
> `datasource_id` 强制路由有了代码层面的硬校验（`DatasourceRoutingMiddleware`）；Redis-based Eval 遥测
> （`agent_core/eval/`）已迁移（去掉了旧 Supervisor 多节点图特有的"agent 切换检测"逻辑，简化为"Lead Agent
> 本轮"单一整体记录，见该模块 docstring 说明）；`POST /chat/`（非流式）、`/conversations/*`、WebSocket
> `/ws/chat`（流式 + 前端工具双通道回环）均已可用。第三期的 `DockerSandboxProvider`/记忆 staleness 定期复核/
> `outputs/` 产物下载链接/checkpoint 孤儿数据清理均已落地，唯一仍未迁移的是 `user_id` 身份认证，详见第七节。

---

## 一、为什么要重做

`diit-agent-server` 现在的骨架是"Supervisor 多图路由"：一个 LangGraph `StateGraph` 里挂 5 个 `create_react_agent` 子节点，`supervisor` 节点每轮用 LLM 输出 JSON 决定路由到哪个子节点，子节点跑完再绕回 `supervisor` 重新决策，直到 `FINISH`。这套东西现在能跑，但存在三个结构性问题，是本次重做要解决的：

1. **没有统一的"运行时外壳"**。权限校验、审计、记忆注入、记忆压缩、标题生成，这些横切关注点分散在 `chat_service.py` 里用手写的 `if/await/asyncio.create_task` 拼起来（详见 `docs/功能说明/记忆功能模块设计.md` 第三节的数据流图）。每加一个新的横切能力，就要在 `chat_service.py` 里再插一段代码，长期会失控。
2. **没有文件系统和沙箱**。`skills/public/` 下 23 个技能里，`ppt-generation`、`video-generation`、`chart-visualization` 这些技能的设计初衷是给"具备文件系统沙箱的 Claude Code"用的，但当前后端只是裸 `subprocess.Popen` 跑脚本（`src/core/skills/content_reader.py::_run_script_blocking`），没有隔离的工作目录、没有产物落盘的地方、Windows 下连内存限制都做不到——这一点在 Skill 机制设计文档"待讨论事项"里已经明确写出来了，是当前架构补不上的缺口。
3. **没有权限控制**。架构设计文档"安全性现状"一节写得很直白："无认证授权机制"、"`user_id` 由前端直接传入，服务端不做身份校验"。工具执行层面也一样，`SkillAccessGuard` 只做了"用户是否关闭了某个技能"这一层最简单的开关，没有更细粒度的"这个工具/这次调用是否被允许执行"的统一拦截点。

这三个问题不是加几个模块能解决的，需要把"整个请求怎么跑"的骨架换掉——这就是要单独立项做"Agent Loop 工程"的原因，而不是继续在 `supervisor.py` 上打补丁。

---

## 二、DeerFlow 2.0 给到的参考，以及哪些能抄、哪些不能抄

DeerFlow（`bytedance/deer-flow`）是同样基于 LangGraph 的 harness，规模比我们大一个数量级（`backend/AGENTS.md` 一份文档就有上千行），照抄整套不现实，但它有几个设计决策值得直接拿来用：

| DeerFlow 的做法 | 值得借鉴的点 | 我们怎么落地（详见第四、五节） |
|---|---|---|
| `packages/harness/` 与 `app/` 严格分层，`harness` 是可发布的独立包，`app` 只做 Gateway/渠道接入 | 核心 Agent 能力和"跑在 FastAPI 里"这件事解耦，方便未来独立测试、甚至独立复用到其它渠道 | `src/` 拆成 `agent_core/`（对应 harness）+ `api/`/`service/`（对应 app），见第六节目录结构 |
| Lead Agent 不是靠"多图路由"，而是**单一 Agent + Middleware Pipeline**，中间件按固定顺序包裹 `wrap_model_call`/`wrap_tool_call` | 把权限校验、沙箱获取、记忆注入、审计这些横切逻辑从 `chat_service.py` 的手写胶水代码里挪出来，变成可插拔、可测试、有固定顺序保证的中间件链 | 第四节"Agent Loop 核心" |
| `GuardrailMiddleware` + 可插拔 `GuardrailProvider`（内置 `AllowlistProvider`，也可接第三方策略引擎），工具调用前置授权，拒绝时返回一条 error `ToolMessage` 而不是抛异常中断整轮 | "失败也要让 Agent 继续推理"这个容错哲学，和现在 `SkillContentReader` 遇错不抛异常、`custom_tool_converter` 遇错返回 JSON 字符串的思路是一致的，权限模块也应该照此设计 | 第五节 5.1 |
| `ThreadDataMiddleware` 给每个 thread 建立隔离目录 `users/{user_id}/threads/{thread_id}/user-data/{workspace,uploads,outputs}`，工具里统一用虚拟路径 `/mnt/user-data/...` 访问，由 Sandbox 层做真实路径映射 | 我们的 `conversation_id` 天然就是 thread_id，直接复用这个粒度做隔离目录 | 第五节 5.2 |
| `Sandbox` 抽象接口（`execute_command`/`read_file`/`write_file`/`list_dir`）+ `SandboxProvider`（Local/Docker/远程可插拔），本地模式零依赖，容器模式按需开启 | 我们不需要一上来就上 Docker/K8s，但接口要照这个形状设计，方便以后从"本地进程"平滑升级到"容器隔离" | 第五节 5.3 |
| Skill 的"渐进式披露"再往前一步：`deferred_discovery` 模式下系统提示里只放技能名索引，Agent 主动调用 `describe_skill` 才拿到完整元数据，进一步省 prompt token；`required-secrets` 让技能声明自己需要哪些密钥，由调用方按请求传入，绝不进 prompt/日志/checkpoint | 我们现有的两级披露（frontmatter 目录 → 正文/脚本）已经是同一思路的简化版，暂不需要再加一层 `describe_skill`，但 `required-secrets` 这个"密钥按需注入、绝不落地"的设计值得直接抄，解决脚本技能要连外部系统时明文写 `.env` 的问题 | 第五节 5.4 |
| `ReadBeforeWriteMiddleware`：写文件前必须先读过、且内容哈希没变过，否则拦截 | 防止 Agent 没看清文件当前内容就覆盖写，属于沙箱文件工具的标配安全网，成本很低 | 第五节 5.3（尚未实现，仍是待办） |
| `SubagentExecutor` + `task()` 工具：Lead Agent 按需把子任务派给专用 subagent，而不是像 Supervisor 那样"每轮都必须回到主控节点重新决策" | 能显著减少路由跳转次数、降低 supervisor 决策失败（当前 `supervisor.py` 里大段代码在处理 DeepSeek JSON 输出退化、自然语言兜底解析）带来的不稳定性 | 第四节（已落地为 `delegation_tools.py::build_task_tool()` + `subagent_profiles.py`） |
| 记忆模块的 staleness pass（老记忆定期复核是否该淘汰）、冲突检测 | 我们现有记忆模块已经有 `importance`/`status`（pending/active/archived）字段，补一个后台定期复核任务成本不高 | staleness pass 已落地（`memory_staleness_reviewer.py`，见第七节路线图第三期）；冲突检测本次未覆盖，仍是待办 |

**不抄的部分**：DeerFlow 的多渠道接入（Feishu/Slack/Discord/GitHub webhook）、K8s Provisioner、BoxLite micro-VM、`.skill` ZIP 安装市场——这些是它作为通用开源产品要覆盖的场景，我们是单一业务系统，不需要这个复杂度。抄的是"骨架形状"，不是"全部功能"。

---

## 三、现状盘点：三大机制哪些原样保留、哪些要动

### 3.1 Skill 机制 —— 保留全部设计，只改"执行"这一环

`src/core/skills/` 六个类（`SkillDefinition` / `SkillLoader` / `SkillRegistry` / `SkillContentReader` / `SkillToolFactory` / `SkillManager`）构成的"渐进式披露"架构已经很干净，不需要重写：

- 启动阶段只解析 frontmatter（`SkillLoader`），运行时才读正文/跑脚本（`SkillContentReader`），这个分层保留。
- `SkillRegistry` 按 `category` 分类、`SkillToolFactory` 动态转 `StructuredTool`，保留。
- `SkillAccessGuard` 的"用户关闭技能→拦截执行"保留，但要**升级为新权限中间件的一个特例**（见 5.1），而不是独立散落的一处校验。

唯一要动的是 `SkillContentReader.run_script()` 里的执行方式：现在是裸 `subprocess.Popen`（`_run_script_blocking`），只在 POSIX 上有 `RLIMIT_AS` 内存限制，Windows 下毫无隔离；执行完全在服务端进程的当前工作目录下进行，没有产物存放的地方。这一环要挪到新的 Sandbox 抽象层后面执行（见 5.3），脚本协议（stdin JSON / stdout `{"content","__metadata__"}`）完全不变，`skills/core/search-knowledge-base/scripts/main.py` 等现有脚本不用改一行。

### 3.2 Memory 机制 —— 保留全部设计，接入点从"手写胶水"改为"中间件"

三层记忆体系（短期 checkpointer / 长期 Elasticsearch 向量库 / 压缩）、`BaseMemoryStore` 抽象接口、`extractor.py` 后处理提取、`compressor.py` 压缩、`sensitive_filter.py` 敏感信息拦截、`audit.py` 审计日志——这套设计成熟且已经过验证用例覆盖（记忆功能模块设计文档"十、验证用例"），全部原样保留，不动内部实现。

要动的只是**接入方式**：现在 `chat_service.py` 里手写"对话前调 `get_relevant_context`、对话后 `asyncio.create_task` 并发跑 `extract_after_chat`/`compress_after_chat`"，这段胶水代码挪成两个中间件——`MemoryInjectionMiddleware`（`wrap_model_call` 前置注入）和 `MemoryExtractionMiddleware`（收尾阶段，仍然是 fire-and-forget 异步任务，语义不变）。这样做的好处是：新增一种记忆触发方式（比如 DeerFlow 的 staleness 复核）只需要新加一个中间件，不用再去改 `chat_service.py` 的主流程代码。

### 3.3 工具机制 —— 后端工具/前端工具的二元划分保留，新增"沙箱工具"第三类

`src/core/tools/` 现有的 `tool_filter`（内部工具黑名单）、`custom_tool_converter`（前端工具→`StructuredTool` 动态转换）、`search_tool`/`database_tool`/`memory_tools` 全部保留。`docs/功能说明/前端工具与后端工具结合设计.md` 里描述的"执行通道"和"展示/审计通道"两条独立消息通道机制，是 WS 协议的核心部分，前端 `AgentApi.js`/`ChatPanel.vue` 依赖这个协议，**不能动**。

新增的是**沙箱工具**这一类：凡是需要读写文件、跑代码/脚本的工具（包括 skill 里的脚本型技能），统一经过 Sandbox 抽象层执行，获得虚拟路径隔离、资源限制、执行审计。三类工具的边界：

| 工具类型 | 执行位置 | 是否经过 Guardrail | 是否经过 Sandbox |
|---|---|---|---|
| 后端工具（`web_search`/`knowledge_base_search`/`query_database`/`save_memory`/`recall_memory`） | 服务端进程内直接调用 | 是 | 否（不涉及文件/代码执行） |
| 前端工具（`extra_tools`，经 WebSocket 回客户端执行） | 浏览器/客户端 | 是（调用前拦截） | 否（沙箱只管服务端侧资源） |
| 沙箱工具（技能脚本、未来的代码执行/文件操作工具） | 服务端 Sandbox 内 | 是 | 是 |

### 3.4 Prompt 机制 —— 保留全部模板，新增 PromptFactory 封装

`src/core/prompts/` 的 15 条提示词模板（`SUPERVISOR`/`GENERAL_AGENT`/`RAG_QUERY_REWRITE` 等 `.md` 文件）原样保留，`PromptLoader` 扫描 + `PromptRegistry` 注册的两段式加载流程也保留。

**改动点**：原项目的 `PromptRegistry` 只做"存 + 取原文"，凡是需要拼变量的模板（如 `rag_query_rewrite.md` 里的 `{history_text}`/`{tool_query}`/`{user_query}`）都由调用方（`rag_query_rewrite_service.py` 等）各自手写 `.format(...)`，拼错变量名、漏传变量只能等运行时 `KeyError` 才暴露；`supervisor.py` 里的 `datasource_note`/`memory_note` 拼接也是类似的手工字符串处理。新增的 `PromptFactory` 把"渲染"这个动作收口成统一入口：

```python
class PromptFactory:
    def get(self, name: str) -> str: ...                    # 取静态模板原文，不渲染
    def render(self, name: str, **variables) -> str: ...     # 按占位符渲染，缺变量时抛 PromptRenderError（而非裸 KeyError）
```

原项目里 `PromptRegistry.__getattr__` 支持的属性式访问（`prompts.SUPERVISOR`）在新版本里去掉了，统一改为显式方法调用（`prompt_factory.get("SUPERVISOR")`）——魔法属性对 IDE 补全、静态类型检查不友好，也和"避免隐式行为"的代码规范相悖，显式方法调用不会更啰嗦。

Prompt 机制不依赖数据库/沙箱/权限等基础设施，因此和原项目一样在模块导入时即完成加载，不需要像 `SkillManager`/`MemoryManager` 那样在 `main.py` 的 lifespan 里显式初始化。

---

## 四、Agent Loop 核心设计

### 4.1 从"Supervisor 多图"到"单 Agent + 中间件流水线"

现在的 `supervisor.py` 有大量代码在跟"LLM 结构化输出不稳定"搏斗——DeepSeek `json_mode` 概率性返回空白 content、需要自然语言兜底解析、需要死循环检测（同一 agent 连续被选中 3 次强制 `FINISH`）。这些问题的根源是"每轮都要额外调一次 LLM 专门做路由决策"，路由决策本身也会出错。

新设计把"路由"从"每轮必须做一次的独立 LLM 调用"降级为**工具调用**：保留 `general_agent`/`rag_agent`/`web_search_agent`/`database_agent`/`tool_agent` 五个专用能力，但不再是 supervisor 图上的节点，而是重新封装成 Lead Agent 可以调用的 `delegate_to_xxx` 工具（对应 DeerFlow 的 `task()` 工具）。Lead Agent 自己就是一个 `create_react_agent`，工具集里既有直接可用的工具（`web_search`、`save_memory`……），也有"委派"工具。这样做的直接好处：

- 简单问题（闲聊、直接能回答的问题）不需要"先转一圈 supervisor 决策再回来"，Lead Agent 自己直接答，省一次 LLM 调用。
- 需要专用能力时，走 `delegate_to_database_agent` 这类工具调用，天然带 `tool_call`/`tool_response` 事件，前端不用改事件处理逻辑（`astream_events` 的 `on_tool_start`/`on_tool_end` 本来就在推给前端展示）。
- `datasource_id` 强制路由这类业务规则，现在是 supervisor prompt 里的一段硬编码文案，未来通过 Guardrail 的"前置校验"实现：检测到 `datasource_id` 但 Lead Agent 第一步没有调用 `delegate_to_database_agent`，直接在中间件层拦下来注入纠正提示，语义更清晰，也更容易写单测。
- 死循环检测这类"防御性代码"不再是 supervisor 专属，而是变成一个通用的 `LoopDetectionMiddleware`（检测连续相同 tool_call），可以顺带保护所有工具，不只是路由这一种循环。

`thinking_agent`（深度思考模式）目前是完全独立构建、携带全量工具集的另一套逻辑，新架构下它退化成"Lead Agent 的一种运行时配置"（`thinking_enabled=True` 时开启扩展推理 + 更大的 `recursion_limit`），不再是并行维护的第二套 Agent 构建代码，减少一份重复逻辑。

`tool_agent`（承载前端 `extra_tools`）也同理：不再是 supervisor 图上的独立节点，而是 Lead Agent 工具集里按请求动态 merge 进来的一部分（这一点和现在 `tool_agent.py::_dispatch()` 里"每次请求都要重新 `create_react_agent`"的实现方式是一致的，不需要改，只是不再需要专门用一个图节点去承载它）。

> **落地说明**（第二期，已演进两轮）：`agent_core/agents/lead_agent.py::build_lead_agent()` 用本仓库
> `langchain==1.3.9` 自带的 `langchain.agents.create_agent`（不是原项目的 `create_react_agent`）
> 组装 Lead Agent，`middleware=build_middlewares(...)` 直接把第一期的 11 个中间件接进这个 Agent。
> 第一轮先做了 `delegate_to_rag_agent`/`delegate_to_web_search_agent` 两个固定委派工具；第二轮把这套
> "每个能力一个固定工具名"的设计换成了通用 `task(subagent_type, task)` 分发（`agent_core/agents/
> delegation_tools.py::build_task_tool()` + `subagent_profiles.py` 注册表），对应 DeerFlow 的 `task()`
> 工具思路——新增一个专用能力只需要在 `subagent_profiles.py` 的 `_PROFILES` 里加一条 `SubagentProfile`，
> 不需要再写一遍"新建 builder 函数 + 接进 `lead_agent.py` 的 `base_tools` + 写新提示词"。目前
> `_PROFILES` 里只注册了 `web-researcher` 一种（对应原来的 `delegate_to_web_search_agent`）；
> `rag_agent`（原 `delegate_to_rag_agent`）与 `database_agent`（原计划的 `delegate_to_database_agent`）
> 都不再走委派机制，改造成 Lead Agent 自己的技能——`search_knowledge_base`/`query_database`（技能分类
> `rag`/`database`），`db_query_service.py`/ECharts 集成也已随 `query_database` 技能落地。`general_agent`
> （`save_memory`/`recall_memory` + skill 分类 `general`）与 `tool_agent`（skill 分类 `tool` + 前端
> `extra_tools`）的工具**直接并入 Lead Agent 自己的基础工具集**，不设对应的委派工具，因为这两者和
> "Lead Agent 自己直接处理"没有本质区别，符合本节"简单问题不需要转一圈"的初衷。`datasource_id` 强制
> 路由的硬校验已通过 `DatasourceRoutingMiddleware` 落地（检测到已配置数据源但模型第一步没调用
> `query_database` 时注入纠正提示强制重试）。`thinking_enabled` 运行时配置、`tool_agent` 工具直接合并
> 这两点同样已按本节设计落地。

> **是否要一步做完**：这是本方案里改动面最大的一步，建议放在路线图第二期（见第七节），第一期先把中间件流水线骨架、权限、文件系统、沙箱做出来，Supervisor 的路由图先保留、只是把它包装成"Lead Agent 众多委派工具之一"的过渡态，降低一次性重构的风险。

### 4.2 中间件流水线设计

参考 DeerFlow 的顺序设计原则——**输入清洗在最外层、资源获取先于业务逻辑、审计先于执行、错误处理包裹执行、收尾类中间件在最后**——给出适合当前项目体量的裁剪版顺序：

| 顺序 | 中间件 | 对应现有代码 | 说明 |
|---|---|---|---|
| 1 | `InputSanitizationMiddleware` | 无（新增） | 统一清洗用户输入，为后续所有中间件提供干净的消息 |
| 2 | `ThreadDataMiddleware` | 无（新增） | 按 `conversation_id` 建立/复用隔离目录，写入 state，见 5.2 |
| 3 | `MemoryInjectionMiddleware` | `chat_service.py::_build_memory_context_message` | 检索长期记忆，构造临时 SystemMessage 注入本轮，不写入持久化历史（沿用现有做法） |
| 4 | `GuardrailMiddleware` | `SkillAccessGuard` 的思路升级版 | 工具调用前置权限校验，见 5.1 |
| 5 | `SandboxMiddleware` | 无（新增） | 按需获取 Sandbox 实例，供沙箱工具使用，见 5.3 |
| 6 | `ToolAuditMiddleware` | `EvalEventEmitter` 的 `eval:tool` 埋点 | 现有监控埋点原样保留，只是统一收口成中间件形式 |
| 7 | `ToolErrorHandlingMiddleware` | 现有各处 `try/except` 兜底（`custom_tool_converter`/`SkillContentReader`） | 把分散的"工具失败不中断对话"逻辑收口到一处，工具异常统一转成 error `ToolMessage` |
| 8 | `LoopDetectionMiddleware` | `supervisor.py` 里的死循环检测 | 从 supervisor 专属逻辑升级为通用中间件 |
| 9 | `SummarizationMiddleware`（对应现有"压缩"） | `src/core/memory/compressor.py` | 触发条件、压缩逻辑完全不变，只是调用时机从 `chat_service.py` 手写 `asyncio.create_task` 改为中间件收尾钩子 |
| 10 | `TitleMiddleware` | `src/conversation/title_generator.py` | 首轮结束后异步生成标题，逻辑不变 |
| 11 | `MemoryExtractionMiddleware` | `src/core/memory/extractor.py` | 后处理记忆提取，逻辑不变，仍是 fire-and-forget |

这条流水线的关键约束：**第 4 步 Guardrail 和第 7 步 ToolErrorHandling 都包裹在"工具调用"这个环节外层，而不是包裹整个对话轮次**——也就是说，权限拒绝、工具报错都只影响单次工具调用，Agent 拿到错误信息后可以决定要不要换个方式继续，不会导致整轮对话直接失败。这个容错哲学和现在 `SkillContentReader`/`CustomToolConverter` 已经在践行的原则完全一致，是新旧代码之间最重要的一条设计延续性。

> **落地说明**：11 个中间件已在 `src/agent_core/middlewares/` 落地为真实的 `langchain.agents.middleware.AgentMiddleware` 子类（本仓库 `langchain==1.3.9` 自带该框架，与 DeerFlow 对齐的中间件写法在这个版本里是官方原生能力），`src/agent_core/loop.py::build_middlewares()` 按本表顺序组装成列表。`SummarizationMiddleware` 依赖 `MemoryCompressor` 新增的 graph-free 方法 `compress_messages()`（不再要求一个真实的 LangGraph 编译图）；`TitleMiddleware` 因为原 `src/conversation/title_generator.py` 未随基础设施迁移，改为自包含实现（复用 `MemoryCompressor`/`MemoryExtractor` 的 LLM 调用方式 + 新增的 `TITLE_GENERATION` 提示词模板），持久化通过可选回调交给二期 conversation 层接入。**这仍然只是骨架**：本仓库没有编排层，`build_middlewares()` 返回的列表还没有被传给任何真实的 `create_agent(...)` 调用，`main.py` 不需要因此改动。配套单测见 `test/middlewares/`、`test/memory/test_memory_compressor_messages.py`。

---

## 五、三个新增机制的具体设计

### 5.1 权限控制（Permission / Guardrail）

**目标**：把现在散落的"技能开关"（`SkillAccessGuard` 查 `user_skill_settings` 表）和"完全没有的工具级权限"统一成一层。

**接口设计**：

```python
class GuardrailProvider(Protocol):
    async def check(
        self, *, user_id: str | None, tool_name: str, tool_args: dict,
        context: dict,  # 含 conversation_id / thinking / datasource_id 等
    ) -> GuardrailDecision:  # allow / deny(reason) / require_confirmation(reason)
        ...
```

**内置实现（一期只做这一个，够用）**：`AllowlistGuardrailProvider`——组合两张表：

- 复用现有 `user_skill_settings` 表（技能级开关，字段/语义不变）。
- 新增 `user_tool_permissions` 表（`user_id`, `tool_name`, `scope`, `granted`），管理非技能类工具（如未来的"代码执行"、"文件删除"这类高风险沙箱工具）的授权。

拒绝时的行为完全照抄 `SkillToolFactory._invoke` 现在的模式：不抛异常，返回一段说明文本给 Agent（"工具 X 未被授权执行"），Agent 据此继续推理或告知用户——这一点和 DeerFlow `GuardrailMiddleware` "deny 时返回 error ToolMessage 而不是中断"的设计完全对齐，也是现有代码已经在用的模式，零学习成本。

**预留扩展点**：`GuardrailProvider` 做成可插拔接口，是为了以后接入更复杂的策略（比如按角色的 RBAC、按数据源的行级权限）时不用改中间件本身，只需要换一个 Provider 实现——这一点直接照抄 DeerFlow "内置 `AllowlistProvider`，也可接第三方策略引擎"的做法。

**和"待讨论事项"里 `user_id` 鉴权问题的关系**：架构文档和记忆模块文档都提到"`user_id` 由前端直接传入，服务端不做身份校验"这个已知缺口。Guardrail 层不解决身份认证（这是另一个独立的问题，需要接入统一鉴权中间件，从 token/session 解析 `user_id`），但 Guardrail 的所有校验都以"信任 `user_id`"为前提在做授权，如果身份认证问题不解决，权限控制本身的意义会打折扣——建议这两件事同期推进（详见第七节路线图）。

### 5.2 文件系统（Per-Thread 虚拟工作区）

**目标**：给每个对话一个隔离的、有生命周期的文件区域，供沙箱工具和脚本型技能落地产物（图表图片、生成的 PPT/文档、下载的文件等），解决"技能脚本执行完产物没地方放"的问题。

**目录结构**（照抄 DeerFlow 的三段式，路径按 `conversation_id` 隔离）：

```
{DATA_ROOT}/users/{user_id}/threads/{conversation_id}/
  ├── workspace/   # 工作区：Agent 读写的中间文件
  ├── uploads/     # 用户上传的文件（对接现有多模态消息支持中的文件上传能力）
  └── outputs/     # 最终产物：图表、生成的文档，供前端下载链接引用
```

**虚拟路径约定**：沙箱工具和技能脚本统一用 `/workspace/...`、`/uploads/...`、`/outputs/...` 这类相对虚拟路径寻址，不直接暴露宿主机绝对路径；由 `ThreadDataMiddleware` 在 state 里注入当次请求的真实路径映射，`Sandbox.read_file`/`write_file` 内部做转换。这样做两个好处：一是脚本代码里不会硬编码宿主机路径，换部署环境不用改代码；二是天然做到"这个技能只能碰自己 thread 的文件，碰不到别的对话/别的用户的文件"。

**路径穿越防护**：直接复用现有 `SkillFileTreeReader._resolve()` 里"解析绝对路径后校验没有越出 `skill_dir`"的做法（`src/core/skills/file_tree.py`），同一套校验逻辑用在虚拟工作区的路径解析上。

**生命周期**：`outputs/` 下的产物需要有对应的静态文件服务/下载接口暴露给前端（前端目前没有这类展示位，需要和 `MessageItem.vue` 配合新增，这一点超出"零改动 web"的约束，需要和前端团队单独沟通，不在本次后端方案强制范围内，先按"生成 URL 挂进现有 Markdown 正文"的方式兼容，类似现在 `database_agent` 返回的 `<echart>` 图表块）。

> **落地说明**（第三期）：已实现。新增 `save_output_file` 工具（`agent_core/tools/sandbox_tool.py`），
> 把内容写入 `outputs/` 目录并返回一条相对路径下载链接；`conversation_router.py` 新增
> `GET /conversations/{conversation_id}/outputs/{file_path}` 接口，按 `user_id` 做 owner 校验后用
> `FileResponse` 返回文件——完全按本节设想的"生成 URL 挂进 Markdown 正文"方式落地，没有新增
> `PUBLIC_BASE_URL` 之类的配置，也不需要前端改动。

### 5.3 沙箱（Sandbox 执行环境）

**目标**：技能脚本、未来的代码执行类工具，统一经过这一层执行，取代裸 `subprocess.Popen`。

**接口设计**（照抄 DeerFlow 的最小接口集）：

```python
class Sandbox(Protocol):
    async def execute_command(self, command: str, *, cwd: str | None = None,
                               env: dict | None = None, timeout: float = 60) -> CommandResult: ...
    async def read_file(self, path: str) -> str: ...
    async def write_file(self, path: str, content: str, mode: str = "overwrite") -> None: ...
    async def list_dir(self, path: str) -> list[dict]: ...

class SandboxProvider(Protocol):
    async def acquire(self, conversation_id: str) -> Sandbox: ...
    async def release(self, sandbox: Sandbox) -> None: ...
```

**一期只做 `LocalSandboxProvider`**：本质是把现有 `_run_script_blocking` 的能力重新包一层接口，行为不变（子进程 + `asyncio.to_thread` 调度，POSIX 下保留 `RLIMIT_AS` 内存限制），但补两件事：

1. 执行时的 `cwd` 固定为该 thread 的 `workspace/` 目录（对接 5.2），脚本/命令里的相对路径天然落在隔离目录里。
2. 执行前的环境变量按 DeerFlow `env_policy.build_sandbox_env` 的思路做一次清洗——不把服务端进程的完整 `os.environ` 传给子进程，过滤掉 `*_KEY`/`*_SECRET`/`*_TOKEN`/`*_PASSWORD`/`DATABASE_URL` 这类敏感变量，需要的密钥通过技能 frontmatter 新增的 `required_secrets` 字段显式声明、按请求注入（对应第二节表格里提到的"密钥按需注入"设计）。这一步直接堵住"技能脚本子进程意外拿到数据库连接串/LLM API Key"这个当前架构完全没设防的口子。

**二期预留 `DockerSandboxProvider`**：接口不变，只是 `execute_command` 内部换成往容器里下发命令。当业务上出现"需要跑不受信任的用户自定义代码"这类场景时再启用，一期不做，因为当前 23 个技能里真正需要执行的是"项目自己写的脚本"，不是"用户上传的任意代码"，风险等级不一样，没必要一开始就上容器化的复杂度。

> **落地说明**（第三期）：已实现。`agent_core/sandbox/docker_sandbox.py` + `docker_sandbox_provider.py`，
> `SANDBOX_PROVIDER=docker` 时启用。只有 `execute_command` 真正起容器（一次性 `--rm` 语义，用完即删），
> `read_file`/`write_file`/`list_dir` 委托给内部组合的 `LocalSandbox` 复用（文件操作本质是宿主机文件
> 系统操作，不需要容器隔离）；命令里的 `sys.executable`（宿主机解释器路径）会被翻译成容器内的
> `python3`，技能脚本用到的项目内绝对路径也会被翻译成容器内的等价路径（项目根目录只读挂载到
> `/app`）。已知限制：默认镜像 `python:3.11-slim` 不含 Node.js，涉及 `.js` 脚本的技能需要换镜像；
> 暂不支持 `execute_command(stdin=...)` 管道输入（`SkillContentReader` 传 JSON 参数给脚本用到这个），
> 需要用到 stdin 的场景请继续用 `SANDBOX_PROVIDER=local`。

**`ReadBeforeWriteMiddleware`（可选，性价比高，建议一期就做）**：`write_file`/`str_replace` 类工具执行前，校验该路径是不是刚被 `read_file` 读过且内容哈希没变，否则拦截并提示"请先读取该文件当前内容"。这条防护成本很低（一个内存里的 path→hash 映射），能有效防止 Agent 在没看清文件当前内容的情况下盲写覆盖。

> **现状**：尚未实现，`middlewares/` 目录下没有这个文件，`write_file` 目前可以在没有先 `read_file`
> 的情况下直接覆盖写。本轮第三期任务未覆盖这一项（用户明确点的是 `DockerSandboxProvider`/staleness/
> outputs 下载/checkpoint 清理这四项），仍是待办。

### 5.4 技能脚本的密钥声明（`required_secrets`）

在 SKILL.md frontmatter 新增可选字段：

```yaml
required_secrets:
  - name: ERP_API_TOKEN
    optional: false
```

调用方（前端/API 调用者）在请求里通过独立的 `secrets` 字段传入（不进入对话消息、不进 checkpoint、不进日志），中间件在真正调用该技能脚本时才注入到 Sandbox 的 `execute_command(env=...)` 里，作用域仅限当次调用。这一条完全照抄 DeerFlow 的"三重授权交集"模型：技能被管理员启用 × 调用方本次请求提供了值 × frontmatter 声明了这个名字，三者都满足才注入，任何一个环节缺失都不注入、不报错，只是这个技能这次拿不到这个密钥。

> **落地说明**：已实现。`SkillDefinition` 新增 `required_secrets` 字段（`src/agent_core/skills/skill_definition.py`），`SkillLoader` 解析 frontmatter 里的声明，`SkillToolFactory._resolve_secret_env()` 做三重交集判断后透传给 `SkillContentReader.assemble()/run_script()`，最终经 `Sandbox.execute_command(env=...)` 注入子进程，`env_policy.build_sandbox_env` 的清洗逻辑不变。配套单测见 `test/skills/test_required_secrets.py`。

---

## 六、目录结构建议

```
diit-agent-server/
├── src/
│   ├── main.py                     # 组装入口，只做依赖注入和路由挂载
│   ├── agent_core/                 # 【新】对应 DeerFlow 的 harness 层，理论上可独立发包
│   │   ├── loop.py                 # Lead Agent 构建 + 中间件流水线组装
│   │   ├── middlewares/            # 第四节的 11 个中间件，每个独立文件
│   │   ├── guardrail/              # 5.1：GuardrailProvider 协议 + AllowlistGuardrailProvider
│   │   ├── sandbox/                # 5.3：Sandbox 协议 + LocalSandboxProvider（二期加 docker/）
│   │   ├── workspace/              # 5.2：ThreadData 路径管理 + 路径穿越校验
│   │   ├── skills/                 # 原 src/core/skills 原样搬入，只改 content_reader 里执行脚本这一步接 sandbox
│   │   ├── memory/                 # 原 src/core/memory 原样搬入，不改内部实现
│   │   ├── tools/                  # 原 src/core/tools 原样搬入
│   │   └── prompts/                # 3.4：原 src/core/prompts 原样搬入，新增 PromptFactory 封装渲染逻辑
│   ├── agents/                     # 保留期：general_agent 等 5 个专用能力，逐步从"图节点"改造成"委派工具"
│   ├── api/                        # 不变：router / websocket，协议契约冻结
│   ├── service/                    # 变薄：chat_service.py 大部分逻辑挪进 agent_core 中间件，只保留"调用 loop + 落库"的编排代码
│   ├── conversation/ storage/ schema/ model/ utils/   # 不变
├── plugins/                        # 不变
├── skills/                         # 不变（core/ + public/），新增技能可选带 required_secrets 字段
└── docs/
```

`api`/`service` 层是"App 层"，`agent_core` 是"Harness 层"——这个切分对应第二节表格里 DeerFlow "harness 与 app 严格分层、依赖方向单向"的做法，好处是 `agent_core` 未来具备独立测试、甚至被其它入口（比如批处理脚本、IM 渠道）复用的可能性，而不必每次都套一层 FastAPI。

> **实际落地说明**：`raster-agent-server` 仓库已按此思路建工程（`api/` → `agent_core/`+`storage/` → `schema/`，依赖方向单向），`agent_core/` 下 guardrail/sandbox/workspace/skills/memory/tools/prompts 七个子模块、`loop.py`/`middlewares/` 中间件流水线、`model/`（LLM 工厂）、`agents/`（Lead Agent + 通用 `task()` 分发，取代原方案里"5 个专用能力"的提法——只有 `web-researcher` 一种 subagent 走委派，`rag`/`database` 落地为 Lead Agent 自己的技能，`general_agent`/`tool_agent` 并入 Lead Agent 基础工具集）均已落地。仍未迁移的是 `service/`（`chat_service.py` 的 WS 部分）与 WebSocket 层，`api/router/` 新增了 `chat_router.py`（非流式 `POST /chat/`）与 `conversation_router.py`，取代了原方案里"service 变薄，只保留调用 loop + 落库"的定位——本仓库直接在 router 层做这件事，没有再单独设 `service/` 目录，因为中间件已经把大部分横切逻辑接管了，router 层剩下的编排代码本身已经很薄。

---

## 七、分期路线图

| 期 | 范围 | 对 Web 的影响 | 落地状态 |
|---|---|---|---|
| **第一期** | 中间件流水线骨架落地（第四节 4.2 的 11 个中间件，先不改路由方式，Supervisor 图原样保留）；权限控制（5.1）；虚拟工作区 + LocalSandboxProvider（5.2、5.3）；技能脚本执行迁移到 Sandbox 后执行；`required_secrets` | 零影响，REST/WS 契约不变 | 全部落地：权限控制、虚拟工作区、LocalSandboxProvider、技能脚本沙箱化、`required_secrets`、中间件流水线骨架（`agent_core/loop.py` + `agent_core/middlewares/`）均已在 `raster-agent-server` 完成，第一期收尾 |
| **第二期** | Supervisor 多图路由 → Lead Agent + 通用 `task()` 分发（4.1）；`create_agent` 组装 + 中间件真正接入；`thinking_agent`/`tool_agent` 收敛为 Lead Agent 运行时配置/基础工具集；`datasource_id` 强制路由硬校验；Redis-based Eval 遥测；REST `POST /chat/` + `/conversations/*` + WebSocket `/ws/chat` 流式协议 + 前端工具双通道回环 | REST/WS 均新增，不改现有 `/skills`/`/memories` 契约 | 全部完成：`agent_core/model/`（LLM 工厂）、`agent_core/agents/`（Lead Agent + `task(subagent_type, task)` 通用分发 + `DatasourceRoutingMiddleware`，只注册了 `web-researcher` 一种 subagent，`rag`/`database` 落地为 `search_knowledge_base`/`query_database` 两个技能而非委派工具）、`agent_core/eval/`（Redis Eval 遥测）、`storage/conversation_store.py`/`message_store.py`、`api/router/chat_router.py`/`conversation_router.py`、`api/websocket/`（`connection_manager.py`/`chat_ws.py`）均已落地并通过单测 |
| **第三期（按需）** | `user_id` 身份认证接入（配合 Guardrail 才有实际意义）；`DockerSandboxProvider`；记忆 staleness 定期复核；`outputs/` 产物的下载链接；`checkpoint` 孤儿数据清理 | 后四项零影响（下载链接按"生成 URL 挂进 Markdown 正文"方式兼容，不需要前端改动） | `DockerSandboxProvider`（`agent_core/sandbox/docker_sandbox*.py`，`SANDBOX_PROVIDER=docker` 时启用）、记忆 staleness 定期复核（`agent_core/memory/memory_staleness_reviewer.py`）、`outputs/` 产物下载链接（`save_output_file` 工具 + `GET /conversations/{id}/outputs/{path}`）、checkpoint 孤儿数据清理（`storage/checkpoint_cleanup.py`，删除会话时立即清理 + `main.py` 后台维护循环兜底历史孤儿数据）均已完成；唯一仍未开始的是 `user_id` 身份认证 |

第一期是本方案的核心交付物，能独立验证权限/文件系统/沙箱三块新能力，且完全不触碰现有路由逻辑，风险最低、收益最直接（补上安全性现状文档里点名的两个缺口：技能脚本无隔离、无工具级权限控制）。第二期验证了"完整 Agent Loop"意义上最核心的一步——Lead Agent + 委派工具跑通 REST 与 WebSocket 两条对话链路，是改动面最大的一轮，落地时把原项目 `chat_service.py` 里 REST/WS 两条路径的重复逻辑收敛成了一个共享的 `chat_pipeline.py::run_chat_turn()`，比原项目实现更精简。

---

## 八、和 diit-agent-web 的兼容性核对清单

逐条对照前端实际依赖的契约，确认本方案不破坏任何一条：

| 前端依赖点 | 来源 | 本方案是否改动 |
|---|---|---|
| WS 消息类型 `chat/send`、`tool/response`，服务端推 `start/token/thinking/tool_call/tool_response/reference/done` | `AgentApi.js`、`前端工具与后端工具结合设计.md` | 已在 `raster-agent-server` 的 `api/websocket/chat_ws.py` 落地，事件类型/字段与原协议一致（额外新增 `skill_call`/`skill_response` 两类事件，前端已有对应展示逻辑） |
| `POST /chat/`、`GET/POST/PUT/DELETE /conversations/` | `conversationApi.js` | 字段名/响应结构已在 `raster-agent-server` 按原契约实现（`chat_router.py`/`conversation_router.py`），`POST /chat/` 仍是非流式（原项目同款设计，内部也用事件流驱动） |
| `GET /skills/`、`GET /skills/{tool_name}`、`/tree`、`/file`、`PUT /skills/{tool_name}/toggle` | `skillsApi.js` | 不改，`SkillAccessGuard` 升级为 Guardrail 的一部分后，`user_skill_settings` 表结构和这几个接口的行为完全不变，已在 `raster-agent-server` 落地并保持路径/参数/响应结构一致 |
| `GET/POST/PUT/DELETE /memories/`、`/search`、`/audit-logs` | `memoryApi.js` | 不改，`MemoryManager` 对外接口不变，已在 `raster-agent-server` 落地并保持路径/参数/响应结构一致 |
| 前端工具双通道（执行通道 `type:"tool"` + 展示通道 `tool_call`/`tool_response`） | `前端工具与后端工具结合设计.md` 第二节 | 已在 `raster-agent-server` 落地：`CustomToolConverter.merge_tools()` + `ConnectionManager`（第一期迁移的代码原样复用）+ `chat_ws.py` 完整走通这条回环 |

---

## 九、风险与取舍

**为什么不一步做完 Lead Agent + Subagent（即不把第二期并入第一期）**：`supervisor.py` 里那一大坨"跟 LLM 结构化输出退化搏斗"的代码，是过去踩坑攒下来的经验（DeepSeek `json_mode` 空白 content、`length limit` 截断、重复退化生成……），换成"委派工具"模式后这些坑不会自动消失，只是换了个位置踩（工具调用同样依赖 LLM 决定"要不要调用哪个委派工具"，只是不再要求严格 JSON schema，理论上更稳，但需要实测验证）。建议先在第一期把可独立验证的三块新能力（权限/文件系统/沙箱）做稳，第二期路由方式的改造单独起一轮，避免"一次性重构 + 新架构不稳定"两个风险叠加。

**为什么沙箱一期不上容器隔离**：当前 23 个技能的脚本都是项目自己写的、经过 review 的代码，不存在"执行用户上传的任意代码"这种场景，`RLIMIT_AS` + 超时 + 输出截断 + 环境变量清洗这几层已经能覆盖当前风险等级；上 Docker/K8s 会引入部署复杂度（需要额外的容器编排、镜像维护），在没有实际"跑不受信任代码"的需求之前，属于过度设计。接口按 `SandboxProvider` 协议预留好，需要时可以平滑切换，不需要现在就买单。

**权限控制和身份认证的先后顺序**：Guardrail 能先做，但如果 `user_id` 仍然是前端明文传入、服务端不校验，那么"用户 A 冒充用户 B 的 `user_id`"这件事依然成立，权限控制形同虚设。这不是本方案能单独解决的问题，需要和身份认证一起排期（见第七节第三期），本方案只保证 Guardrail 的接口设计不会在身份认证接入后需要推倒重来。
