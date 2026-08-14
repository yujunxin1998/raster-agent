# Skill 注入与 Load-Skill 重构设计

> 参考用户提供的 GPT 方案（"单一 `load_skill` 工具 + 渐进式披露 + 依赖交集校验"）与
> Claude 自身 Agent Skills 的设计思路，结合本项目 `src/agent_core/skills/` 现状与
> [`docs/工具注册中心与热重载设计.md`](./工具注册中心与热重载设计.md) 已落地的 `ToolRegistry`，
> 给出的落地方案。核心结论：GPT 方案里"N 个 Skill 收敛成 1 个 `load_skill` 工具"这个
> 主张是对的，但**只对本项目一半的技能成立**——另一半技能本质是参数化业务工具，不是
> "加载指令"，硬套会破坏现有能力。方案因此拆成两条腿走。

## 一、现状盘点

### 1.1 组件与职责（已经是"渐进式披露"架构，比 GPT 方案假设的起点更成熟）

`SkillDefinition`（[skill_definition.py](../src/agent_core/skills/skill_definition.py)）→
`SkillLoader`（[skill_loader.py](../src/agent_core/skills/skill_loader.py)，启动时只解析
frontmatter，不读正文）→ `SkillRegistry`（内存索引，按 `tool_name` 精确查 + 按
`category` 批量查）→ `SkillToolFactory`（按需把一个 `SkillDefinition` 包成
`StructuredTool`，正文/参考资料/脚本执行全部委托 `SkillContentReader` 现读现跑）→
`SkillManager`（门面，`init_skill_manager()`/`get_skill_manager()` 单例）。

`SkillContentReader.assemble()` 按"指令 + 参考资料 + 脚本结果"拼装最终返回给模型的文本，
脚本执行经 `Sandbox` 接口（`LocalSandbox`，cwd 固定在该会话的 `workspace/` 子目录）。

这条链路已经实现了 GPT 方案里"启动只读 frontmatter、正文延迟到调用时才读"的核心诉求
（GPT 方案第三节"SkillLoader 只轻加载 frontmatter"），**不需要重做**。

### 1.2 两种技能形态已经并存，但代码里从未显式区分

扫描 `skills/core/`（2 个）+ `skills/public/`（10 个）全部 12 个 SKILL.md，发现一个
GPT 方案完全没有预料到的事实：**本项目的"Skill"其实是两种不同的东西，混用同一套
`SkillDefinition`/`SkillToolFactory` 路径**：

| | `search_knowledge_base` / `query_database`（core，2 个） | 其余 10 个（public） |
|---|---|---|
| frontmatter `parameters` | 有（`query`/`top_k`、`query_text`） | **全部为空** |
| `runtime_context_keys` | 有（`workflow_id`/`db_id`/... 、`datasource_id`） | 全部为空 |
| `scripts/main.py` 入口 | 有，`SkillContentReader` 自动跑 | 无（`has_script()` 恒 False） |
| 正文内容 | "如何解读脚本返回结果"的格式规范 | 完整的多步工作流指令，指导模型自己调用
`run_command`/`run_python`/`web_search`/`web_fetch` 等**已有工具** |
| 本质 | 参数化业务工具，SKILL.md 只是它的元数据外壳 | Claude Agent Skills 意义上的"工作流指令"，本身不是工具 |

这正是 GPT 方案默认的形态（技能 = 加载一段指令，指令里再调用别的工具）——但只对这 10
个成立。2 个 core 技能如果套用"收敛成 `load_skill`"，会把一个有真实入参和脚本执行的
业务工具降级成"先加载文档，模型再假装调用一个不存在的工具"，是倒退。

判定信号：**`bool(skill.parameters)` 已经可以无成本、零迁移成本地区分两种形态**——
不需要新增 frontmatter 字段，12 个现有 SKILL.md 一个都不用改。

### 1.3 `SkillCategory` 是路由轴，不是形态轴，两者正交

`category` 决定挂到哪个 Agent（`general`/`tool`/`rag`/`database` 4 类挂 Lead Agent，
`web_search` 挂 `web-researcher` subagent，见
[providers.py:73-78](../src/agent_core/tools/registry/providers.py#L73)、
[subagent_profiles.py:58](../src/agent_core/agents/subagent_profiles.py#L58)）。
实测 12 个技能的 category 分布：`general`(2) `rag`(1,core) `database`(2，1 core+1
public) `tool`(3，全 public) `web_search`(3，全 public)。同一个 `database` 分类下
同时混着 1 个参数化工具技能（`query_database`）和 1 个工作流技能（`data-analysis`）
——形态判定必须独立于 category，用 1.2 的信号，不能借用 category。

### 1.4 真实 bug：10 个 public 工作流技能里，至少 6 个的脚本路径在本项目沙箱里完全不可达

`skills/public/` 下 6 个 SKILL.md 正文里出现 `/mnt/skills/public/...`（脚本路径）与
`/mnt/user-data/{uploads,outputs}/...`（用户文件路径），共 51 处引用：

```
academic-paper-review/SKILL.md   1 处
data-analysis/SKILL.md          23 处
code-documentation/SKILL.md      7 处
image-generation/SKILL.md       14 处
systematic-literature-review/SKILL.md + evals/evals.json   6 处
```

这些路径是原样照搬 Anthropic 公开 Agent Skills 库的约定（对应 Claude 自己的代码执行
沙箱挂载布局），但本项目的 `Sandbox` 抽象（[sandbox.py](../src/agent_core/sandbox/sandbox.py)）
从未实现过 `/mnt` 挂载——`LocalSandbox.execute_command()` 的 cwd 固定在
`ThreadWorkspace.workspace_dir`（每会话隔离目录下的 `workspace/` 子目录），命令行参数
原样传给 `subprocess.Popen`，不做任何路径改写（[local_sandbox.py:74-82](../src/agent_core/sandbox/local_sandbox.py#L74)）。
全仓库 grep `/mnt/skills`/`/mnt/user-data` 只在这 6 个 SKILL.md 里出现，`src/` 下零处理逻辑。

结论：**模型如果老实按这 6 个技能的字面指令去调用
`run_command(["python", "/mnt/skills/public/data-analysis/scripts/analyze.py", ...])`，
会因为路径不存在直接失败**——这几个技能今天实际不可用，与本次要不要做 `load_skill`
无关，是一个独立的、更基础的缺口，本次重构必须一并修。

另有 `chart-visualization/SKILL.md` 用的是另一种同样错误的写法——裸相对路径
`node ./scripts/generate.js`，隐含假设"命令执行时 cwd 就是技能自己的目录"，但沙箱 cwd
固定是会话 `workspace/`，不是技能目录，同样不可达。`consulting-analysis`/`surprise-me`
两个技能不涉及脚本，没有这个问题。

本项目其实已经有 `/mnt/user-data/{uploads,outputs}` 的对应物——
`ThreadWorkspace`（[thread_workspace.py](../src/agent_core/workspace/thread_workspace.py)）
早就实现了 `workspace/`/`uploads/`/`outputs/` 三个同级子目录 + 虚拟路径解析，只是
`uploads_dir` 目前只在文件上传落盘（API 层）时使用，`read_file`/`run_command` 等沙箱
工具没有把它暴露给模型——即"应有的目标路径系统已经存在"，缺的只是 SKILL.md 里写的
路径 token 到这个真实系统的映射（见六）。

### 1.5 `skill_system.md` 是静态手写模块，10 个 public 技能对模型完全"隐身"

Lead Agent 的系统提示词里 `<skill_system>` 模块
（[skill_system.md](../src/agent_core/prompts/system/lead_agent/skill_system.md)）
是纯手写文本，只提到 `search_knowledge_base`/`query_database` 两个 core 技能。10 个
public 技能虽然各自作为独立 `StructuredTool` 挂在 Lead Agent 工具集里（经
`SkillToolProvider` → `_LEAD_AGENT_SKILL_CATEGORIES` 覆盖 general/tool/rag/database），
但**模型只能靠每个工具自己的 name+description 猜什么时候用**，系统提示词没有一句话
提示"你有这些工作流技能可用"。这正是 GPT 方案"工具目录污染"问题在本项目里的真实体现：
每多一个 public 技能，Lead Agent 的工具 schema 就多一条，且没有任何 catalog 兜底。

### 1.6 与 `ToolRegistry`/热重载的既有集成点

上一轮改动已经把 Skill 接入统一工具注册中心（详见
[工具注册中心与热重载设计.md](./工具注册中心与热重载设计.md)）：`SkillToolProvider.discover()`
按 `_LEAD_AGENT_SKILL_CATEGORIES` 遍历，每个技能生成一条 `ToolDefinition`；
`SkillHotReloader.reload_once()` 监听 `SKILL.md` 变化，"先用新 `SkillRegistry` 算出候选
`ToolDefinition` 列表试发布，`ToolRegistry.publish()` 成功后才切换 `skill_manager.registry`"
的两阶段设计。**本次重构必须复用这套发布/热重载机制，不能绕开**——下面设计的改动只发生
在"`SkillToolProvider.discover()` 产出什么" 这一层，`ToolRegistry`/`SkillHotReloader`
内部机制不动。

## 二、目标与非目标

**目标：**

1. 引入"技能形态"区分（`SkillKind.TOOL` / `SkillKind.WORKFLOW`），从现有 `parameters`
   字段零成本推断，不新增 frontmatter 字段、不改任何现有 SKILL.md。
2. `WORKFLOW` 形态的技能收敛成一个通用 `load_skill(skill_name)` 工具 + 系统提示词里
   动态生成的技能目录（catalog），解决 1.5 的"隐身"问题，同时把 Lead Agent 工具 schema
   从"内置 7 个 + 技能 N 个"降到"内置 7 个 + `load_skill` 1 个"。
3. `TOOL` 形态的技能（`search_knowledge_base`/`query_database`）**原样保留**现有
   `SkillToolFactory` 路径（独立 `StructuredTool` + 真实参数 schema + 脚本执行），
   不受本次改动影响。
4. 修复 1.4 的路径不可达问题：技能激活时对正文做一次路径 token 替换，不改 SKILL.md
   源文件本体（保持与上游 Anthropic 技能库的可比对性，未来技能库整体更新时不会被
   本地修改污染 diff）。
5. 新设计原样接入已有 `ToolRegistry`/`SkillHotReloader`，两者内部机制零改动。

**非目标（明确不做，原因写在第十一节"与 GPT 原方案的偏差"）：**

- 不引入 GPT 方案里 `required_tools`/`available_tool_names` 交集式依赖校验器——本项目
  已经用 `category` 路由达到同等效果（web_search 技能只会出现在
  `web-researcher` subagent 的工具集里，那里保证有 `web_search`/`web_fetch`）。
- 不新增 `execute_skill_script` 专用工具——脚本执行复用已有 `run_python`/`run_command`
  （Lead Agent 自己的工具，走完整中间件链），新增一个平行的技能专属执行器只是重复造轮子。
- 不把 frontmatter 改成 GPT 方案里 `metadata:` 嵌套格式——现有扁平 frontmatter
  （`parameters`/`runtime_context_keys`/`required_secrets` 都是顶层字段）已经工作良好，
  嵌套化要求重写全部 12 个 SKILL.md 换不来任何功能收益。
- 不引入 `search_skills(query)` 检索式发现——12 个技能全量塞进系统提示词的 token
  成本可忽略（每条一行 name+description），技能数量到两位数以上量级时再考虑。
- 不批量重写 `skills/public/*/SKILL.md` 里 `/mnt/...` 路径本体——用运行时 token 替换
  代替源码迁移（见六）。

## 三、`SkillKind` 判定

不新增存储字段，`SkillDefinition` 加一个计算属性：

```python
# skill_definition.py 新增

class SkillKind(str, Enum):
    """技能形态：决定它被投影成"独立参数化工具"还是"load_skill 可加载的工作流"。"""

    TOOL = "tool"          # 有 parameters，本质是业务工具（如 query_database）
    WORKFLOW = "workflow"  # 无 parameters，纯指令，指导模型调用别的已有工具


@dataclass(frozen=True)
class SkillDefinition:
    ...  # 字段不变

    @property
    def kind(self) -> SkillKind:
        """从既有 `parameters` 字段推断形态，零迁移成本——12 个现有 SKILL.md 无需改动。"""
        return SkillKind.TOOL if self.parameters else SkillKind.WORKFLOW
```

`SkillLoader`/`SkillRegistry` 不需要任何改动——`kind` 是纯派生属性，不影响加载和索引逻辑。

## 四、`load_skill` 加载机制设计（Middleware 优先，Tool 兜底）

> **本节是实现阶段与最初设计发生的最大一次改动**，记录在案而不是悄悄改掉：最初版本
> （见文档历史）设想"`load_skill` 是一个普通 `StructuredTool`，校验+加载逻辑写在它
> 自己的 `coroutine` 里"，模型调用它、拿到一条 ToolMessage，跟调用其它任何工具没有
> 区别。用户看完这版设计后指出一个更本质的问题：**Skill 不是"要执行的业务能力"，
> 是"给模型看的指令/上下文"，把它包装成一个业务 Tool 会让权限模型、审计语义都变得
> 别扭**——LangChain 官方也确实同时支持"`load_skill` Tool"和"Skill Middleware
> 自动发现+加载"两种模式，后者才是更贴近 Skill 本质的表达。本节据此重新设计。

### 4.1 `SkillActivationService`（不变）

新文件 `skill_activation_service.py`，职责：按名称激活一个 `WORKFLOW` 技能，返回
（已做路径替换的）正文文本。**不处理 `TOOL` 形态技能**——那些继续走
`SkillToolFactory`，`SkillActivationService` 只在被调用时校验 `kind == WORKFLOW`，
防止模型/攻击者通过 `load_skill("query_database")` 绕过参数校验直接拿到原始正文。

```python
class SkillActivationService:
    def __init__(self, registry: SkillRegistry, path_rewriter: SkillPathRewriter | None = None) -> None:
        self._registry = registry
        self._path_rewriter = path_rewriter or SkillPathRewriter()

    async def activate(self, skill_name: str, allowed_categories: frozenset[str]) -> str:
        skill = self._registry.get(skill_name)  # 未找到抛 SkillNotFoundError

        if skill.category not in allowed_categories:
            # 与"技能未找到"用同一措辞（不泄露"存在但你无权用"这一区分）
            raise SkillNotFoundError(f"技能 '{skill_name}' 未找到。已注册：{self._registry.names}")

        if skill.kind is not SkillKind.WORKFLOW:
            raise SkillDefinitionInvalidError(
                f"技能 '{skill_name}' 是参数化工具技能，请直接调用同名工具而不是 load_skill"
            )

        reader = SkillContentReader(skill)
        body = reader.read_instructions()
        references = reader.read_references()
        if references:
            body += "\n\n## 参考资料\n" + "\n\n".join(references)
        return self._path_rewriter.rewrite(skill, body)
```

`SkillContentReader` 不改：只用它已有的 `read_instructions()`/`read_references()`，
不调用 `assemble()`——`assemble()` 还会走脚本执行分支，`WORKFLOW` 技能的脚本由模型
自己后续调用 `run_command`/`run_python` 触发，不是激活时自动跑。

### 4.2 两条加载路径：Lead Agent 走 Middleware，subagent 走自包含 Tool

`load_skill` 在**模型看到的 schema 层面**始终是同一个工具（`skill_name: str` →
返回文本），但**谁来做真正的权限校验+加载**因为一个既有架构约束而分成两条路径：

- **Lead Agent**：新增 `agent_core/agents/skill_middleware.py::SkillMiddleware`。
  它的 `awrap_tool_call` 拦截对 `load_skill` 的调用，在这里完成"按具体
  `skill_name` 做 Guardrail 校验 + 调用 `SkillActivationService`"，然后直接返回
  `ToolMessage`，**不调用 `handler(request)`**——`load_skill` 自己注册的
  `StructuredTool.coroutine` 在这条路径上永远不会被执行，是真正意义上的"Skill 交给
  Middleware 处理，不伪装成业务 Tool 的私有实现"。
- **`web-researcher` subagent**：`sub_agent_factory.py::run_subagent()` 现造的子
  Agent **不挂任何中间件**（安全边界，见该模块 docstring——`GuardrailMiddleware`/
  `LoopDetectionMiddleware` 都不在场，是有意为之的隔离设计，不是本次要动的范围）。
  这条路径上没有 `SkillMiddleware` 可以拦截，`load_skill` 只能靠自己的
  `StructuredTool.coroutine`（`skill_load_tool.py::create_load_skill_tool`）独立
  完成权限校验+加载，是这条路径上**真正被执行**的实现，不是兜底。

两条路径复用同一个 `SkillActivationService`/`GuardrailProvider` 实例、同一份
校验逻辑（"按具体 `skill_name` 查权限，不是恒定的 `"load_skill"`"），只是"谁来触发
这份逻辑"因为 subagent 无中间件这一既有约束而不同——**这是架构现实决定的必然分叉，
不是设计偏好**，被迫保留 Tool 版本恰好也验证了"Tool 兜底不能完全去掉"这一点。

```python
# skill_load_tool.py —— 两条路径共用的 schema + 自包含兜底实现
def create_load_skill_tool(
    activation_service: SkillActivationService,
    guardrail_provider: GuardrailProvider,
    allowed_categories: frozenset[str],
) -> StructuredTool:
    async def _invoke(skill_name: str, config: RunnableConfig) -> str:
        configurable = config.get("configurable", {}) if config else {}
        decision = await guardrail_provider.check(
            user_id=configurable.get("user_id"),
            tool_name=skill_name,  # 按具体技能名校验，不是 "load_skill"
            tool_args={},
            context=GuardrailContext(conversation_id=configurable.get("thread_id")),
        )
        if not decision.is_allowed:
            return decision.reason
        try:
            return await activation_service.activate(skill_name, allowed_categories)
        except (SkillNotFoundError, SkillDefinitionInvalidError) as exc:
            return str(exc)

    return StructuredTool(name="load_skill", args_schema=LoadSkillInput, coroutine=_invoke, ...)
```

```python
# skill_middleware.py —— Lead Agent 专属，拦截 + 短路
class SkillMiddleware(AgentMiddleware[Any, AgentRuntimeContext]):
    async def awrap_tool_call(self, request: ToolCallRequest, handler) -> ToolMessage | Command:
        tool_call = request.tool_call
        if tool_call["name"] != LOAD_SKILL_TOOL_NAME:
            return await handler(request)  # 其余工具调用原样透传，不受影响

        context = request.runtime.context
        skill_name = str((tool_call.get("args") or {}).get("skill_name") or "")
        decision = await self._guardrail_provider.check(
            user_id=context.user_id if context else None,
            tool_name=skill_name,
            tool_args={},
            context=GuardrailContext(conversation_id=context.conversation_id if context else None),
        )
        if not decision.is_allowed:
            return ToolMessage(content=decision.reason, tool_call_id=tool_call["id"], status="error")
        try:
            content = await self._activation_service.activate(skill_name, self._allowed_categories)
        except (SkillNotFoundError, SkillDefinitionInvalidError) as exc:
            content = str(exc)
        return ToolMessage(content=content, tool_call_id=tool_call["id"])
```

### 4.3 挂载顺序约束

`SkillMiddleware` 不进 `agent_core/loop.py::build_middlewares()`（那 12 个是跨
Lead Agent/subagent 通用的流水线，subagent 完全不挂）——随 `lead_agent.py` 里
`DatasourceRoutingMiddleware`/`StreamingModelMiddleware` 同一挂载方式追加，且顺序
必须是：

```python
middlewares = [
    *build_middlewares(...),          # 含 ToolAudit/ToolErrorHandling/LoopDetection
    DatasourceRoutingMiddleware(),
    SkillMiddleware(...),             # 必须在这三者之后、StreamingModelMiddleware 之前
    StreamingModelMiddleware(),       # 必须是模型调用链路最内层，见其模块文档
]
```

排在 `ToolAuditMiddleware`/`ToolErrorHandlingMiddleware`/`LoopDetectionMiddleware`
之后（列表里越靠后越贴近实际执行）是关键：这样 `load_skill` 调用依然会被审计日志、
异常兜底、死循环检测覆盖，`SkillMiddleware` 只是这条链路里最内层、真正做"要不要把
技能正文喂给模型"这个决策的一层，短路发生在离真实执行最近的地方，前面几层通用防护
不会因为 Skill 短路而失效——这与 `GuardrailMiddleware` 拒绝时"不调用 handler，直接
返回 error ToolMessage"是完全一致的既有模式，只是本类只对 `load_skill` 这一个工具名
生效，其余调用原样透传给下一层。

### 4.4 权限粒度不能丢（两条路径都要满足）

无论走哪条路径，**收敛成一个 `load_skill` 工具后，外层 `GuardrailMiddleware`/
`user_tool_permissions` 表按 `tool_name="load_skill"` 只能整体禁用，无法再单独禁用
某一个具体技能**（比如运营想单独关掉 `image-generation` 但保留其它工作流技能）。
现有 `SkillToolFactory._invoke` 内部会用 `skill.tool_name` 做一次 Guardrail 校验
（[skill_tool_factory.py:121-128](../src/agent_core/skills/skill_tool_factory.py#L121)），
这个粒度不能丢——`SkillMiddleware.awrap_tool_call` 和 `create_load_skill_tool` 的
`_invoke` 都按被请求的具体 `skill_name` 再做一次 Guardrail 校验，而不是依赖外层
`GuardrailMiddleware`（后者只看到 `tool_name="load_skill"`，无法做到这一层）。

## 五、系统提示词技能目录注入

发现（discovery）和激活（activation）分开处理，且**同样因为 subagent 无中间件而
分成两种注入方式**：

- **Lead Agent**：目录注入是 `SkillMiddleware.awrap_model_call` 的职责，现算当前
  `allowed_categories` 范围内的 `WORKFLOW` 技能（不缓存，Skill 热重载后下一次模型
  调用就能看到最新目录），追加进 `request.system_prompt`：

  ```python
  async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse:
      catalog = build_skill_catalog(self._workflow_skills())
      if catalog:
          block = f"<skill_catalog>\n{catalog}\n</skill_catalog>"
          request = request.override(
              system_prompt=f"{request.system_prompt}\n\n{block}" if request.system_prompt else block
          )
      return await handler(request)
  ```

- **`web-researcher` subagent**：没有中间件，`subagent_profiles.py` 的
  `system_prompt_factory`（惰性函数，`task` 工具真正派发时才现算）里直接拼接一次：
  `prompt_factory.get("WEB_SEARCH_AGENT") + build_skill_catalog(web_search 分类下的 WORKFLOW 技能)`。

两处共用同一个格式化函数 `skill_catalog.py::build_skill_catalog()`，避免目录文案
两处各写一套、后续要改格式要改两遍：

```python
def build_skill_catalog(skills: list[SkillDefinition]) -> str:
    if not skills:
        return ""
    lines = ["以下是可通过 `load_skill(skill_name)` 加载的工作流技能，加载后按返回的指令执行，不要凭空猜测流程："]
    for skill in sorted(skills, key=lambda s: s.tool_name):
        lines.append(f"- {skill.tool_name}: {skill.description}")
    return "\n".join(lines)
```

`skill_system.md`（静态手写模块，描述 `search_knowledge_base`/`query_database`
两个 `TOOL` 形态技能的调用规则）保持不变——这部分是业务规则，不是"有哪些技能"的
枚举，不需要动态化，也不做成 `system_prompt_builder` 的新静态模块（目录内容是运行时
数据，随 `SkillRegistry` 热重载变化，不是手写文案，混进 `.md` 文件反而两头不讨好）。

## 六、路径映射修复

新文件 `skill_path_rewriter.py`，只做字符串替换，不解析 Markdown：

```python
class SkillPathRewriter:
    """把 SKILL.md 正文里 Anthropic 公开技能库的路径约定，替换成本项目沙箱的真实约定。

    只替换"这个技能自己的"路径前缀（用 skill.skill_dir 精确匹配，不是全局正则），
    避免误改到正文里恰好提到的别的技能名字符串。
    """

    def rewrite(self, skill: SkillDefinition, text: str) -> str:
        skill_root_token = f"/mnt/skills/{skill.source}/{skill.skill_dir.name}"
        text = text.replace(skill_root_token, str(skill.skill_dir))
        # 会话上传/产物目录：沙箱工具执行 cwd 固定在 workspace/，uploads/outputs 是同级目录
        text = text.replace("/mnt/user-data/uploads", "../uploads")
        text = text.replace("/mnt/user-data/outputs", "../outputs")
        text = text.replace("/mnt/user-data/workspace", ".")
        return text
```

`skill.source`（已有属性，取 `skill_dir.parent.name`，即 `core`/`public`）+
`skill.skill_dir.name` 精确定位到"这个技能自己的脚本目录前缀"，替换成
`SkillLoader` 扫描时就已经拿到的真实绝对路径（宿主机上 `skills/public/data-analysis`
这一层的绝对路径）——模型看到的指令里 `python /mnt/skills/public/data-analysis/scripts/analyze.py`
变成 `python F:\pythonProject\raster-agent\skills\public\data-analysis\scripts\analyze.py`
（或部署环境对应的绝对路径），可以直接原样传给 `run_command`。

`../uploads`/`../outputs` 相对路径能work的前提：`run_command`/`run_python` 的 cwd 固定是
`workspace/`，与 `uploads/`/`outputs/` 是 `ThreadWorkspace.root` 下的同级子目录
（见 [thread_workspace.py](../src/agent_core/workspace/thread_workspace.py)），
`subprocess.Popen` 不经过 `PathGuard`，`../uploads/xxx` 会被 OS 正常解析到
`root/uploads/xxx`，不需要新增任何沙箱接口。

**这个自动替换覆盖不到的情况，已在实现阶段一次性手动改了 SKILL.md 正文**（不是运行时
能解决的，见落地路线图阶段 0，均已完成）：

1. `chart-visualization` 原来用裸相对路径 `node ./scripts/generate.js`，没有 `/mnt/`
   前缀可匹配——已改成 `node /mnt/skills/public/chart-visualization/scripts/generate.js`，
   与其它技能的写法保持一致，可以被自动替换覆盖。
2. `data-analysis` 正文原来提到 `present_files` 工具（"export to file and share via
   `present_files` tool"）——本项目没有这个工具，已改写成"先用 `--output-file` 导出
   到 workspace，再用 `save_output_file` 工具取得真正的下载链接"（`save_output_file`
   接受的是内容字符串，不是"已存在文件的路径"，不能一步到位，见
   [sandbox_tool.py](../src/agent_core/tools/sandbox_tool.py)）。
3. **实现阶段核实后新发现**：`data-analysis`/`image-generation` 正文里各有一句裸
   `` `/mnt/user-data` ``（不带 `/uploads`、`/outputs` 等子路径，如"You don't need to
   check the folder under `/mnt/user-data`"），不匹配 `SkillPathRewriter` 的任何一条
   替换规则（规则只处理带子路径的三种 token）。这类纯说明性文字不影响脚本实际执行，
   但会在激活后的正文里留下一个本项目里不存在的路径引用，已同步改写成不含路径的等价
   说明。验证方法见落地路线图阶段 6：对全部 12 个技能过一遍 `activate()`，断言产出文本
   不包含任何 `/mnt/` 子串。

## 七、`SkillToolProvider`/`ToolRegistry` 集成改动

`providers.py::SkillToolProvider.discover()` 改动（`ToolRegistry`/`SkillHotReloader`
内部机制不动，只改这一处产出什么）：

```python
def discover(self, registry: SkillRegistry | None = None) -> list[ToolDefinition]:
    if not self._skill_manager.enabled:
        return []

    factory = self._skill_manager.tool_factory
    source_registry = registry if registry is not None else self._skill_manager.registry
    activation_service = SkillActivationService(source_registry)
    definitions: list[ToolDefinition] = []
    has_workflow_skill = False

    for category in LEAD_AGENT_SKILL_CATEGORIES:
        for skill in source_registry.by_category(category):
            if skill.kind is SkillKind.WORKFLOW:
                has_workflow_skill = True
                continue  # 不再逐个注册，统一走下面的 load_skill
            definitions.append(
                ToolDefinition(  # TOOL 形态：原逻辑不变
                    canonical_name=f"skill.{category}.{skill.tool_name}", ...,
                    build_tool=(lambda s=skill: factory.create(s)),
                )
            )

    if has_workflow_skill:
        definitions.append(
            ToolDefinition(
                canonical_name="skill.load_skill",
                model_name="load_skill",
                description="按名称加载一个工作流技能的完整操作指令",
                source_type="skill",
                source_id=self.source_id,
                scope="application",
                permissions_key="load_skill",
                # 这里的 build_tool 只负责"注册 schema"——Lead Agent 场景下真正
                # 被执行的是 SkillMiddleware.awrap_tool_call（见第四节），这个
                # StructuredTool 的 coroutine 只在没有该中间件的调用方手里才会
                # 真的跑起来。
                build_tool=(lambda: create_load_skill_tool(
                    activation_service, self._guardrail_provider,
                    allowed_categories=frozenset(LEAD_AGENT_SKILL_CATEGORIES),
                )),
            )
        )
    return definitions
```

`SkillToolProvider.__init__` 新增 `guardrail_provider` 参数（透传给 `load_skill` 的
自包含兜底实现），`main.py`/`SkillHotReloader` 两个调用点同步补上——`_LEAD_AGENT_
SKILL_CATEGORIES` 顺带从模块私有常量提升为 `LEAD_AGENT_SKILL_CATEGORIES`（去掉下划线
前缀并从 `agent_core.tools.registry` 包导出），因为 `lead_agent.py` 组装
`SkillMiddleware` 时也需要这份分类集合，不再只是 `providers.py` 内部细节。

- 快照唯一性约束不受影响：`load_skill` 只注册一条 `ToolDefinition`（`canonical_name`
  固定），不会因为技能数量变化产生冲突；`RegistrySnapshot.replace_source()` 的
  重复校验逻辑不用改。
- 热重载兼容：`SkillHotReloader.reload_once()` 调用
  `SkillToolProvider(...).discover(registry=new_registry)` 时天然拿到基于新
  `SkillRegistry` 重新计算的 `definitions`（`load_skill` 是否出现、覆盖哪些技能都是
  现算的）——两阶段发布/回滚机制原样复用，不用改
  [skill_hot_reload.py](../src/agent_core/tools/registry/skill_hot_reload.py)。
- `subagent_profiles.py` 里 `web-researcher` 的 `tools_factory` 不经过
  `ToolRegistry`（子 Agent 工具集是独立现算的 `list[BaseTool]`，见既有设计说明），
  同理把 `*get_skill_manager().get_tools("web_search")`（3 个全是 WORKFLOW 形态）
  替换成一个按 `allowed_categories=frozenset({"web_search"})` 构造的 `load_skill`
  自包含实现——这条路径没有 `SkillMiddleware`，`_invoke` 就是真正被执行的逻辑，
  不是兜底（呼应第四节 4.2）。

## 八、`SkillRegistry` 重复注册：从静默覆盖改成显式报错（顺带修复，独立小项）

现有 `SkillRegistry.register()` 同名 `tool_name` 静默覆盖（"后加载的生效"，见
[skill_registry.py:24-30](../src/agent_core/skills/skill_registry.py#L24)），依赖
`SkillLoader` 里 `sorted(skill_dir_root.glob(...))` 的目录扫描顺序，本质是"谁在文件系统
排序里靠后谁赢"，不是有意为之的优先级设计。这是 GPT 方案里"不建议同名覆盖，应该报错"
的建议，本项目也适用，属于独立的小型加固，不依赖本文档其余部分：

```python
def register(self, skill: SkillDefinition) -> None:
    if skill.tool_name in self._store:
        raise DuplicateSkillError(f"技能 tool_name 冲突: {skill.tool_name}")
    self._store[skill.tool_name] = skill
```

`SkillLoader.load()` 单个技能解析失败已经是"记 error 日志 + 跳过"的容错策略
（不中断整体加载），`register()` 抛出的 `DuplicateSkillError` 会被同一个 `except
Exception` 捕获，行为上是"两个同名技能都不注册、只留错误日志"，不会导致启动失败——
与现有容错哲学一致，不需要额外改 `SkillLoader`。

## 九、落地路线图（均已完成，见第十三节）

| 阶段 | 内容 | 依赖 | 状态 |
|---|---|---|---|
| 0 | 手动修 `chart-visualization` 路径写法 + `data-analysis` 的 `present_files→save_output_file` 措辞 + 两处裸 `/mnt/user-data` 措辞 | 无，独立可做 | ✅ |
| 1 | `SkillKind` 计算属性 + `SkillPathRewriter` + `SkillActivationService` + 单测 | 无 | ✅ |
| 2 | `create_load_skill_tool`（自包含实现）+ `SkillMiddleware`（Lead Agent 拦截层）+ 单测 | 阶段 1 | ✅ |
| 3 | `SkillToolProvider.discover()` 改造（TOOL/WORKFLOW 分流）+ `subagent_profiles.py` 改造 | 阶段 2 | ✅ |
| 4 | `build_skill_catalog()` + `SkillMiddleware.awrap_model_call`/`web-researcher` 系统提示词接入 | 阶段 3 | ✅ |
| 5（可选，独立） | `SkillRegistry.register()` 改成报错 | 无 | ✅ |
| 6 | 全量回归：12 个技能逐个过一遍（脚本验证 `activate()` 产出不含任何 `/mnt/` 残留）+ 单测 + 热重载回归 | 阶段 1-4 | ✅ 279 passed |

## 十、测试方案

- `SkillKind`：`parameters` 非空 → `TOOL`；为空 → `WORKFLOW`（用现有 2 个 core + 任一
  public 技能的真实 frontmatter 数据驱动，不需要 mock）。
- `SkillPathRewriter`：`/mnt/skills/public/x/scripts/y.py` → 正确替换为
  `skill.skill_dir/scripts/y.py`；`/mnt/user-data/uploads/a.xlsx` →
  `../uploads/a.xlsx`；正文里其它技能名字符串不受影响（验证前缀精确匹配，不是全局替换）。
- `SkillActivationService.activate()`：
  - 正常加载一个 WORKFLOW 技能，返回文本包含路径替换后的内容。
  - 技能不存在 → 抛 `SkillNotFoundError`。
  - 技能存在但 `category` 不在 `allowed_categories` → 同样抛 `SkillNotFoundError`
    （不泄露存在性）。
  - 技能存在但 `kind is TOOL` → 抛 `SkillDefinitionInvalidError`。
- `load_skill` 自包含实现（`create_load_skill_tool` 的 `_invoke`，subagent 路径真正
  执行的那份）：
  - Guardrail 拒绝某个具体 `skill_name` 时返回拒绝理由，不实际加载正文。
  - 两个不同技能名分别调用，Guardrail 收到的 `tool_name` 分别是各自的
    `skill_name`，不是恒定的 `"load_skill"`。
- `SkillMiddleware`（Lead Agent 路径真正执行的那份，`test/agents/test_skill_middleware.py`）：
  - `awrap_model_call`：存在 `WORKFLOW` 技能时把目录注入 `system_prompt`；不存在时
    整段跳过，不留一个只有标题没有条目的空 `<skill_catalog>`。
  - `awrap_tool_call`：非 `load_skill` 调用原样透传给 `handler`（验证不会误伤其它
    工具）；`load_skill` 调用短路返回，`handler` 全程不被调用（验证"不放行给
    `load_skill` 自己的实现"这条不是纸面设计）；Guardrail 校验按具体 `skill_name`
    而不是恒定的 `"load_skill"`；Guardrail 拒绝时不会触达 `SkillActivationService`。
- `SkillToolProvider.discover()`：
  - 全 TOOL 形态注册表 → 不产生 `load_skill` 条目。
  - 混合注册表 → TOOL 技能各自一条定义 + 恰好一条 `load_skill` 定义。
  - 热重载场景：新注册表技能形态从 WORKFLOW 变成 TOOL（假设有人给一个 public 技能加了
    `parameters`）→ 下一次 `discover()` 结果里它从"被 `load_skill` 覆盖"变成"独立
    `ToolDefinition`"，验证形态判定是现算的，不是缓存的。
- `SkillRegistry.register()` 重复 `tool_name` → 抛 `DuplicateSkillError`（阶段 5，独立测试）。
- 端到端（人工，不写自动化）：真实调一次 `data-analysis` 技能，确认
  `load_skill("data-analysis")` 返回的指令里脚本路径是可执行的绝对路径，再手动照着
  指令跑一次 `run_command`，确认脚本真的能跑通（验证六节的路径修复不是纸上谈兵）。

## 十一、与 GPT 原方案的偏差说明

| GPT 方案主张 | 本设计的处理 | 原因 |
|---|---|---|
| `SkillDefinition` 删除 `tool_name`/`parameters`/`args_schema`，Skill 不再是 Tool | 保留全部字段，新增 `kind` 派生属性区分 | 本项目 2 个 core 技能是真实参数化业务工具，删掉这些字段会破坏它们；GPT 方案假设"技能=指令"这个前提只对本项目一半的技能成立（见 1.2） |
| `required_tools`/`allowed_tools` + `available_tool_names` 交集依赖校验器 | 不引入，复用既有 `category` 路由 | `category` 已经保证"技能只会出现在真正装了它依赖的工具的那个 Agent 手里"（web_search 分类技能只挂
`web-researcher`，那里保证有 `web_search`/`web_fetch`）；新增一套依赖声明+运行时交集校验是重复约束同一件事，且要求新增 `configurable["available_tool_names"]` 这条本项目当前不存在的透传通道 |
| 独立 `execute_skill_script` 工具，带路径逃逸/allowlist 校验 | 不引入，复用已有 `run_python`/`run_command` | 这两个工具已经是 Lead Agent 自己的工具、已经在完整中间件链上（Guardrail/LoopDetection/ErrorHandling 全覆盖），新增一个平行的技能专属执行器只会制造两条要维护的执行路径 |
| SKILL.md frontmatter 迁移到 `metadata:` 嵌套结构 | 不迁移，保持现有扁平结构 | 现有 `parameters`/`runtime_context_keys`/`required_secrets` 都是顶层字段且工作良好，嵌套化要求重写全部 12 个 SKILL.md，换不来任何本设计需要的新能力 |
| `search_skills(query)` 检索式技能发现 | 不引入，全量注入系统提示词 | 12 个技能、每条一行 name+description 的 token 成本可忽略；技能数量到两位数以上再考虑，避免过早引入检索层的复杂度 |
| "同名覆盖应改成报错" | 采纳（第八节），作为独立加固项 | 与本设计其余部分正交，本身是合理的启动期防御，成本低 |
| `load_skill` 是一个普通 `StructuredTool`，校验+加载逻辑写在它自己 `coroutine` 里 | 采纳的是"模型看到的仍是一个 Tool"，但**不采纳**"逻辑写在 Tool 里"——Lead Agent 路径改由 `SkillMiddleware.awrap_tool_call` 拦截并短路，`StructuredTool.coroutine` 只在 subagent（无中间件）路径上才真正执行，见第四节 | 用户明确指出 Skill 不该"伪装成业务 Tool"，权限校验/加载这类横切逻辑更适合放中间件（同 `GuardrailMiddleware` 的既有模式）；但 `sub_agent_factory.py` 现造的子 Agent 不挂任何中间件是既有的、有意为之的安全边界，不在本次改动范围内，因此这条路径必须保留一份自包含实现，不能完全去掉 Tool 版本 |

## 十二、一句话结论

`load_skill` 收敛只对本项目里"纯指令、无参数"的 10 个 public 技能成立，2 个
core 参数化技能保持原状；这次重构的真正价值一半在"减少工具 schema + 系统提示词
可见性"，另一半在**顺带修复了一个此前从未被发现的、10 个技能里至少 7 个实际不可用的
路径断裂问题**——后者比 GPT 方案本身讨论的"工具目录污染"更紧迫。

## 十三、实现落地记录：与本文档设计的实际偏差

1. **`load_skill` 从"纯 Tool"改成"Middleware 优先 + Tool 兜底"**（第四节已经是更新
   后的版本，这里只记录触发原因）：最初设计只有 §4 的 Tool 版本，用户看完后指出"Skill
   不该伪装成业务 Tool，应该考虑 Middleware"，实现阶段确认 `sub_agent_factory.py` 的
   subagent 无中间件这一约束后，定型为两条路径共存的最终形态，不是简单二选一。
2. **`_LEAD_AGENT_SKILL_CATEGORIES` 提升为公开常量** `LEAD_AGENT_SKILL_CATEGORIES`
   （从 `providers.py` 模块私有改为 `agent_core.tools.registry` 包导出）：
   `lead_agent.py` 组装 `SkillMiddleware` 时需要同一份分类集合，原来的下划线私有约定
   不再成立。
3. **`SkillToolProvider.__init__` 新增 `guardrail_provider` 参数**：`load_skill` 的
   自包含兜底实现需要它，`main.py`/`SkillHotReloader.__init__`（同步新增
   `guardrail_provider` 参数）两个调用点同步改动。
4. **实现阶段扫描全部 12 个技能的激活结果，发现两处第六节设计时没预料到的裸
   `` /mnt/user-data `` 引用**（不带子路径，纯说明性文字），`SkillPathRewriter` 的
   三条替换规则覆盖不到，已手动改写源文件，验证方法（"扫描全部 WORKFLOW 技能激活结果
   断言不含 `/mnt/` 残留"）本身也补进了自动化之外的一次性检查脚本。
5. **测试覆盖比设计文档原计划更完整**：新增 `test/agents/test_skill_middleware.py`
   （§4.2/4.3 的 Middleware 拦截行为）和 `test/tools/registry/test_providers.py`
   （`SkillToolProvider.discover()` 的 TOOL/WORKFLOW 分流，此前这个 Provider 完全没有
   独立单测覆盖）；`test/tools/registry/test_skill_hot_reload.py` 的既有技能改成
   `TOOL` 形态（补 `parameters`）以保持原有断言（按 `tool_name` 直接查
   `by_model_name`）不必因为 Skill 分流改动而重写，热重载机制本身的回归目的更纯粹。

全量测试：`python -m pytest test/ -q` → **279 passed**（较改动前的 234 新增 45 个，
覆盖本次新增的全部模块），无回归。

## 十四、一句话结论（实现后更新）

`load_skill` 在模型侧始终是一个工具 schema，但"谁来执行"分裂成两条路径——Lead Agent
交给 `SkillMiddleware` 处理，验证了"Skill 不是业务 Tool"这一更准确的语义；subagent
因为架构上没有中间件，只能保留一份自包含实现。这次重构除了解决 GPT 方案讨论的"工具
目录污染"，更大的隐藏价值是**顺带修复了一个此前从未被发现的、10 个 public 技能里至少
7 个此前实际不可用的路径断裂问题**——这个 bug 与要不要做 `load_skill`、要不要用
Middleware 都无关，是任何认真梳理这套 Skill 机制现状都会撞见的独立缺口。
