# raster-agent 长期记忆重构设计

> 状态：设计稿（Memory v2）  
> 依据：当前 raster-agent 代码、DeerFlow 三层记忆与异步更新流水线资料  
> 核心决策：L3 Facts 按 DeerFlow 方式维护——LLM 输出显式事实操作；ES 仅用于候选检索和在线召回，不按向量相似度自动判定冲突或归档。

## 1. 背景与目标

当前系统已有三层长期记忆：L1 用户画像、L2 用户时间线、L3 可语义检索的 Facts。Agent 回合结束后，MemoryExtractionMiddleware 同时发起 UserProfileUpdater（L1/L2）和 MemoryExtractor（L3）两个后台 LLM 任务；Profile 写入 PostgreSQL，Facts 写入 Elasticsearch。

这一版本成本低且易接入，但在连续对话、并发更新、任务可靠性和记忆可解释性上已有明确边界：

- 每回合两次独立 LLM 调用，成本高，L1/L2 与 L3 可能得出互相不一致的结论。
- asyncio.create_task() 是进程内 best-effort；进程退出或 Worker 回收会丢失更新。
- Profile 读取旧六字段后整体重写，可能发生并发覆盖。
- L3 当前将向量相似度区间用于冲突判定；相似不是矛盾，可能错误归档旧 Fact。
- Profile 无条件注入，Facts 按 query 召回，但二者没有统一 token 预算与单 Run 缓存。
- Profile 更新路径没有字段级来源、证据、版本和统一敏感信息过滤。

Memory v2 的目标：

1. 以统一的 MemoryDelta 表达本轮对长期记忆的变化。
2. 将记忆更新改为持久化、可去重、可重试、可恢复的异步作业。
3. LLM 只做语义判断；代码负责 Schema、权限、去重、事务、审计和落库。
4. 保留 ES 的 kNN + Reranker 检索能力，但移除相似度自动冲突归档。
5. 为用户记忆提供清晰的来源、状态、人工确认与删除能力。

非目标：不替换 LangGraph Checkpoint；不替换现有 ES 检索能力；不让模型拥有无约束的记忆删除权；第一阶段不强制让所有 Sub-Agent 注入长期记忆。

## 2. 当前实现与问题

### 2.1 当前链路

~~~mermaid
flowchart TD
    A["Agent 回合结束"] --> B["MemoryExtractionMiddleware"]
    B --> C["最后一条 User 消息 + AI 回复"]
    C --> D["asyncio.create_task"]
    D --> E["UserProfileUpdater: L1/L2"]
    D --> F["MemoryExtractor: L3 Facts"]
    E --> G["PostgreSQL user_memory_profile"]
    F --> H["Elasticsearch Memory Index"]
    I["下次模型调用"] --> J["Profile 无条件注入"]
    I --> K["Facts kNN + Rerank"]
    J --> L["system_prompt"]
    K --> L
~~~

### 2.2 已有能力及保留策略

| 能力 | 当前实现 | v2 策略 |
|---|---|---|
| L1/L2 | PostgreSQL 一用户一行、六个摘要字段 | 保留，增加 revision、字段来源和字段更新时间 |
| L3 分类 | preference、knowledge、context、behavior、goal | 保持 DeerFlow 一致 |
| L3 状态 | active、pending、archived、过期派生 | 保留，pending 为企业可控扩展 |
| 检索 | ES kNN 粗召回 + Reranker 精排 | 保留，ES 改为可重放索引投影 |
| 安全 | Facts 写入前敏感信息检查 | 扩展至 Profile、事件和 Delta |
| 审计 | CRUD、注入、压缩审计 | 扩展为事件、Delta、投影审计 |
| 陈旧治理 | 低价值、未命中记忆定期归档 | 保留并补充显式替代链 |

### 2.3 问题到方案映射

| 当前问题 | 根因 | v2 应对 |
|---|---|---|
| 更新任务丢失 | 非持久化 create_task | memory_update_job + Worker + 重试 |
| 连续对话重复提取 | 每回合即时处理 | watermark + 防抖合并 |
| Profile 覆盖 | 整行无版本 upsert | Profile patch + 乐观锁 |
| L1/L2 和 L3 不一致 | 两次独立 LLM 调用 | 一次 LLM 输出统一 Delta |
| 相似即冲突 | embedding 阈值承担事实判断 | 只允许显式 supersede/archive |
| 上下文膨胀 | 两条注入链独立预算 | 统一 MemoryContextBuilder |
| 记忆污染 | 工具、上传、临时资源可进入抽取 | 捕获阶段过滤和内容策略 |

## 3. 总体架构

~~~mermaid
flowchart LR
    A["MemoryCaptureMiddleware"] --> B["memory_event"]
    B --> C["memory_update_job"]
    C --> D["MemoryUpdateWorker"]
    D --> E["读取 Profile 快照"]
    D --> F["检索相关 Facts 候选"]
    E --> G["LLM Structured Output: MemoryDelta"]
    F --> G
    G --> H["Delta Validator"]
    H --> I["MemoryApplyEngine"]
    I --> J["Profile Projector"]
    I --> K["Fact Projector"]
    J --> L["PostgreSQL"]
    K --> L
    L --> M["memory_outbox"]
    M --> N["ES Index Projector"]
    N --> O["Elasticsearch"]
    P["MemoryInjectionMiddleware"] --> L
    P --> O
~~~

| 组件 | 职责 | 明确不负责 |
|---|---|---|
| Capture Middleware | 过滤对话、创建事件和作业 | 调 LLM、直接改记忆 |
| Update Worker | 领取任务、构造上下文、调用 LLM、重试 | 绕过 Apply Engine 直接写数据 |
| MemoryUpdater | 将对话变成 MemoryDelta | 决定最终 SQL/ES 写入细节 |
| Apply Engine | 幂等、校验、合并、事务和 Outbox | 语义猜测 |
| Profile Projector | 应用 L1/L2 patch | 从原始对话推理事实 |
| Fact Projector | 应用 Fact 操作 | 通过 ES 相似度自行冲突归档 |
| ES Projector | 构建可检索向量投影 | 作为事实真相源 |
| Context Builder | 选择、预算、渲染记忆 | 修改持久化记忆 |

## 4. 数据模型

### 4.1 Memory Event

memory_event 是清洗后的不可变输入，既不是 Fact，也不直接注入模型。

~~~sql
CREATE TABLE memory_event (
    event_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    agent_name TEXT NULL,
    trace_id TEXT NULL,
    from_message_id TEXT NOT NULL,
    to_message_id TEXT NOT NULL,
    conversation_json JSONB NOT NULL,
    content_hash TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (user_id, conversation_id, from_message_id, to_message_id)
);
~~~

conversation_json 只保存用户原话与最终、无悬挂 Tool Call 的 AI 回复。必须剔除 ToolMessage、工具参数、内部推理、上传文件标签、临时路径、工作区路径和其他会话级资源。

### 4.2 Memory Update Job

~~~sql
CREATE TABLE memory_update_job (
    job_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    conversation_id TEXT NOT NULL,
    agent_name TEXT NULL,
    first_event_id UUID NOT NULL,
    last_event_id UUID NOT NULL,
    idempotency_key TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL,
    attempts INT NOT NULL DEFAULT 0,
    next_retry_at TIMESTAMPTZ NULL,
    locked_until TIMESTAMPTZ NULL,
    last_error TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ NULL
);
~~~

状态机：

~~~text
pending → processing → succeeded
                   ├→ pending（可重试失败）
                   └→ dead（不可恢复失败）
~~~

同一 user_id + conversation_id + agent_name 的相邻事件可在防抖窗口内合并。watermark 保证只处理未消费的新消息，避免反复向 LLM 发送完整会话。

### 4.3 L1/L2 Profile

保留现有六字段，但从整行覆盖升级为字段级元数据加行 revision。

~~~sql
CREATE TABLE user_memory_profile (
    user_id TEXT PRIMARY KEY,
    work_context TEXT NOT NULL DEFAULT '',
    personal_context TEXT NOT NULL DEFAULT '',
    top_of_mind TEXT NOT NULL DEFAULT '',
    recent_months TEXT NOT NULL DEFAULT '',
    earlier_context TEXT NOT NULL DEFAULT '',
    long_term_background TEXT NOT NULL DEFAULT '',
    revision BIGINT NOT NULL DEFAULT 0,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE user_memory_profile_field_meta (
    user_id TEXT NOT NULL,
    field_name TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    source_event_id UUID NOT NULL,
    confidence NUMERIC(3,2) NOT NULL,
    PRIMARY KEY (user_id, field_name)
);
~~~

六字段保持不变：work_context、personal_context、top_of_mind、recent_months、earlier_context、long_term_background。

### 4.4 L3 Facts

L3 保持 DeerFlow 的五类事实模型。PostgreSQL 为规范化事实主存，ES 是向量检索投影。

~~~sql
CREATE TABLE user_memory_fact (
    fact_id UUID PRIMARY KEY,
    user_id TEXT NOT NULL,
    agent_name TEXT NULL,
    content TEXT NOT NULL,
    normalized_content TEXT NOT NULL,
    category TEXT NOT NULL,
    importance SMALLINT NOT NULL,
    confidence NUMERIC(3,2) NOT NULL,
    status TEXT NOT NULL,
    source_event_id UUID NOT NULL,
    source_conversation_id TEXT NULL,
    evidence_message_ids JSONB NOT NULL DEFAULT '[]',
    supersedes_fact_id UUID NULL,
    superseded_by_fact_id UUID NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NULL,
    last_accessed_at TIMESTAMPTZ NULL,
    access_count BIGINT NOT NULL DEFAULT 0,
    revision BIGINT NOT NULL DEFAULT 0
);

CREATE UNIQUE INDEX ux_active_fact_normalized
ON user_memory_fact(user_id, COALESCE(agent_name, ''), category, normalized_content)
WHERE status IN ('active', 'pending');
~~~

类别：preference、knowledge、context、behavior、goal。状态：pending、active、archived、superseded、expired。

## 5. MemoryDelta 合约

### 5.1 原则

LLM 不生成整份 Memory，也不能直接覆盖数据库。它只输出建议发生哪些变更的严格 JSON Schema。

~~~text
LLM：语义判断和变更意图
代码：合法性、幂等、去重、事务、权限与审计
~~~

### 5.2 示例

~~~json
{
  "schema_version": 1,
  "source_event_ids": ["e-001"],
  "profile_patches": [
    {
      "field": "top_of_mind",
      "op": "merge",
      "value": "正在重构 Agent 的 Skill 加载方式和长期记忆更新链路。",
      "confidence": 0.94,
      "evidence_message_ids": ["m-101"]
    }
  ],
  "fact_operations": [
    {
      "op": "add",
      "content": "用户计划将 Skill 设计为按需加载的工作流上下文，而不是每个 Skill 都注册为业务 Tool。",
      "category": "goal",
      "importance": 8,
      "confidence": 0.95,
      "evidence_message_ids": ["m-101"]
    }
  ]
}
~~~

### 5.3 Profile Patch

~~~text
field: 合法的六个 Profile 字段之一
op: set | merge | clear
value: 目标字段文本
confidence: 0.0～1.0
evidence_message_ids: 非空，且必须属于当前输入
~~~

clear 仅限用户明确纠正或删除要求；不能因为本轮没有再次提到某背景而清空旧画像。

### 5.4 Fact Operation

~~~text
op: add | update | supersede | archive | reinforce
~~~

| 操作 | 规则 |
|---|---|
| add | 必须包含内容、类别、置信度和证据 |
| update | 必须显式给出 target_fact_id |
| supersede | 必须给出旧 Fact ID、新 Fact、理由和证据 |
| archive | 必须给出目标 Fact ID 和明确证据 |
| reinforce | 不新增 Fact，只补充确认时间、置信度或证据 |

## 6. L3 Facts：保持 DeerFlow 的去重与冲突边界

### 6.1 强制规则

**禁止以 ES 向量相似度区间直接判定冲突或自动归档旧 Fact。**

ES 相似度只用于：

1. 找出供 LLM 参考的可能相关或重复候选 Fact。
2. 在用户提问时检索 active Facts。

它不能推出：

~~~text
相似 ≠ 重复
相似 ≠ 矛盾
不相似 ≠ 没有逻辑矛盾
~~~

### 6.2 确定性精确去重

add 仅按规范化文本唯一键去重：

~~~python
def normalize_fact(content: str) -> str:
    return " ".join(content.strip().split())
~~~

在相同 user_id + agent_name + category 范围内，normalized_content 相同则跳过 add 或转换为 reinforce。该规则可解释、可测试，且不依赖 embedding 阈值。

### 6.3 显式替代

用户明确修正事实时，LLM 才能输出：

~~~json
{
  "op": "supersede",
  "target_fact_id": "old-fact-id",
  "content": "用户目前在 B 公司担任后端工程师。",
  "category": "context",
  "confidence": 0.97,
  "reason": "用户明确说明已经从 A 公司离职并加入 B 公司。",
  "evidence_message_ids": ["m-201"]
}
~~~

事务内依次：校验旧 Fact 归属和状态；插入新 Fact；旧 Fact 标记 superseded 并写 superseded_by_fact_id；新 Fact 写 supersedes_fact_id；再写审计和 Outbox。

### 6.4 Pending

保留当前系统比 DeerFlow 更可控的 pending 扩展：

~~~text
confidence < 0.70          → 丢弃
0.70 ≤ confidence < 0.85   → pending
confidence ≥ 0.85          → active
~~~

用户明确说“记住、以后请、务必遵守”时可以提升为 goal 候选，但仍必须经过敏感信息、证据和策略校验。pending 不参与普通召回，确认后才转为 active。

## 7. 更新流水线

### 7.1 Capture

MemoryCaptureMiddleware.aafter_agent()：

1. 校验 user_id、conversation_id 和记忆开关。
2. 读取 watermark 后的增量消息。
3. 仅保留 User 消息与最终 AI 回复。
4. 清除工具输出、上传块、临时路径、内部状态和敏感信息。
5. 对纯问候、无事实信息或只有工具循环的回合跳过。
6. 在同一数据库事务中写 memory_event 和 memory_update_job。

Capture 不调用 LLM，不直接修改 Profile/Facts。

### 7.2 Queue 与 Worker

防抖降低成本；可靠性由持久化队列、租约、重试和幂等键保证。Worker 领取任务：

~~~sql
SELECT job_id
FROM memory_update_job
WHERE status = 'pending'
  AND (next_retry_at IS NULL OR next_retry_at <= NOW())
  AND created_at <= NOW() - INTERVAL '30 seconds'
ORDER BY created_at
FOR UPDATE SKIP LOCKED
LIMIT 1;
~~~

Worker 应支持指数退避、租约超时回收和 dead-letter。进程内 asyncio.create_task()、threading.Timer 不能作为生产可靠性边界。

### 7.3 构造更新上下文

每个 Job 读取：

~~~text
Profile 六字段 + revision
清洗后的新增对话
与新增对话相关的 10～20 条 active/pending Fact 候选
候选 Fact ID、类别、内容、状态
~~~

候选优先级：规范化文本精确匹配、语义高相关 active Fact、近期被用户明确确认或修正的 Fact。禁止把全量 Facts 放进 Prompt。

### 7.4 LLM 和校验

优先使用模型的 JSON Schema、structured output 或 function calling；json.loads() 仅作兼容性降级。

校验顺序：

1. JSON 可解析。
2. Pydantic Schema、枚举、长度和置信度范围正确。
3. target_fact_id、evidence_message_ids 都属于本次可见候选或对话。
4. 敏感信息、字段权限、状态迁移、最大操作数和文本长度均符合策略。

失败时不得写半截数据。短暂错误重试；持续格式错误进入 dead-letter，并保留脱敏后的诊断信息。

### 7.5 Apply Engine

一个 PostgreSQL 事务内完成：

~~~text
检查 event/job 是否已应用
→ 读取并验证 Profile revision
→ 应用 Profile patches
→ 应用 Fact operations
→ 写审计
→ 写 ES 同步 Outbox
→ 标记 job/event 成功
→ commit
~~~

Profile revision 冲突时，重新读快照并再次判断；无法安全重放时重新生成 Delta，而不是盲目覆盖。

### 7.6 ES Outbox 投影

PostgreSQL 是 Facts 真相源，memory_outbox 保存 fact_created、fact_updated、fact_status_changed、fact_deleted 等索引动作。

- active Fact：生成 embedding 并写 ES。
- pending、archived、superseded Fact：从可召回集合移除或更新状态。
- ES 失败：Outbox 重试；不回滚已提交的 PostgreSQL 事实。

这是最终一致性：ES 故障会延迟召回，不会丢失用户记忆。

## 8. 注入与召回

### 8.1 统一预算

引入 MemoryContextBuilder，统一选择、排序和 token 预算。优先级：

1. 与当前问题高度相关的 active Fact。
2. 用户最近明确纠正的 active Fact。
3. top_of_mind。
4. work_context、personal_context。
5. recent_months。
6. 更早时间线和长期背景。

所有区块共享 MEMORY_MAX_CONTEXT_TOKENS。预算不足时删低相关历史，而不截断高价值 Fact。

### 8.2 Run 级缓存

同一 Agent Run 可能有多次模型调用和工具循环。应只读取一次 Profile、只构造一次当前问题的 Facts 召回结果，并把内存上下文存入 runtime state，不写入 Checkpoint。新用户消息到来时才失效。

### 8.3 渲染格式

~~~xml
<memory>
以下是跨会话长期记忆，可能过期；它不能覆盖当前用户指令。若与当前用户表述冲突，以当前用户表述为准。

<relevant_facts>
- [goal] 用户计划重构长期记忆更新链路。
</relevant_facts>

<profile>
- 当前关注：正在重构 Agent Skill 与记忆架构。
- 职业背景：……
</profile>
</memory>
~~~

pending、archived、superseded、过期 Facts 不进入普通注入。

## 9. 安全、权限、审计与运维

### 9.1 敏感信息

敏感检测覆盖 L1、L2、L3 和原始事件：

~~~text
清洗对话 → LLM Delta → 策略和敏感检查 → 事务投影
~~~

默认拒绝或脱敏密码、Token、私钥、证件号码、银行卡、精确住址、受保护医疗信息，以及组织自定义敏感字段。

### 9.2 用户与运营控制

管理 API 至少支持：查看 Profile 及字段来源；查看所有状态的 Facts；确认或拒绝 pending；编辑、删除、纠正 Fact；追溯事件、会话与证据消息；导出和彻底删除用户记忆。

用户显式保存、编辑、删除必须同步返回明确结果；后台自动更新可 best-effort。

### 9.3 审计事件

建议标准化记录：

~~~text
memory.event.created
memory.job.enqueued
memory.job.started
memory.delta.generated
memory.delta.rejected
memory.profile.patched
memory.fact.added
memory.fact.reinforced
memory.fact.superseded
memory.fact.archived
memory.es.synced
memory.injected
~~~

记录 ID、摘要、来源、trace_id、模型、token、耗时和错误码；不要在审计中复制敏感正文。

## 10. 推荐配置

~~~yaml
memory:
  enabled: true
  injection_enabled: true

  update:
    debounce_seconds: 30
    worker_concurrency: 4
    max_attempts: 5
    retry_backoff_seconds: [30, 120, 600, 1800]
    max_events_per_job: 20
    max_conversation_chars: 12000

  facts:
    discard_confidence_threshold: 0.70
    active_confidence_threshold: 0.85
    max_active_facts_per_scope: 100
    semantic_auto_conflict: false
    normalize_whitespace_dedup: true

  retrieval:
    candidate_k: 20
    max_recall: 5
    min_recall_score: 0.35
    max_update_candidate_facts: 15

  injection:
    max_context_tokens: 2000
    cache_per_run: true
~~~

semantic_auto_conflict 应固定为 false：ES 相似度不可重新成为自动归档依据。

## 11. 一致性与失败边界

| 场景 | 处理 | 是否影响主对话 |
|---|---|---|
| 自动记忆入队失败 | 记录并重试/降级 | 否 |
| LLM 或 Worker 暂时失败 | 任务延迟重试 | 否 |
| Delta 非法 | 不应用，dead-letter 或重试 | 否 |
| PostgreSQL 事务失败 | 不完成 Job，重试 | 否 |
| ES 同步失败 | PostgreSQL 已保存，Outbox 重试 | 否，召回可能延迟 |
| 敏感内容 | 拒绝对应 patch/Fact 并审计 | 否 |
| 用户显式保存失败 | 返回明确失败 | 是，必须可见 |
| Profile revision 冲突 | 重读快照并重算或重放 | 否 |

## 12. 分阶段迁移

### Phase 0：观测基线

- 统计现有自动记忆成功率、更新耗时、每回合 LLM 调用数、重复 Fact 率和误归档样本。
- 不改变线上业务行为。

### Phase 1：持久化作业

- 新增 memory_event、memory_update_job 和 Worker。
- MemoryExtractionMiddleware 从直接 create_task() 改为入队。
- Worker 暂时复用旧 UserProfileUpdater、MemoryExtractor。
- 验证任务恢复、会话合并、幂等、重试和观测。

### Phase 2：统一 Delta

- 实现 Pydantic MemoryDelta 与 MemoryApplyEngine。
- 两次独立 LLM 调用改为一次结构化调用。
- Profile 改为 patch + revision。
- 删除当前 MemoryExtractor._detect_conflict() 中基于相似度区间自动归档的分支。

### Phase 3：Facts 规范化主存与 ES Outbox

- 创建 user_memory_fact，迁移现有 ES 文档并保留 ID、状态、来源和时间。
- PostgreSQL 成为主存，ES 转为异步索引投影。
- 双读校验后切换读取主路径。

### Phase 4：注入治理与人工闭环

- 合并 Profile/Facts token 预算，增加 Run 级缓存和注入 trace。
- 上线 pending 审核、用户纠正、来源展示和遗忘评测。

## 13. 验收标准

### 可靠性

- 记忆任务最终完成率不低于 99.5%。
- 重启后未完成任务可恢复率为 100%。
- 同一 idempotency_key 不产生重复事实。
- Profile 并发覆盖造成的数据丢失率为 0。

### 质量

- 不再发生仅因 embedding 相似度而自动归档旧 Fact。
- 规范化文本重复 Fact 写入率低于阈值。
- 用户显式记住请求的保存成功率不低于 99%。
- pending 的确认率、拒绝率、存量和时延可观测。

### 性能

- 高频连续对话的自动记忆 LLM 调用较逐回合更新下降 30% 以上。
- 主对话 P95 延迟不因后台记忆更新上升。
- 单次注入不超过 max_context_tokens。
- 同一 Agent Run 的 Profile/Facts 上下文至多构造一次。

## 14. 结论

本设计不照搬 DeerFlow 早期的 memory.json + Timer，而是保留其最有价值的原则：

~~~text
异步批处理
+ 旧记忆与新对话共同参与判断
+ LLM 输出结构化更新意图
+ 确定性代码合并、清洗和持久化
~~~

再结合 raster-agent 已有的 PostgreSQL、Elasticsearch、Reranker、审计和多状态 Fact 能力，形成面向多 Worker 生产环境的记忆系统。

最终原则：

> Fact 的新增、修正、替代和归档必须由携带证据的 MemoryDelta 显式表达；ES 只提供候选与检索能力，不再替代事实判断。

