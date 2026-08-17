"""Lead Agent 中间件链的顺序约束：唯一的、可执行的声明。

这些约束原本分散写在 `loop.py`/`skill_middleware.py`/`plan_middleware.py`/
`lead_agent.py` 四处的模块文档和内联注释里——同一条规则被复述好几遍，且只有
"人记得读注释"这一层保障：真出现有人调整 `lead_agent.py` 里的中间件列表顺序、
不小心破坏了某条约束，不会有任何报错，只会在某个具体场景下（比如 Skill 正文
不再注入 system_prompt）表现出行为异常，很难跟"顺序错了"直接联系起来。

这里把约束收拢成一份声明式列表，`validate_middleware_order()` 在
`build_lead_agent()` 组装完列表后立即校验，顺序错误会当场抛异常，而不是等到
某次真实请求表现异常才被发现。

LangChain 中间件的 `awrap_model_call`/`awrap_tool_call` 都是洋葱模型：列表里
下标越小的实例在对应钩子上越靠外层（越先拿到请求、越晚看到返回结果）。下面
每条约束的语义是"前者的下标必须小于后者"，对应关系分别来自：

- `(GuardrailMiddleware, ToolErrorHandlingMiddleware)`：`loop.py` 模块文档——
  权限拒绝是短路（不执行），跟"执行过程中抛出的异常"是不同职责的两层包装，
  顺序不能颠倒。
- `(ToolAuditMiddleware/ToolErrorHandlingMiddleware/LoopDetectionMiddleware,
  SkillMiddleware)`：`skill_middleware.py` 模块文档——`load_skill` 是一次
  普通工具调用（`SkillMiddleware.awrap_tool_call` 拦截），需要继续被这三层
  通用治理（审计日志/异常兜底/死循环检测）覆盖，SkillMiddleware 必须是这条
  链路里更靠内层的一环。
- `(SkillMiddleware, StreamingModelMiddleware)`、
  `(PlanContextMiddleware, StreamingModelMiddleware)`：两者都要往
  `request.system_prompt` 里追加内容（技能正文/`<current_plan>` 块），必须
  发生在真正的模型调用之前；`StreamingModelMiddleware` 必须是整条链路里
  最贴近真实模型调用的一层（见其模块文档"必须是最内层"）。
"""
from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware

from src.agent_core.agents.plan_middleware import PlanContextMiddleware
from src.agent_core.agents.skill_middleware import SkillMiddleware
from src.agent_core.agents.streaming_model_middleware import StreamingModelMiddleware
from src.agent_core.middlewares.guardrail import GuardrailMiddleware
from src.agent_core.middlewares.loop_detection import LoopDetectionMiddleware
from src.agent_core.middlewares.tool_audit import ToolAuditMiddleware
from src.agent_core.middlewares.tool_error_handling import ToolErrorHandlingMiddleware

# (更靠外层的类型, 更靠内层的类型)：前者在最终列表里的下标必须小于后者。
ORDER_CONSTRAINTS: tuple[tuple[type[AgentMiddleware], type[AgentMiddleware]], ...] = (
    (GuardrailMiddleware, ToolErrorHandlingMiddleware),
    (ToolAuditMiddleware, SkillMiddleware),
    (ToolErrorHandlingMiddleware, SkillMiddleware),
    (LoopDetectionMiddleware, SkillMiddleware),
    (SkillMiddleware, StreamingModelMiddleware),
    (PlanContextMiddleware, StreamingModelMiddleware),
)


def validate_middleware_order(middlewares: list[AgentMiddleware]) -> None:
    """校验中间件列表满足 `ORDER_CONSTRAINTS` 里的每一条顺序约束。

    只按类型比较列表下标，不感知具体构造参数——足够在"列表顺序被意外调整"
    这类改动上快速失败，不需要真的跑一次 Agent 才能发现。不用 `assert`：
    `python -O` 会整段跳过 assert 语句，这里要的是不可关闭的硬校验。

    Args:
        middlewares: 组装完成、即将传给 `create_agent(middleware=...)` 的列表。

    Raises:
        RuntimeError: 任意一条约束被违反，报错信息直接点出违反的是哪一条
            类型、实际下标是多少，方便定位。约束涉及的两个类型只要有一个不在
            列表里就跳过该条（部分测试场景可能只装了部分中间件）。
    """
    positions = {type(m): i for i, m in enumerate(middlewares)}
    for outer_cls, inner_cls in ORDER_CONSTRAINTS:
        outer_pos = positions.get(outer_cls)
        inner_pos = positions.get(inner_cls)
        if outer_pos is None or inner_pos is None:
            continue
        if not outer_pos < inner_pos:
            raise RuntimeError(
                f"中间件顺序错误：{outer_cls.__name__}（下标 {outer_pos}）必须排在 "
                f"{inner_cls.__name__}（下标 {inner_pos}）之前，否则后者依赖的横切"
                f"治理（审计/异常兜底/死循环检测/权限校验）或 system_prompt 注入"
                f"时机会失效，见 middleware_order.py 顶部说明。"
            )
