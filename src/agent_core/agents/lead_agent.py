"""Lead Agent 组装（设计文档 4.1 节）。

取代原项目"Supervisor 多图路由"的核心改动：不再有一个专门做结构化路由决策的
LLM 调用，Lead Agent 自身就是一个普通的 `create_agent`，工具集里绝大部分都是可以
直接执行的基础工具（记忆、沙箱读写/执行、`search_knowledge_base`/`query_database`
等技能、前端 `extra_tools`），只有仍然需要"子 Agent 做多轮判断再回话"的能力
（目前只有联网搜索）走 `task(subagent_type, task)` 这一个通用分发工具——路由
降级为一次普通的工具调用决策，而不是每轮都必须做的独立 LLM 调用。

`rag`/`database` 曾经也各自包过一层 `delegate_to_xxx` 委派工具，现在改成直接挂：
`search_knowledge_base`/`query_database` 本身就是技能（分类 `rag`/`database`），
委派层纯属多余的一层包装，且委派子 Agent 不挂中间件、执行结果对外层不可见，会让
`chat_pipeline.py` 的引用来源解析（`<ref_json>` 状态机）拿不到数据——直接挂了之后
这两个技能才第一次真正对外层可见。

沙箱工具（`write_file`/`read_file`/`run_python`/`run_command`，见
`agent_core/tools/sandbox_tool.py`）挂在 Lead Agent 自己的工具集而不是包装成
委派工具，是特意的选择：委派工具背后的子 Agent（`sub_agent_factory.py`）不挂
任何中间件，"写代码→跑→改→再跑"这类迭代循环如果发生在子 Agent 内部，会完全
跑在死循环检测（`LoopDetectionMiddleware`）和权限校验（`GuardrailMiddleware`）
之外；放在 Lead Agent 自己的工具集里，每一次调用都会经过完整的中间件链。

`thinking_enabled` 是运行时配置，不是另一套并行维护的 Agent 构建代码：复用同一套
委派工具，只是模型侧开启扩展推理 + 调用方在 `.ainvoke(..., config={"recursion_limit": ...})`
传入更大的步数预算。

本函数不做单例缓存，每次调用都现造（`create_agent` 构造成本是纯 Python 对象组装，
没有网络调用，可以忽略）——这样委派工具/技能工具集/`web_search_agent` 的当天日期
都能拿到最新状态，不会像原项目那样把 `{current_date}` 冻结在进程启动的那一刻。
"""
from __future__ import annotations

from langchain.agents import create_agent
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from src.agent_core.agents.datasource_routing_middleware import DatasourceRoutingMiddleware
from src.agent_core.agents.delegation_tools import build_task_tool
from src.agent_core.agents.streaming_model_middleware import StreamingModelMiddleware
from src.agent_core.agents.subagent_profiles import list_subagent_types
from src.agent_core.guardrail import get_guardrail_provider
from src.agent_core.loop import TitleModelSettings, build_middlewares
from src.agent_core.memory import get_memory_manager
from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.model import create_chat_model
from src.agent_core.prompts import prompt_factory
from src.agent_core.prompts.system_prompt_builder import system_prompt_builder
from src.agent_core.sandbox import get_sandbox_provider
from src.agent_core.skills import get_skill_manager
from src.agent_core.tools.memory_tools import recall_memory, save_memory
from src.agent_core.tools.sandbox_tool import read_file, run_command, run_python, save_output_file, write_file
from src.agent_core.workspace import get_thread_workspace_manager
from src.config.settings import get_settings
from src.storage.conversation_store import get_conversation_store

# 编译出的图里，一次"模型响应工具调用 → 实际执行"往返固定消耗 3 个
# recursion 单位（`InputSanitizationMiddleware.before_model` -> `model` ->
# `tools`，工具执行完再绕回 `InputSanitizationMiddleware.before_model`，见
# `agent.get_graph()` 的实测节点/边），首尾还有约 6 个固定开销的中间件节点。
# 旧值 25/40 在没有沙箱工具时够用（委派工具把多步操作封在子 Agent 内部，
# 只算 Lead Agent 自己的 1 次工具调用）；有了 write_file/run_python 之后，
# 迭代写代码调试是常见任务形状，一次"写→跑→看报错→改→再跑"就要好几轮，
# 25 很容易在真实调试任务里被打满（实测一次 3 轮的 quicksort 调试正好卡在
# 25——`GRAPH_RECURSION_LIMIT` 报错）。这里改成按"能扛多少轮工具调用"倒推：
# `LoopDetectionMiddleware` 已经防住了真正的死循环（连续 3 次完全相同调用
# 会被短路），recursion_limit 现在只是防失控任务无限烧 token 的兜底上限，
# 不是"每轮任务的预算"，可以放宽。
_DEFAULT_RECURSION_LIMIT = 75  # ≈ 24 轮工具调用
_THINKING_RECURSION_LIMIT = 150  # ≈ 48 轮工具调用，深度思考模式预算更大


def build_lead_agent(
    *,
    thinking_enabled: bool = False,
    extra_tools: list[BaseTool] | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    user_id: str | None = None,
):
    """组装一个 Lead Agent。

    Args:
        thinking_enabled: 是否开启深度思考模式（更大推理预算 + 模型侧扩展推理）。
        extra_tools: 前端本次请求注入的工具（经 `CustomToolConverter` 转换），
            并入 Lead Agent 自己的基础工具集，对应原 `tool_agent` 的职责。
        checkpointer: 会话持久化 checkpointer，为空时该 Agent 不具备跨轮记忆
            （仅供单次调用/测试场景使用）。
        user_id: 归属用户 ID，用于 `TitleMiddleware` 生成标题后持久化到
            `conversation_store`；为空时标题只记日志，不落库。

    Returns:
        编译好的 Lead Agent（`CompiledStateGraph`），可直接
        `.ainvoke({"messages": [...]}, config)` 调用。
    """
    skill_manager = get_skill_manager()
    settings = get_settings()

    async def _persist_title(conversation_id: str, title: str) -> None:
        if user_id:
            await get_conversation_store().update_title(conversation_id, user_id, title)

    base_tools: list[BaseTool] = [
        save_memory,
        recall_memory,
        write_file,
        read_file,
        run_python,
        run_command,
        save_output_file,
        *skill_manager.get_tools("general"),
        *skill_manager.get_tools("tool"),
        *skill_manager.get_tools("rag"),
        *skill_manager.get_tools("database"),
        *(extra_tools or []),
        build_task_tool(),
    ]

    skill_enabled = bool(
        skill_manager.get_tools("general") or skill_manager.get_tools("tool")
        or skill_manager.get_tools("rag") or skill_manager.get_tools("database")
    )

    middlewares = [
        *build_middlewares(
            guardrail_provider=get_guardrail_provider(),
            sandbox_provider=get_sandbox_provider(),
            workspace_manager=get_thread_workspace_manager(),
            memory_manager=get_memory_manager(),
            prompt_factory=prompt_factory,
            title_model_settings=TitleModelSettings(
                model_name=settings.DEFAULT_MODEL,
                provider=settings.PROVIDER,
                api_key=settings.API_KEY,
                base_url=settings.BASE_URL,
            ),
            on_title_generated=_persist_title,
        ),
        # Lead Agent 专属业务规则，不是通用中间件流水线的一部分，追加在最后。
        DatasourceRoutingMiddleware(),
        # 必须是最内层（离真实模型调用最近）：见该模块说明。
        StreamingModelMiddleware(),
    ]

    return create_agent(
        model=create_chat_model(thinking_enabled=thinking_enabled),
        tools=base_tools,
        system_prompt=system_prompt_builder.build(
            "lead_agent",
            thinking_enabled=thinking_enabled,
            subagent_enabled=bool(list_subagent_types()),
            skill_enabled=skill_enabled,
        ),
        middleware=middlewares,
        context_schema=AgentRuntimeContext,
        checkpointer=checkpointer,
    )


def resolve_recursion_limit(thinking_enabled: bool) -> int:
    """按是否深度思考模式返回本次调用应使用的 `recursion_limit`。"""
    return _THINKING_RECURSION_LIMIT if thinking_enabled else _DEFAULT_RECURSION_LIMIT
