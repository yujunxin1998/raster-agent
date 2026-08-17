"""`update_plan` 工具单元测试。"""
from __future__ import annotations

import json
import re
from types import SimpleNamespace

from langgraph.types import Command

from src.agent_core.middlewares.context import AgentRuntimeContext
from src.agent_core.tools.plan_tools import PlanStep, update_plan

_RUNTIME = SimpleNamespace(
    context=AgentRuntimeContext(conversation_id="conv-1", user_id="user-1"),
    tool_call_id="call-1",
)


async def test_update_plan_writes_serialized_steps_to_state() -> None:
    plan = [
        PlanStep(content="搜索 AWS 信息", status="in_progress"),
        PlanStep(content="搜索 Azure 信息", status="pending"),
    ]

    result = await update_plan.coroutine(plan=plan, runtime=_RUNTIME)

    assert isinstance(result, Command)
    assert result.update["plan"] == [
        {"content": "搜索 AWS 信息", "status": "in_progress"},
        {"content": "搜索 Azure 信息", "status": "pending"},
    ]


async def test_update_plan_returns_tool_message_with_correct_call_id() -> None:
    plan = [PlanStep(content="唯一步骤")]

    result = await update_plan.coroutine(plan=plan, runtime=_RUNTIME)

    messages = result.update["messages"]
    assert len(messages) == 1
    assert messages[0].tool_call_id == "call-1"
    assert "唯一步骤" in messages[0].content
    assert "[pending]" in messages[0].content


async def test_update_plan_with_empty_list_clears_plan() -> None:
    result = await update_plan.coroutine(plan=[], runtime=_RUNTIME)

    assert result.update["plan"] == []
    assert "(空计划)" in result.update["messages"][0].content


async def test_update_plan_default_status_is_pending() -> None:
    plan = [PlanStep(content="没写 status 的步骤")]

    result = await update_plan.coroutine(plan=plan, runtime=_RUNTIME)

    assert result.update["plan"][0]["status"] == "pending"


async def test_update_plan_embeds_plan_tag_for_frontend_parsing() -> None:
    """回归测试：`<plan>{json}</plan>` 标签必须存在且能被前端按
    `save_output_file` 的 `<image>` 标签同款约定解析出结构化数据——不是只有
    人类可读的 summary 文本。"""
    plan = [
        PlanStep(content="搜索 AWS 信息", status="in_progress"),
        PlanStep(content="搜索 Azure 信息", status="pending"),
    ]

    result = await update_plan.coroutine(plan=plan, runtime=_RUNTIME)

    content = result.update["messages"][0].content
    match = re.search(r"<plan>(.*)</plan>", content, re.DOTALL)
    assert match is not None, "content 里必须带 <plan>...</plan> 标签"
    parsed = json.loads(match.group(1))
    assert parsed == [
        {"content": "搜索 AWS 信息", "status": "in_progress"},
        {"content": "搜索 Azure 信息", "status": "pending"},
    ]


async def test_update_plan_tag_present_even_when_plan_is_empty() -> None:
    result = await update_plan.coroutine(plan=[], runtime=_RUNTIME)

    content = result.update["messages"][0].content
    match = re.search(r"<plan>(.*)</plan>", content, re.DOTALL)
    assert match is not None
    assert json.loads(match.group(1)) == []
