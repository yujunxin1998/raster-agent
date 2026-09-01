"""subagent 通讯协议测试。"""
from src.agent_core.agents.delegation_protocol import (
    SubagentResult,
    build_delegation_task_id,
    derive_subagent_context,
)
from src.agent_core.middlewares.context import AgentRuntimeContext


def test_task_id_is_stable_and_payload_sensitive() -> None:
    first = build_delegation_task_id(
        conversation_id="c1", subagent_type="web-researcher", task="天气",
    )
    same = build_delegation_task_id(
        conversation_id="c1", subagent_type="web-researcher", task="天气",
    )
    different = build_delegation_task_id(
        conversation_id="c1", subagent_type="web-researcher", task="新闻",
    )

    assert first == same
    assert first != different


def test_child_context_has_delegation_lineage_and_fresh_cache() -> None:
    parent = AgentRuntimeContext(
        conversation_id="c1", user_id="u1", task_id="parent", root_task_id="root",
        delegation_depth=1, memory_cache={"facts": [1]},
    )

    child = derive_subagent_context(parent, task_id="child", agent_name="web-researcher")

    assert child.task_id == "child"
    assert child.parent_task_id == "parent"
    assert child.root_task_id == "root"
    assert child.delegation_depth == 2
    assert child.agent_name == "web-researcher"
    assert child.memory_cache == {}
    assert parent.memory_cache == {"facts": [1]}


def test_structured_result_preserves_error_semantics_for_text_compatibility() -> None:
    result = SubagentResult(
        task_id="t1", agent_name="worker", status="timed_out",
        error_code="subagent_timeout", error_message="执行超时", retryable=True,
    )

    assert result.status == "timed_out"
    assert result.retryable is True
    assert result.to_text() == "[worker] 执行失败: 执行超时"
