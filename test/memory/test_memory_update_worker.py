"""`MemoryUpdateWorker` 单元测试：mock 各 Store + Apply Engine + LLM 调用。"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import AIMessage

from src.agent_core.memory.memory_apply_engine import ProfilePatchConflictError
from src.agent_core.memory.memory_delta import FactOperation, MemoryDelta
from src.agent_core.memory.memory_update_worker import MemoryUpdateWorker

_JOB = {
    "job_id": "job-1", "user_id": "user-1", "conversation_id": "conv-1",
    "agent_name": None, "first_event_id": "evt-1", "last_event_id": "evt-1", "trace_id": None,
}


def _worker(**overrides):
    event_store = overrides.get("event_store") or MagicMock()
    job_store = overrides.get("job_store") or MagicMock()
    profile_store = overrides.get("profile_store") or MagicMock()
    fact_store = overrides.get("fact_store") or MagicMock()
    apply_engine = overrides.get("apply_engine") or MagicMock()
    audit_store = overrides.get("audit_store", MagicMock())

    profile_store.get_with_revision = AsyncMock(return_value={"revision": 0, **{
        f: "" for f in ("work_context", "personal_context", "top_of_mind",
                        "recent_months", "earlier_context", "long_term_background")
    }})
    fact_store.find_candidates_for_update = AsyncMock(return_value=[])
    job_store.mark_succeeded = AsyncMock()
    job_store.mark_retry = AsyncMock()
    if audit_store is not None:
        audit_store.record = AsyncMock()
    apply_engine.apply = AsyncMock()

    worker = MemoryUpdateWorker(
        event_store, job_store, profile_store, fact_store, apply_engine, audit_store,
        model_name="m", provider="p", api_key="k", base_url="b",
        debounce_seconds=30, lease_seconds=120, poll_interval_seconds=5,
        max_attempts=5, retry_backoff_seconds=[30, 120],
        candidate_facts_limit=15, max_conversation_chars=12000,
        discard_confidence_threshold=0.70,
        max_profile_patches=5, max_fact_operations=10, max_text_length=2000,
        sensitive_filter_enabled=True,
    )
    return worker, event_store, job_store, profile_store, fact_store, apply_engine, audit_store


def _conversation():
    return [
        {"role": "user", "content": "我计划学习 Rust", "id": "m1"},
        {"role": "assistant", "content": "好的，记下了", "id": "m2"},
    ]


def _llm_response(content: str):
    fake_llm = MagicMock()
    fake_llm.ainvoke = AsyncMock(return_value=AIMessage(content=content))
    return fake_llm


async def test_process_with_empty_conversation_marks_succeeded_without_llm_call() -> None:
    worker, event_store, job_store, *_ = _worker()
    event_store.get_conversation_for_job = AsyncMock(return_value=[])

    with patch("src.agent_core.memory.memory_update_worker.create_chat_model") as mock_create:
        await worker._process(_JOB)

    mock_create.assert_not_called()
    job_store.mark_succeeded.assert_awaited_once_with("job-1")


async def test_llm_failure_marks_retry() -> None:
    worker, event_store, job_store, *_ = _worker()
    event_store.get_conversation_for_job = AsyncMock(return_value=_conversation())

    with patch("src.agent_core.memory.memory_update_worker.create_chat_model", side_effect=Exception("boom")):
        await worker._process(_JOB)

    job_store.mark_retry.assert_awaited_once()
    assert "LLM 调用失败" in job_store.mark_retry.await_args.args[1]


async def test_invalid_json_marks_retry() -> None:
    worker, event_store, job_store, *_ = _worker()
    event_store.get_conversation_for_job = AsyncMock(return_value=_conversation())

    with patch("src.agent_core.memory.memory_update_worker.create_chat_model",
               return_value=_llm_response("not json at all")):
        await worker._process(_JOB)

    job_store.mark_retry.assert_awaited_once()
    assert "解析失败" in job_store.mark_retry.await_args.args[1]


async def test_delta_with_evidence_outside_conversation_is_rejected_and_retried() -> None:
    worker, event_store, job_store, _, _, apply_engine, audit_store = _worker()
    event_store.get_conversation_for_job = AsyncMock(return_value=_conversation())
    payload = json.dumps({
        "schema_version": 1, "source_event_ids": ["evt-1"],
        "profile_patches": [{
            "field": "top_of_mind", "op": "set", "value": "x",
            "confidence": 0.9, "evidence_message_ids": ["m-not-real"],
        }],
        "fact_operations": [],
    })

    with patch("src.agent_core.memory.memory_update_worker.create_chat_model", return_value=_llm_response(payload)):
        await worker._process(_JOB)

    apply_engine.apply.assert_not_awaited()
    job_store.mark_retry.assert_awaited_once()
    audit_store.record.assert_awaited_once()
    assert audit_store.record.await_args.kwargs["action"].value == "delta_rejected"


async def test_valid_delta_applies_and_marks_succeeded() -> None:
    worker, event_store, job_store, _, _, apply_engine, audit_store = _worker()
    event_store.get_conversation_for_job = AsyncMock(return_value=_conversation())
    payload = json.dumps({
        "schema_version": 1, "source_event_ids": ["evt-1"],
        "profile_patches": [],
        "fact_operations": [{
            "op": "add", "content": "用户计划学习 Rust", "category": "goal",
            "importance": 7, "confidence": 0.9, "evidence_message_ids": ["m1"],
        }],
    })

    with patch("src.agent_core.memory.memory_update_worker.create_chat_model", return_value=_llm_response(payload)):
        await worker._process(_JOB)

    apply_engine.apply.assert_awaited_once()
    applied_delta = apply_engine.apply.await_args.args[0]
    assert len(applied_delta.fact_operations) == 1
    job_store.mark_succeeded.assert_awaited_once_with("job-1")


async def test_low_confidence_add_is_dropped_before_apply() -> None:
    worker, event_store, job_store, _, _, apply_engine, _ = _worker()
    event_store.get_conversation_for_job = AsyncMock(return_value=_conversation())
    payload = json.dumps({
        "schema_version": 1, "source_event_ids": ["evt-1"],
        "profile_patches": [],
        "fact_operations": [{
            "op": "add", "content": "用户可能对 Go 感兴趣", "category": "preference",
            "importance": 4, "confidence": 0.5, "evidence_message_ids": ["m1"],
        }],
    })

    with patch("src.agent_core.memory.memory_update_worker.create_chat_model", return_value=_llm_response(payload)):
        await worker._process(_JOB)

    applied_delta = apply_engine.apply.await_args.args[0]
    assert applied_delta.fact_operations == []
    job_store.mark_succeeded.assert_awaited_once()


async def test_profile_patch_conflict_marks_retry_instead_of_succeeded() -> None:
    worker, event_store, job_store, _, _, apply_engine, _ = _worker()
    event_store.get_conversation_for_job = AsyncMock(return_value=_conversation())
    apply_engine.apply = AsyncMock(side_effect=ProfilePatchConflictError("冲突"))
    payload = json.dumps({
        "schema_version": 1, "source_event_ids": ["evt-1"],
        "profile_patches": [{
            "field": "top_of_mind", "op": "set", "value": "重构记忆模块",
            "confidence": 0.9, "evidence_message_ids": ["m1"],
        }],
        "fact_operations": [],
    })

    with patch("src.agent_core.memory.memory_update_worker.create_chat_model", return_value=_llm_response(payload)):
        await worker._process(_JOB)

    job_store.mark_retry.assert_awaited_once()
    job_store.mark_succeeded.assert_not_awaited()


async def test_process_job_safely_catches_unexpected_exception() -> None:
    worker, event_store, job_store, *_ = _worker()
    event_store.get_conversation_for_job = AsyncMock(side_effect=RuntimeError("db down"))

    await worker._process_job_safely(_JOB)

    job_store.mark_retry.assert_awaited_once()


def test_truncate_conversation_keeps_tail_within_budget() -> None:
    worker, *_ = _worker()
    conversation = [
        {"role": "user", "content": "a" * 100, "id": "m1"},
        {"role": "assistant", "content": "b" * 100, "id": "m2"},
        {"role": "user", "content": "c" * 100, "id": "m3"},
    ]
    worker = worker  # 复用同一个 worker 实例（max_conversation_chars=12000 太宽松，改小测试）
    worker._max_conversation_chars = 150

    result = worker._truncate_conversation(conversation)

    assert [m["id"] for m in result] == ["m3"]


def test_drop_low_confidence_adds_keeps_non_add_operations() -> None:
    worker, *_ = _worker()
    delta = MemoryDelta(fact_operations=[
        FactOperation(op="add", content="低置信度", category="context", confidence=0.5, evidence_message_ids=["m1"]),
        FactOperation(op="reinforce", target_fact_id="f-1"),
    ])

    result = worker._drop_low_confidence_adds(delta)

    assert len(result.fact_operations) == 1
    assert result.fact_operations[0].op.value == "reinforce"
