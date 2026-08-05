"""`EvalEventEmitter` 单元测试：mock Redis 客户端，不发起真实网络请求。"""
from __future__ import annotations

from unittest.mock import AsyncMock, patch

from src.agent_core.eval.emitter import EvalEventEmitter


def _make_emitter() -> EvalEventEmitter:
    return EvalEventEmitter(trace_id="t1", conversation_id="c1", user_id="u1", user_input="你好")


async def test_tool_start_end_pairing_writes_eval_tool() -> None:
    fake_redis = AsyncMock()
    emitter = _make_emitter()

    with patch("src.agent_core.eval.emitter._get_redis", return_value=fake_redis):
        emitter.on_tool_start_sync("web_search", "run-1", {"query": "天气"})
        await emitter.on_tool_end_sync("run-1", "晴天", status="success")

    fake_redis.xadd.assert_awaited_once()
    stream_name, fields = fake_redis.xadd.await_args.args
    assert stream_name == "eval:tool"
    assert fields["tool_name"] == "web_search"
    assert fields["tool_result"] == "晴天"
    assert fields["status"] == "success"
    assert fields["trace_id"] == "t1"


async def test_tool_end_without_matching_start_is_noop() -> None:
    fake_redis = AsyncMock()
    emitter = _make_emitter()

    with patch("src.agent_core.eval.emitter._get_redis", return_value=fake_redis):
        await emitter.on_tool_end_sync("unknown-run-id", "result")

    fake_redis.xadd.assert_not_awaited()


async def test_flush_trace_writes_eval_trace_with_token_usage() -> None:
    fake_redis = AsyncMock()
    emitter = _make_emitter()
    emitter.on_token_usage(100, 50)
    emitter.on_token_usage(150, 80)  # max-delta：应更新为 150/80，不是累加成 250/130

    with patch("src.agent_core.eval.emitter._get_redis", return_value=fake_redis):
        await emitter.flush_trace("最终回复", task_status="success")

    fake_redis.xadd.assert_awaited_once()
    stream_name, fields = fake_redis.xadd.await_args.args
    assert stream_name == "eval:trace"
    assert fields["final_answer"] == "最终回复"
    assert fields["token_input"] == "150"
    assert fields["token_output"] == "80"
    assert fields["task_status"] == "success"


async def test_redis_unavailable_all_calls_are_noop() -> None:
    emitter = _make_emitter()

    with patch("src.agent_core.eval.emitter._get_redis", return_value=None):
        emitter.on_tool_start_sync("web_search", "run-1", {})
        await emitter.on_tool_end_sync("run-1", "result")  # 不应抛异常
        await emitter.flush_trace("答案")  # 不应抛异常
