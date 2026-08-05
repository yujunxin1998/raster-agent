"""`chat_pipeline.py` 里 `<ref_json>` 状态机 + 来源提取的单元测试。"""
from __future__ import annotations

from src.agent_core.agents.chat_pipeline import (
    _build_final_sources,
    _extract_knowledge_base_sources,
    _extract_web_sources,
    _push_ref_text,
)


def _new_ref_state() -> dict:
    return {"pending": "", "in_ref": False, "buf": "", "sources": None}


def test_ref_json_fully_in_one_chunk_is_stripped_and_parsed() -> None:
    state = _new_ref_state()
    text = '这是回答内容<ref_json>{"sources":[{"index":1,"name":"a.pdf"}]}</ref_json>'

    emitted = _push_ref_text(text, state)

    assert emitted == ["这是回答内容"]
    assert state["sources"] == [{"index": 1, "name": "a.pdf"}]
    assert state["in_ref"] is False
    assert state["buf"] == ""


def test_ref_json_split_across_chunks() -> None:
    state = _new_ref_state()

    emitted_1 = _push_ref_text("正文<ref_json>", state)
    assert emitted_1 == ["正文"]
    assert state["in_ref"] is True

    emitted_2 = _push_ref_text('{"sources":[{"index":1,"name":"b.pdf"}]}</ref_json>', state)
    assert emitted_2 == []  # ref_json 内部内容不对外输出
    assert state["sources"] == [{"index": 1, "name": "b.pdf"}]
    assert state["in_ref"] is False


def test_unclosed_ref_json_stays_buffered_not_emitted() -> None:
    state = _new_ref_state()
    text = '正文<ref_json>{"sources":[]}'  # 缺少 </ref_json>

    emitted = _push_ref_text(text, state)

    assert emitted == ["正文"]
    assert state["in_ref"] is True
    assert state["buf"] == '{"sources":[]}'
    assert state["sources"] is None  # 尚未解析（等待闭合标签或调用方兜底 flush）


def test_plain_text_without_sentinel_flushes_with_safety_tail() -> None:
    """没有 <ref_json> 哨兵时，状态机会保留末尾 len('<ref_json>') 个字符作为安全缓冲，
    防止哨兵被拆到下一个 chunk 里而漏检——这是原项目状态机的既有行为，不是 bug。"""
    state = _new_ref_state()

    emitted = _push_ref_text("hello world", state)  # 11 字符，> 10 字符的安全缓冲长度

    assert emitted == ["h"]
    assert state["pending"] == "ello world"


def test_extract_knowledge_base_sources_parses_chunks() -> None:
    text = (
        '<知识片段 [1] id=doc1 title="报告.pdf" score=0.92>\n这里是内容\n</知识片段>\n'
        '<知识片段 [2] id=doc2 title="手册.pdf" score=0.81>\n另一段内容\n</知识片段>'
    )

    sources = _extract_knowledge_base_sources(text)

    assert [s["id"] for s in sources] == ["doc1", "doc2"]
    assert sources[0]["name"] == "报告.pdf"
    assert sources[0]["type"] == "file"


def test_extract_web_sources_parses_numbered_list() -> None:
    text = "【来源】\n1. 标题A\n   链接：http://a.com\n   摘要：内容A\n2. 标题B\n   链接：http://b.com\n   摘要：内容B"

    sources = _extract_web_sources(text)

    assert [s["name"] for s in sources] == ["标题A", "标题B"]
    assert sources[0]["url"] == "http://a.com"
    assert sources[0]["type"] == "web"


def test_build_final_sources_prefers_model_selection_backfilled_from_fallback() -> None:
    ref_state = {"sources": [{"index": 1, "name": "报告.pdf"}]}
    fallback_sources = [
        {"id": "doc1", "type": "file", "name": "报告.pdf", "desc": "摘要", "url": ""},
        {"id": "doc2", "type": "file", "name": "未引用的.pdf", "desc": "摘要2", "url": ""},
    ]

    final_sources = _build_final_sources(ref_state, fallback_sources)

    assert len(final_sources) == 1  # 模型只选了一个，未引用的那条不应该出现
    assert final_sources[0]["name"] == "报告.pdf"
    assert final_sources[0]["index"] == 1
    assert final_sources[0]["desc"] == "摘要"  # 完整字段值来自 fallback，不依赖模型 echo


def test_build_final_sources_falls_back_when_model_omits_ref_json() -> None:
    ref_state = {"sources": None}
    fallback_sources = [{"id": "doc1", "type": "file", "name": "报告.pdf", "desc": "摘要", "url": ""}]

    final_sources = _build_final_sources(ref_state, fallback_sources)

    assert final_sources == fallback_sources
