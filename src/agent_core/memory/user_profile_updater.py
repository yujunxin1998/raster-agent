"""维护用户长期画像与时间线（三层记忆架构的 L1/L2）。

跟 `MemoryExtractor`（L3 Facts，逐条提取离散记录）是两套独立的 LLM 调用：
这里每轮对话结束后把"现有画像/时间线 + 这一轮对话"整体喂给模型，产出一份
重写后的完整画像/时间线（不是增量 diff），是否继承旧内容由模型自己判断
（比如 `top_of_mind` 这种高频变化的字段，模型会自然地把已经不再相关的
焦点替换掉）。

调用失败（LLM 异常/JSON 解析失败）按项目里其它记忆相关组件一致的降级
风格处理：只记 warning 后跳过，不写入半截数据，不影响主对话流程。
"""
from __future__ import annotations

import json
from typing import Optional

from langchain_core.messages import HumanMessage
from loguru import logger

from src.agent_core.model import create_chat_model
from src.storage.user_profile_store import UserProfileStore

_UPDATE_PROMPT_TEMPLATE = """你在维护一个用户的长期画像。下面是这个用户目前的
画像和时间线（如果是全新用户，各字段为空），以及最新一轮对话。请输出更新后的
完整画像和时间线（不是增量，是重写后的完整内容——旧内容需要保留的部分你自己
决定要不要继承，不需要的自然淘汰）。

【现有画像】
职业背景（work_context）：{work_context}
个人背景（personal_context）：{personal_context}
当前关注（top_of_mind）：{top_of_mind}

【现有时间线】
近 1-3 个月（recent_months）：{recent_months}
3-12 个月前（earlier_context）：{earlier_context}
长期背景（long_term_background）：{long_term_background}

【本轮对话】
用户：{user_message}
AI：{ai_response}

【字段说明】
- work_context: 职业角色、公司、关键项目、主力技术栈，2-3 句话
- personal_context: 语言能力、沟通偏好、兴趣领域，1-2 句话
- top_of_mind: 当前关注的多个并行焦点，3-5 句话，这是更新频率最高的字段
- recent_months: 近 1-3 个月的详细活动摘要，4-6 句话
- earlier_context: 3-12 个月前的重要模式，3-5 句话
- long_term_background: 长期不变的基础背景，2-4 句话

如果本轮对话里没有任何值得更新画像/时间线的信息，原样返回现有内容即可，
不要凭空编造。只输出 JSON，不要任何其他文字：
{{"work_context": "...", "personal_context": "...", "top_of_mind": "...",
  "recent_months": "...", "earlier_context": "...", "long_term_background": "..."}}"""

_EMPTY_PROFILE = {
    "work_context": "",
    "personal_context": "",
    "top_of_mind": "",
    "recent_months": "",
    "earlier_context": "",
    "long_term_background": "",
}

_PROFILE_FIELDS = tuple(_EMPTY_PROFILE.keys())


class UserProfileUpdater:
    """每轮对话结束后，把画像/时间线跟这一轮对话合并重写并存回。"""

    def __init__(self, model_name: str, provider: str, api_key: str, base_url: str) -> None:
        """初始化更新器。

        Args:
            model_name: 更新用的模型名称，为空时由调用方回退到主模型。
            provider: 模型供应商标识。
            api_key: 模型服务 API Key。
            base_url: 模型服务 Base URL。
        """
        self._model_name = model_name
        self._provider = provider
        self._api_key = api_key
        self._base_url = base_url

    async def update_after_chat(
        self,
        store: UserProfileStore,
        user_message: str,
        ai_response: str,
        user_id: str,
    ) -> None:
        """合并现有画像/时间线与本轮对话，重写后存回。

        Args:
            store: 目标画像存储。
            user_message: 用户本轮发言。
            ai_response: AI 本轮回复。
            user_id: 归属用户 ID。
        """
        existing = await store.get(user_id) or _EMPTY_PROFILE
        prompt = _UPDATE_PROMPT_TEMPLATE.format(
            work_context=existing["work_context"],
            personal_context=existing["personal_context"],
            top_of_mind=existing["top_of_mind"],
            recent_months=existing["recent_months"],
            earlier_context=existing["earlier_context"],
            long_term_background=existing["long_term_background"],
            user_message=user_message,
            ai_response=ai_response,
        )

        parsed = await self._call_llm_update(prompt)
        if parsed is None:
            return

        await store.upsert(
            user_id,
            work_context=str(parsed.get("work_context", "")),
            personal_context=str(parsed.get("personal_context", "")),
            top_of_mind=str(parsed.get("top_of_mind", "")),
            recent_months=str(parsed.get("recent_months", "")),
            earlier_context=str(parsed.get("earlier_context", "")),
            long_term_background=str(parsed.get("long_term_background", "")),
        )

    async def _call_llm_update(self, prompt: str) -> Optional[dict]:
        try:
            llm = create_chat_model(
                model=self._model_name, provider=self._provider,
                api_key=self._api_key, base_url=self._base_url,
            )
            result = await llm.ainvoke([HumanMessage(content=prompt)])
            raw = result.content.strip()

            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]

            data = json.loads(raw)
        except Exception as exc:
            logger.warning(f"[UserProfileUpdater] 画像更新失败，已跳过: {exc}")
            return None

        if not isinstance(data, dict):
            logger.warning(f"[UserProfileUpdater] 模型输出不是 JSON 对象，已跳过: {raw[:200]!r}")
            return None
        return {field: data.get(field, "") for field in _PROFILE_FIELDS}
