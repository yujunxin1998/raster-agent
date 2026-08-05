"""LLM 工厂：统一的 Chat Model 构建入口。

参考 DeerFlow（`bytedance/deer-flow`）`packages/harness/deerflow/models/` 的封装
思路重新设计，取代原项目 `LLMFactory` 里"按 (purpose, kwargs) 缓存单例 + 无条件
剥离 tool_choice"的写法：本模块只提供一个无状态的工厂函数
`create_chat_model()`，每次调用现造一个模型实例，供应商特有的修复收敛到
`PatchedChatDeepSeek` 一个类里。
"""
from __future__ import annotations

from src.agent_core.model.model_factory import create_chat_model

__all__ = ["create_chat_model"]
