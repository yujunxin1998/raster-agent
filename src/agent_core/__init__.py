"""Agent Core（Harness 层）。

对应设计文档《新一代AgentLoop工程设计方案.md》中 harness/app 分层里的 harness 部分：
只包含与"跑在 FastAPI 里"这件事无关的核心能力——技能(skills)、记忆(memory)、
工具(tools)、权限控制(guardrail)、沙箱(sandbox)、虚拟工作区(workspace)。

依赖方向：agent_core 可以依赖 common / storage / config，不允许依赖 api / schema，
保证这一层未来具备独立测试、甚至被其它入口（而不仅是 FastAPI）复用的可能性。
"""
