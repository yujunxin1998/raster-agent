"""持久化层（对应 Alibaba 分层架构里的 DAO 层）。

每个 Store 类只负责一张表的读写，不掺杂业务判断（业务判断放在
agent_core 对应的 Manager/Provider 类里）。所有 Store 均以"构造函数注入
asyncpg.Pool + 模块级单例访问器"的方式组织，风格与 agent_core 下的
SkillManager/MemoryManager 保持一致。
"""
