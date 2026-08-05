"""通用基础设施：异常体系、统一响应封装、枚举常量。

其余所有模块（agent_core / storage / api）都允许依赖 common，
但 common 不允许反向依赖任何业务模块，保持依赖方向单向。
"""
