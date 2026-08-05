"""Controller 层：REST 路由，只做参数校验、调用 Service/Manager、包装响应。

不允许在这一层写业务逻辑（业务逻辑属于 agent_core 下的 Manager/Provider），
Controller 只负责把 HTTP 请求翻译成对下层的调用。
"""
