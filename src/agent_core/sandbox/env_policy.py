"""沙箱子进程环境变量清洗策略。

对应设计文档 5.3 节：沙箱执行前不再把服务端进程的完整 `os.environ` 传给子进程，
过滤掉看起来像密钥的变量名，需要的密钥由技能 frontmatter 的 `required_secrets`
字段显式声明、按请求注入（该按需注入能力属于 guardrail + skill 联动的后续增量，
本模块只负责"默认清洗"这一层兜底防线）。

另外强制注入 `PYTHONIOENCODING=utf-8`：Windows 上 Python 子进程的
`sys.stdout`/`sys.stderr` 默认编码取自系统区域码页（中文 Windows 通常是
GBK/cp936），而 `Sandbox.execute_command()` 的调用方（`sandbox_tool.py`/
`skill_content_reader.py`）统一按 UTF-8 解码 stdout/stderr——两边编码不一致时，
子进程脚本里 `print()` 输出的中文字符会被错误解码成乱码（用
`bytes.decode("utf-8", errors="replace")` 解码 GBK 字节得到的替换字符），
这与"Sandbox 实现有 bug"无关，是纯粹的编码不匹配，跟 POSIX 下默认就是 UTF-8
locale 因而从未暴露出来的情况不同。强制设置后子进程无论运行在哪个操作系统/
区域设置下，`print()` 输出都固定按 UTF-8 编码，和这一层的解码方式对齐。
"""
from __future__ import annotations

import os
import re

# 变量名中出现以下任一关键字（不区分大小写）即视为"看起来像密钥"，默认从
# 子进程环境变量中剔除。命中规则的常见例子：OPENAI_API_KEY、DATABASE_URL 的
# 连接串本身虽不直接匹配，但 *_URL 类不在此列，交由下面的显式连接串黑名单处理。
_SECRET_NAME_PATTERN = re.compile(r"(KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL)", re.IGNORECASE)

# 显式连接串类变量名黑名单：变量名本身不含上面的关键字，但值通常是带凭证的连接串。
_CONNECTION_STRING_DENYLIST = frozenset({"DATABASE_URL", "REDIS_URL", "ES_URL"})

# 良性变量白名单：即使命中上面的模式也应当保留（目前没有实际冲突场景，预留位）。
_BENIGN_NAMES = frozenset({"PATH", "HOME", "LANG", "VIRTUAL_ENV", "TMPDIR", "TMP", "TEMP"})


def build_sandbox_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    """构造沙箱子进程使用的环境变量字典。

    做法：从当前进程的 `os.environ` 拷贝一份，剔除命中密钥模式/连接串黑名单
    的变量，再叠加调用方显式传入的 `extra_env`（调用方传入的值优先级最高，
    即使同名变量在清洗阶段被剔除，显式传入的值依然会生效）。

    Args:
        extra_env: 调用方希望额外注入给子进程的环境变量（如技能声明的
            `required_secrets` 按请求注入的值），可以为空。

    Returns:
        清洗后的环境变量字典，可直接传给 subprocess 的 env 参数。
    """
    cleaned: dict[str, str] = {}
    for name, value in os.environ.items():
        if name in _BENIGN_NAMES:
            cleaned[name] = value
            continue
        if name in _CONNECTION_STRING_DENYLIST:
            continue
        if _SECRET_NAME_PATTERN.search(name):
            continue
        cleaned[name] = value

    if extra_env:
        cleaned.update(extra_env)

    # 强制覆盖，不允许被 extra_env 意外带偏——这是编码正确性问题，不是可配置项。
    cleaned["PYTHONIOENCODING"] = "utf-8"
    return cleaned
