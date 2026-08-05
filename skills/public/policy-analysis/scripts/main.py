"""Policy analysis skill script.

Reads JSON parameters from stdin, performs any preprocessing or
data enrichment, then writes a plain-text result to stdout.

Interface contract:
  stdin  – JSON object matching the skill's parameter schema
  stdout – plain text to be included in the LLM context as "脚本执行结果"
  stderr – diagnostic output (not shown to the LLM)
  exit 0 – success; any other code is treated as an error
"""

import json
import sys


def main() -> None:
    raw = sys.stdin.read()
    try:
        params = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        print(f"参数解析错误: {exc}", file=sys.stderr)
        sys.exit(1)

    query: str = params.get("query", "")
    context: str = params.get("context", "")

    # ------------------------------------------------------------------
    # Add custom logic here, e.g. query a local policy database,
    # call an internal API, or enrich the query with metadata.
    # ------------------------------------------------------------------

    lines = []
    if context:
        lines.append(f"背景信息：{context}")
    lines.append(f"用户查询：{query}")
    lines.append("（此脚本可扩展为调用内部政策数据库或 API）")

    print("\n".join(lines))


if __name__ == "__main__":
    main()
