"""渐进式披露中"重"的那一半：只有技能工具被真正调用时才会用到这个类。

与原项目 `src/core/skills/content_reader.py` 的差异（对应设计文档 5.3 节
"Skill 机制 —— 保留全部设计，只改执行这一环"）：脚本执行不再是裸
`subprocess.Popen`，而是通过传入的 `Sandbox` 实例执行，获得虚拟路径隔离、
资源限制、环境变量清洗的统一处理。脚本协议本身（stdin JSON / stdout
`{"content", "__metadata__"}`）完全不变，`skills/core/search-knowledge-base/
scripts/main.py` 等现有脚本无需任何改动。
"""
from __future__ import annotations

import json
import sys

from loguru import logger

from src.agent_core.sandbox.sandbox import Sandbox
from src.agent_core.skills.skill_definition import SkillDefinition
from src.common.constants import SandboxCommandStatus

_FRONTMATTER_BOUNDARY = "---"
_INSTRUCTIONS_SECTION_TITLE = "## 技能指令"
_REFERENCES_SECTION_TITLE = "## 参考资料"
_SCRIPT_RESULT_SECTION_TITLE = "## 脚本执行结果"


class SkillContentReader:
    """负责读取 SKILL.md 正文、参考资料，以及经沙箱执行脚本并拼装最终结果。"""

    def __init__(self, skill: SkillDefinition) -> None:
        """初始化内容读取器。

        Args:
            skill: 目标技能的元数据。
        """
        self._skill = skill

    def read_instructions(self) -> str:
        """读取 SKILL.md，返回 frontmatter 之后的正文部分。

        Returns:
            SKILL.md 正文文本（已去除首尾空白）。
        """
        text = self._skill.skill_md_path.read_text(encoding="utf-8")
        parts = text.split(_FRONTMATTER_BOUNDARY, 2)
        # parts: ["", frontmatter, body] —— 取最后一段作为正文
        if len(parts) >= 3:
            return parts[2].strip()
        return text.strip()

    def read_references(self) -> list[str]:
        """遍历 references/ 目录，逐个文件读取内容。

        Returns:
            每个参考资料文件格式化为 "### 文件名\\n内容" 的字符串列表，
            references/ 目录不存在时返回空列表。
        """
        references: list[str] = []
        references_dir = self._skill.references_dir
        if not references_dir.is_dir():
            return references

        for reference_file in sorted(references_dir.iterdir()):
            if not reference_file.is_file():
                continue
            try:
                content = reference_file.read_text(encoding="utf-8")
                references.append(f"### {reference_file.name}\n{content}")
            except OSError as exc:
                logger.warning(
                    f"[SkillContentReader] 无法读取参考资料文件 "
                    f"skill={self._skill.name} file={reference_file} error={exc}"
                )
        return references

    async def run_script(
        self,
        params: dict,
        sandbox: Sandbox,
        timeout_seconds: int,
        secret_env: dict[str, str] | None = None,
    ) -> tuple[str, dict]:
        """在沙箱中执行 scripts/main.{py,js}。

        Args:
            params: 调用参数（工具入参 + runtime_context_keys 注入的运行时上下文），
                会被 JSON 序列化后写入子进程 stdin。
            sandbox: 已获取的沙箱实例，脚本将在其绑定的会话工作区内以该会话的
                cwd 执行。
            timeout_seconds: 脚本执行超时时间，对应配置项 `SKILL_SCRIPT_TIMEOUT_SECONDS`。
            secret_env: 按"三重交集"规则（技能启用 × 调用方提供 × frontmatter
                声明）算出的密钥环境变量，注入子进程时优先级最高（见设计文档
                5.4 节），可以为空。

        Returns:
            (content, metadata) 二元组：
                - content：脚本产出的正文文本，超时/失败/输出超限时为对应的错误说明文本，
                  不抛出异常，保证不中断 Agent 推理链。
                - metadata：脚本 stdout 若为 `{"content":..., "__metadata__":...}` 结构，
                  这里返回拆出的 `__metadata__`；否则为空字典。
        """
        script_path = self._skill.script_path
        if script_path is None:
            return "", {}

        interpreter = sys.executable if script_path.suffix == ".py" else "node"
        stdin_data = json.dumps(params, ensure_ascii=False).encode("utf-8")

        result = await sandbox.execute_command(
            [interpreter, str(script_path)],
            stdin=stdin_data,
            env=secret_env or None,
            timeout=timeout_seconds,
        )

        if result.status == SandboxCommandStatus.TIMEOUT:
            logger.warning(f"[SkillContentReader] 脚本超时 skill={self._skill.name} script={script_path}")
            return f"[技能脚本超时 (>{timeout_seconds}s)]", {}

        if result.status == SandboxCommandStatus.OUTPUT_TRUNCATED:
            logger.warning(f"[SkillContentReader] 脚本输出超过上限 skill={self._skill.name}")
            return "[技能脚本输出超过上限]", {}

        if result.status == SandboxCommandStatus.FAILED:
            stderr_text = result.stderr_text().strip()
            logger.warning(
                f"[SkillContentReader] 脚本返回非零退出码 skill={self._skill.name} "
                f"code={result.return_code} stderr={stderr_text}"
            )
            return f"[技能脚本执行失败: {stderr_text}]", {}

        raw = result.stdout_text().strip()
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and "__metadata__" in parsed:
                return str(parsed.get("content", "")), parsed["__metadata__"]
        except (json.JSONDecodeError, ValueError):
            pass

        return raw, {}

    async def assemble(
        self,
        params: dict,
        sandbox: Sandbox | None,
        script_timeout_seconds: int,
        secret_env: dict[str, str] | None = None,
    ) -> str:
        """按"指令 + 参考资料 + 脚本结果"的顺序拼装最终结果。

        Args:
            params: 传给脚本的调用参数（无脚本技能不使用）。
            sandbox: 已获取的沙箱实例；无脚本技能可以传 None。
            script_timeout_seconds: 脚本执行超时时间。
            secret_env: 按需注入脚本子进程的密钥环境变量，见 `run_script`。

        Returns:
            拼装好的最终文本，作为工具返回值交给 LLM。
        """
        result_parts: list[str] = []

        instructions = self.read_instructions()
        if instructions:
            result_parts.append(f"{_INSTRUCTIONS_SECTION_TITLE}\n{instructions}")

        references = self.read_references()
        if references:
            result_parts.append(f"{_REFERENCES_SECTION_TITLE}\n" + "\n\n".join(references))

        if self._skill.has_script():
            if sandbox is None:
                logger.warning(
                    f"[SkillContentReader] 技能 {self._skill.tool_name} 需要沙箱执行，"
                    f"但未提供 sandbox 实例，跳过脚本执行"
                )
                result_parts.append(f"{_SCRIPT_RESULT_SECTION_TITLE}\n[缺少沙箱环境，脚本未执行]")
            else:
                try:
                    script_output, _metadata = await self.run_script(
                        params, sandbox, script_timeout_seconds, secret_env=secret_env
                    )
                except Exception as exc:
                    logger.error(f"[SkillContentReader] 脚本执行异常 skill={self._skill.name} error={exc}")
                    script_output = f"[脚本执行错误: {exc}]"
                if script_output:
                    result_parts.append(f"{_SCRIPT_RESULT_SECTION_TITLE}\n{script_output}")

        return "\n\n".join(result_parts)
