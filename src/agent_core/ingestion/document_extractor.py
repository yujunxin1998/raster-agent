"""把常见办公文档格式提取为 Markdown，供沙箱工具 `read_file` 消费。

背景：`read_file`（见 `agent_core/tools/sandbox_tool.py`）内部是
`Path.read_text(encoding="utf-8")`，只认纯文本。.docx/.xlsx/.pdf 是结构化
二进制格式，直接 read_text 要么乱码要么报编码错误。以前的做法是把原始
二进制文件复制进 workspace/ 后甩给模型一句"可以用 read_file 读"，模型撞到
乱码/异常后，只能现场尝试 `pip install python-docx` 之类的库自己解析——
但沙箱没有外网访问权限（详见 sandbox_tool.py 模块文档"安全边界"一节），
这条路在这个环境里总是走不通。

解决方式是不指望模型运行时解析：本模块在附件进入模型视野之前，用预装的
纯 Python 库（不需要系统级二进制、不需要联网）离线把可识别格式转成
Markdown，连同原始文件一起放进 workspace/，模型只需要对准转换后的 .md
调用 `read_file` 即可，不用再自己写解析代码。参考 bytedance/deer-flow
`uploads.py` 里 `convert_file_to_markdown` 的做法——统一转 Markdown 而不是
纯文本，是因为 Markdown 能保留标题层级/表格结构，模型读起来比拍平的纯文本
更容易定位到具体章节或单元格；`.pdf` 是例外，pypdf 只能拿到文字流，没有
语义层级信息，硬套 `#`/表格语法只是伪结构，不如老实输出按页分隔的纯文本
（仍然是合法的 Markdown，只是不含额外标记）。

支持的格式是"能被纯 Python 库稳定解析、不依赖外部程序"的交集：
    - .docx  python-docx（标题样式映射为 # 层级，正文段落原样输出，
             表格转 Markdown 表格语法）
    - .xlsx  openpyxl（每个 sheet 转一张 Markdown 表格）
    - .pdf   pypdf（逐页文字层，按页分隔；扫描件没有文字层，提取到空文本时
             如实告知，不是解析失败，不应该假装成功）
    - .doc   （旧版二进制 OLE 格式）明确不支持——没有轻量纯 Python 库能解析，
             需要 LibreOffice 等外部程序，这个环境里不具备；直接返回清晰的
             不支持提示，好过静默失败或返回乱码。
.txt/.md/.csv/.json 本来就是纯文本，不需要经过本模块，`read_file` 直接读
原文件即可（见 `EXTRACTABLE_EXTENSIONS` 只收录需要转换的格式）。

是否在附件进入对话时自动触发这层转换，由 `settings.ATTACHMENT_AUTO_CONVERT`
开关控制（默认开启），命名和语义对齐 deer-flow 的
`uploads.auto_convert_documents` 配置项——调用方（chat_pipeline.py）负责
读取这个开关，本模块只管"给了就转"，不关心开关状态。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

# 需要经过本模块转换才能被 read_file 读出可读文本的扩展名。
# .txt/.md/.csv/.json 本身已是纯文本，不在此列，调用方应直接用原文件。
EXTRACTABLE_EXTENSIONS = {".docx", ".xlsx", ".pdf"}

# 白名单里允许上传、但本模块明确不解析的格式：给出的是"为什么不行"，
# 不是空实现，调用方据此向用户/模型给出可操作的下一步建议。
_UNSUPPORTED_REASONS = {
    ".doc": "旧版 Word 二进制格式（.doc），无轻量纯 Python 库可解析，建议另存为 .docx 后重新上传",
    ".xls": "旧版 Excel 二进制格式（.xls），无轻量纯 Python 库可解析，建议另存为 .xlsx 后重新上传",
}


@dataclass(frozen=True)
class ExtractionResult:
    """一次文本提取的结果。

    Attributes:
        text: 提取出的 Markdown 文本（.pdf 例外，是不含额外标记的纯文本，
            仍是合法 Markdown）；提取失败时为空字符串。
        ok: 是否成功提取到非空文本。
        message: ok=False 时的原因说明（面向模型/用户，可直接展示）。
    """

    text: str
    ok: bool
    message: str = ""


def is_extractable(extension: str) -> bool:
    """判断该扩展名是否需要（且能够）经本模块转换为 Markdown。"""
    return extension.lower() in EXTRACTABLE_EXTENSIONS


def is_known_unsupported(extension: str) -> str | None:
    """扩展名是否属于"已知无法解析"的格式，是则返回原因说明，否则返回 None。"""
    return _UNSUPPORTED_REASONS.get(extension.lower())


def extract_text(path: Path, extension: str) -> ExtractionResult:
    """按扩展名分派到对应解析器，提取 Markdown 文本。

    Args:
        path: 源文件的绝对路径（原始二进制文件，不是转换后的 .txt）。
        extension: 小写扩展名（含前导点），如 ".docx"。

    Returns:
        提取结果；解析器抛出的异常在这里统一兜住，转成 `ok=False` 的结果，
        不向上抛出——附件解析失败不应该中断整轮对话。
    """
    handler = _HANDLERS.get(extension.lower())
    if handler is None:
        return ExtractionResult(text="", ok=False, message=f"不支持的格式：{extension}")

    try:
        text = handler(path)
    except Exception as exc:  # noqa: BLE001 — 第三方解析库的异常类型不可枚举，统一兜底
        logger.warning(f"[document_extractor] 解析失败 path={path} extension={extension} error={exc}")
        return ExtractionResult(text="", ok=False, message=f"解析失败：{exc}")

    if not text.strip():
        return ExtractionResult(
            text="", ok=False,
            message="未提取到文字内容（PDF 常见原因是扫描件/图片版，没有可提取的文字层）",
        )
    return ExtractionResult(text=text, ok=True)


_HEADING_STYLE_PATTERN = re.compile(r"^Heading (\d+)$")


def _extract_docx(path: Path) -> str:
    from docx import Document  # 惰性 import：非文档场景不需要这个依赖常驻内存

    document = Document(str(path))
    parts = []
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()
        if not text:
            continue
        heading_match = _HEADING_STYLE_PATTERN.match(paragraph.style.name or "")
        if heading_match:
            level = min(int(heading_match.group(1)), 6)
            parts.append(f"{'#' * level} {text}")
        else:
            parts.append(text)

    for table in document.tables:
        rows = [
            [cell.text.strip() for cell in row.cells]
            for row in table.rows
        ]
        rows = [row for row in rows if any(row)]
        if not rows:
            continue
        parts.append(_rows_to_markdown_table(rows))

    return "\n\n".join(parts)


def _extract_xlsx(path: Path) -> str:
    from openpyxl import load_workbook

    workbook = load_workbook(str(path), read_only=True, data_only=True)
    try:
        sheets_text = []
        for sheet in workbook.worksheets:
            rows = [
                ["" if cell is None else str(cell) for cell in row]
                for row in sheet.iter_rows(values_only=True)
                if any(cell is not None for cell in row)
            ]
            if not rows:
                continue
            sheets_text.append(f"## Sheet: {sheet.title}\n\n{_rows_to_markdown_table(rows)}")
        return "\n\n".join(sheets_text)
    finally:
        workbook.close()


def _extract_pdf(path: Path) -> str:
    from pypdf import PdfReader

    # pypdf 只能拿到文字流，没有语义层级信息，不硬套 #/表格语法——按页分隔的
    # 纯文本本身也是合法 Markdown，只是不含额外标记，见模块文档说明。
    reader = PdfReader(str(path))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(page.strip() for page in pages if page.strip())


def _rows_to_markdown_table(rows: list[list[str]]) -> str:
    """把二维字符串表格渲染成 Markdown 表格语法（首行作表头）。

    单元格内容里的 `|` 和换行会破坏表格语法（把一个单元格拆成多列/多行），
    渲染前需要转义/替换掉。
    """
    def _escape(cell: str) -> str:
        return cell.replace("|", "\\|").replace("\n", " ").replace("\r", "")

    header, *body = ([_escape(cell) for cell in row] for row in rows)
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(lines)


_HANDLERS = {
    ".docx": _extract_docx,
    ".xlsx": _extract_xlsx,
    ".pdf": _extract_pdf,
}
