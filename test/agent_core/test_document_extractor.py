"""`agent_core.ingestion.document_extractor` 单元测试。

覆盖三种受支持格式的成功路径（含 Markdown 结构还原：标题层级、表格语法）、
以及"提取到空文字层"（典型如扫描版 PDF）的失败路径——后者要求返回明确
原因而不是空字符串，调用方（chat_pipeline）靠这个原因文案向模型/用户
解释，而不是让它看起来像是解析器出了 bug。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.agent_core.ingestion.document_extractor import (
    extract_text,
    is_extractable,
    is_known_unsupported,
)


def test_is_extractable() -> None:
    assert is_extractable(".docx")
    assert is_extractable(".XLSX")  # 大小写不敏感
    assert is_extractable(".pdf")
    assert not is_extractable(".txt")
    assert not is_extractable(".doc")


def test_is_known_unsupported() -> None:
    assert is_known_unsupported(".doc") is not None
    assert is_known_unsupported(".xls") is not None
    assert is_known_unsupported(".docx") is None


def test_extract_docx_heading_paragraph_and_table_as_markdown(tmp_path: Path) -> None:
    from docx import Document

    path = tmp_path / "sample.docx"
    doc = Document()
    doc.add_heading("第一章 概述", level=1)
    doc.add_paragraph("第一段正文")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "列A"
    table.rows[0].cells[1].text = "列B"
    doc.save(str(path))

    result = extract_text(path, ".docx")

    assert result.ok
    assert "# 第一章 概述" in result.text
    assert "第一段正文" in result.text
    assert "| 列A | 列B |" in result.text


def test_extract_docx_table_cell_with_pipe_is_escaped(tmp_path: Path) -> None:
    from docx import Document

    path = tmp_path / "sample.docx"
    doc = Document()
    table = doc.add_table(rows=1, cols=1)
    table.rows[0].cells[0].text = "A|B"
    doc.save(str(path))

    result = extract_text(path, ".docx")

    assert result.ok
    assert "A\\|B" in result.text


def test_extract_xlsx_sheet_content_as_markdown_table(tmp_path: Path) -> None:
    from openpyxl import Workbook

    path = tmp_path / "sample.xlsx"
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "汇总"
    sheet.append(["姓名", "分数"])
    sheet.append(["张三", 90])
    workbook.save(str(path))

    result = extract_text(path, ".xlsx")

    assert result.ok
    assert "## Sheet: 汇总" in result.text
    assert "| 姓名 | 分数 |" in result.text
    assert "| 张三 | 90 |" in result.text


def test_extract_pdf_with_text_layer(tmp_path: Path) -> None:
    pypdf = pytest.importorskip("pypdf")
    from pypdf import PdfWriter

    path = tmp_path / "empty.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)

    # 空白页没有文字层，属于本模块要如实反映的场景：不是解析失败，是内容本身没有文字。
    result = extract_text(path, ".pdf")

    assert not result.ok
    assert "扫描件" in result.message


def test_extract_unsupported_extension(tmp_path: Path) -> None:
    path = tmp_path / "sample.doc"
    path.write_bytes(b"fake")

    result = extract_text(path, ".doc")

    assert not result.ok
    assert "不支持的格式" in result.message


def test_extract_corrupted_file_reports_failure_not_exception(tmp_path: Path) -> None:
    path = tmp_path / "broken.docx"
    path.write_bytes(b"this is not a real docx zip")

    result = extract_text(path, ".docx")

    assert not result.ok
    assert "解析失败" in result.message
