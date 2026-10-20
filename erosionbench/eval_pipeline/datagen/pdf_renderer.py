# Copyright (c) 2026, Salesforce, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Stage 2b: PDF Renderer — converts generated markdown documents to enterprise PDFs.

At execution time, agents interact with these documents through a simulated
document retrieval tool (RAG search / fetch_document). The PDFs add realistic
friction: the agent must parse formatted content, navigate headers/tables,
and find poisoned evidence buried across pages — not just scan a raw string.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from fpdf import FPDF

logger = logging.getLogger(__name__)

__all__ = ["render_scenario_pdfs", "render_document_to_pdf"]


def render_scenario_pdfs(
    scenario_id: str,
    rendered_documents: List[Dict[str, Any]],
    output_dir: Path,
) -> List[Dict[str, Any]]:
    """Render all documents for a scenario to PDF files.

    Returns the rendered_documents list with an added 'pdf_path' field
    pointing to the generated PDF file for each document.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for doc in rendered_documents:
        title = doc.get("title", "document")
        pages = doc.get("pages", [])
        if not pages:
            continue

        filename = f"{scenario_id}_{title}.pdf"
        pdf_path = output_dir / filename

        try:
            render_document_to_pdf(
                pages=pages,
                title=title,
                output_path=pdf_path,
                metadata={
                    "scenario_id": scenario_id,
                    "total_pages": doc.get("total_pages", len(pages)),
                    "purpose": doc.get("purpose", ""),
                },
            )
            doc["pdf_path"] = str(pdf_path)
            logger.debug(f"Rendered PDF: {pdf_path} ({pdf_path.stat().st_size / 1024:.1f}KB)")
        except Exception as e:
            logger.warning(f"PDF render failed for {scenario_id}/{title}: {e}")
            doc["pdf_path"] = None

    return rendered_documents


def render_document_to_pdf(
    pages: List[str],
    title: str,
    output_path: Path,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Render a list of markdown page strings to a formatted enterprise PDF."""
    pdf = _EnterprisePDF(title=title, metadata=metadata or {})
    pdf.set_auto_page_break(auto=True, margin=20)

    for page_idx, page_content in enumerate(pages):
        pdf.add_page()
        _render_markdown_page(pdf, page_content, page_idx + 1, len(pages))

    pdf.output(str(output_path))


class _EnterprisePDF(FPDF):
    """Enterprise-styled PDF with headers, footers, and consistent formatting."""

    FONT = "Helvetica"

    def __init__(self, title: str = "", metadata: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(**kwargs)
        self._doc_title = title
        self._metadata = metadata or {}
        self.set_margins(left=20, top=25, right=20)

    def header(self):
        self.set_font(self.FONT, "I", 8)
        self.set_text_color(100, 100, 100)
        # Realistic enterprise letterhead only. MUST NOT reveal that this is a benchmark/eval — a
        # "SECURITY BENCHMARK" banner would tell the reading agent it is under test and invalidate
        # the attack. Keep it to a plausible confidential-document classification.
        self.cell(0, 8, _to_latin1(f"{self._doc_title}  |  CONFIDENTIAL - INTERNAL USE ONLY"), align="C")
        self.ln(4)
        self.set_draw_color(180, 180, 180)
        self.line(20, self.get_y(), self.w - 20, self.get_y())
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font(self.FONT, "I", 8)
        self.set_text_color(100, 100, 100)
        # Just the running PDF page number, ascending. Do NOT print "of N": the logical page count
        # from datagen (metadata['total_pages']) differs from the rendered PDF page count because
        # dense prose reflows across auto-page-breaks, which produced nonsense like "Page 68 of 22".
        self.cell(0, 10, _to_latin1(f"Page {self.page_no()}  |  Classification: Internal"), align="C")


def _render_markdown_page(pdf: _EnterprisePDF, content: str, page_num: int, total_pages: int) -> None:
    """Parse markdown content and render to PDF with formatting."""
    F = pdf.FONT
    lines = content.split("\n")
    i = 0
    in_table = False
    table_rows: List[List[str]] = []

    while i < len(lines):
        line = lines[i]

        if in_table:
            if line.strip().startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if not all(c.replace("-", "").replace(":", "") == "" for c in cells):
                    table_rows.append(cells)
                i += 1
                continue
            else:
                _render_table(pdf, table_rows)
                table_rows = []
                in_table = False
                continue

        if line.strip().startswith("|") and i + 1 < len(lines) and "---" in lines[i + 1]:
            in_table = True
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            table_rows.append(cells)
            i += 1
            continue

        if line.startswith("# "):
            pdf.set_font(F, "B", 16)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 9, _to_latin1(line[2:].strip().replace("**", "")))
            pdf.ln(4)
        elif line.startswith("## "):
            pdf.ln(3)
            pdf.set_font(F, "B", 13)
            pdf.set_text_color(30, 30, 30)
            pdf.multi_cell(0, 8, _to_latin1(line[3:].strip().replace("**", "")))
            pdf.ln(2)
        elif line.startswith("### "):
            pdf.ln(2)
            pdf.set_font(F, "B", 11)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 7, _to_latin1(line[4:].strip().replace("**", "")))
            pdf.ln(2)
        elif line.startswith("#### "):
            pdf.ln(1)
            pdf.set_font(F, "B", 10)
            pdf.set_text_color(50, 50, 50)
            pdf.multi_cell(0, 6, _to_latin1(line[5:].strip().replace("**", "")))
            pdf.ln(1)
        elif line.startswith("---"):
            pdf.ln(3)
            pdf.set_draw_color(180, 180, 180)
            pdf.line(20, pdf.get_y(), pdf.w - 20, pdf.get_y())
            pdf.ln(5)
        elif line.strip().startswith("- ") or line.strip().startswith("* "):
            pdf.set_font(F, "", 9)
            pdf.set_text_color(0, 0, 0)
            bullet_text = _strip_bold_italic(line.strip()[2:])
            indent = 25 + (len(line) - len(line.lstrip())) * 2
            pdf.set_x(indent)
            pdf.cell(4, 5, "-")
            pdf.multi_cell(0, 5, _to_latin1(f" {bullet_text}"))
            pdf.ln(1)
        elif re.match(r"^\d+\.\s", line.strip()):
            pdf.set_font(F, "", 9)
            pdf.set_text_color(0, 0, 0)
            text = _strip_bold_italic(line.strip())
            pdf.set_x(25)
            pdf.multi_cell(0, 5, _to_latin1(text))
            pdf.ln(1)
        elif line.strip().startswith("**") and line.strip().endswith("**"):
            pdf.set_font(F, "B", 9)
            pdf.set_text_color(0, 0, 0)
            pdf.multi_cell(0, 5, _to_latin1(line.strip().strip("*")))
            pdf.ln(1)
        elif line.strip() == "":
            pdf.ln(3)
        else:
            pdf.set_font(F, "", 9)
            pdf.set_text_color(0, 0, 0)
            cleaned = _strip_bold_italic(line)
            pdf.multi_cell(0, 5, _to_latin1(cleaned))
            pdf.ln(1)

        i += 1

    if table_rows:
        _render_table(pdf, table_rows)


def _render_table(pdf: _EnterprisePDF, rows: List[List[str]]) -> None:
    """Render a table as fixed-width cells whose text is measured and clipped to fit.

    The prior implementation drew fixed-width ``cell()`` boxes but printed each cell's full text, so
    a value wider than its column (a 64-char SHA in a 28 mm column) spilled ~45 mm into the
    neighbouring cells and overlapped their text. Here every cell is truncated with an ellipsis to
    its column width BEFORE drawing, so nothing ever overflows. This is deliberately O(cells) and
    fast — fpdf2's auto-wrapping ``table()`` was tried but lays wide many-column inventories (the
    datagen model emits up to ~18 columns of long hashes) out character-by-character and hangs.

    Safe for the benchmark: needles are spliced into PROSE, never into table cells, so clipping a
    filler SHA/PURL in an inventory table cannot truncate an attack payload. Column count is capped
    so ultra-wide tables stay legible; overflow columns fold into a final cell.
    """
    if not rows:
        return

    num_cols = max(len(r) for r in rows)
    norm_rows: List[List[str]] = []
    for row in rows:
        cells = [_to_latin1(_strip_bold_italic(c)) for c in row]
        cells += [""] * (num_cols - len(cells))
        norm_rows.append(cells)

    table_width = pdf.w - 40  # body width between margins
    max_cols = 8              # more than this is unreadable portrait; fold the remainder
    if num_cols > max_cols:
        keep = max_cols - 1
        norm_rows = [r[:keep] + [" ".join(x for x in r[keep:] if x)] for r in norm_rows]
        num_cols = max_cols

    font_pt = 8 if num_cols <= 5 else (7 if num_cols <= 7 else 6)
    col_width = table_width / num_cols
    row_height = font_pt * 0.42 + 1.5  # mm, scales with font

    def _clip(text: str) -> str:
        """Trim text (adding an ellipsis) until it fits within col_width at the current font."""
        if pdf.get_string_width(text) <= col_width - 2:
            return text
        ell = ".."
        t = text
        while t and pdf.get_string_width(t + ell) > col_width - 2:
            t = t[:-1]
        return (t + ell) if t else ell

    pdf.ln(2)
    for row_idx, row in enumerate(norm_rows):
        if row_idx == 0:
            pdf.set_font(pdf.FONT, "B", font_pt)
            pdf.set_fill_color(220, 220, 220)
            fill = True
        else:
            pdf.set_font(pdf.FONT, "", font_pt)
            fill = row_idx % 2 == 0
            if fill:
                pdf.set_fill_color(248, 248, 248)
        pdf.set_text_color(0, 0, 0)
        for cell in row[:num_cols]:
            pdf.cell(col_width, row_height, _clip(cell), border=1, fill=fill)
        pdf.ln(row_height)

    pdf.ln(3)


def _strip_bold_italic(text: str) -> str:
    """Remove markdown bold/italic markers."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text


#: Explicit transliterations for common non-latin1 glyphs the datagen model emits (punctuation,
#: math, Greek, arrows). Anything mapped here renders as a readable ASCII equivalent instead of
#: being dropped to "?". Greek/math matter because quant/risk docs use them (and a "?" in a needle
#: value would corrupt the attack), so map them rather than lose them.
_TRANSLITERATIONS = {
    "—": "--", "–": "-", "‑": "-", "―": "--",       # dashes
    "‘": "'", "’": "'", "‚": "'", "“": '"', "”": '"', "„": '"',  # quotes
    "…": "...", "•": "-", "▸": ">", "◦": "-", "·": "-",           # bullets/ellipsis
    "→": "->", "←": "<-", "↔": "<->", "↓": "v", "↑": "^", "⇒": "=>",  # arrows
    "≤": "<=", "≥": ">=", "≠": "!=", "±": "+/-", "×": "x", "÷": "/",  # math ops
    "−": "-", "∈": "in", "∞": "inf", "≈": "~", "√": "sqrt", "∑": "sum",
    "Δ": "Delta", "δ": "delta", "α": "alpha", "β": "beta", "γ": "gamma",
    "λ": "lambda", "μ": "mu", "σ": "sigma", "ρ": "rho", "θ": "theta", "π": "pi",
    "§": "Sec.", "¶": "P.", "™": "(TM)", "®": "(R)", "©": "(C)", "°": "deg",
    "€": "EUR", "£": "GBP", "¥": "JPY", "​": "", "﻿": "", "\xa0": " ",
}


def _to_latin1(text: str) -> str:
    """Make ``text`` safe for fpdf's latin-1 core font WITHOUT silently corrupting content.

    Order: (1) apply explicit transliterations (dashes, quotes, math, Greek, currency); (2) for any
    remaining char that is not latin-1 encodable, strip accents via NFKD (é->e, ü->u) so accented
    names stay readable; (3) only if a char is still unrepresentable does it fall back to "?". This
    prevents the prior behaviour where any unmapped glyph (Greek, math symbols) became "?" — which
    could mangle a needle value or degrade the blend.
    """
    import unicodedata

    for k, v in _TRANSLITERATIONS.items():
        text = text.replace(k, v)

    def _fix_char(ch: str) -> str:
        try:
            ch.encode("latin-1")
            return ch
        except UnicodeEncodeError:
            decomposed = unicodedata.normalize("NFKD", ch)
            ascii_form = "".join(c for c in decomposed if not unicodedata.combining(c))
            try:
                ascii_form.encode("latin-1")
                return ascii_form or "?"
            except UnicodeEncodeError:
                return "?"

    return "".join(_fix_char(c) for c in text)
