"""Render docs/CODE_GUIDE.md as a styled Word document.

The markdown file remains the source of truth; this script keeps a formatted
Word copy in step with it for readers who do not work in a repository.
"""

from __future__ import annotations

import re
import sys
from importlib import import_module
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dm_ahr_vit.paths import artifact_dir  # noqa: E402

pre = import_module("13_preprocessing_report")
h1, h2, body, bullet, table, callout = (
    pre.h1, pre.h2, pre.body, pre.bullet, pre.table, pre.callout)
add_page_number = pre.add_page_number
NAVY, SLATE, GREY, ACCENT = pre.NAVY, pre.SLATE, pre.GREY, pre.ACCENT

SOURCE = Path(__file__).resolve().parents[1] / "docs" / "CODE_GUIDE.md"
OUT = artifact_dir("reports") / "Code_Guide_Report.docx"


def strip_md(text: str) -> str:
    """Remove markdown emphasis markers, keeping the words."""
    text = text.replace("\\*", "\u2217")
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text.strip()


def rich(paragraph, text: str, size: float = 10) -> None:
    """Add text to a paragraph, honouring `code` spans and **bold** runs."""
    text = text.replace("\\*", "\u2217")
    tokens = re.split(r"(`[^`]+`|\*\*[^*]+\*\*|\*[^*]+\*)", text)
    for tok in tokens:
        if not tok:
            continue
        if tok.startswith("`") and tok.endswith("`"):
            r = paragraph.add_run(tok[1:-1])
            r.font.name = "Consolas"
            r.font.size = Pt(size - 0.5)
            r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
        elif tok.startswith("**") and tok.endswith("**"):
            r = paragraph.add_run(tok[2:-2])
            r.font.bold = True
            r.font.size = Pt(size)
            r.font.color.rgb = NAVY
        elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
            r = paragraph.add_run(tok[1:-1])
            r.font.italic = True
            r.font.size = Pt(size)
            r.font.color.rgb = GREY
        else:
            r = paragraph.add_run(tok)
            r.font.size = Pt(size)
            r.font.color.rgb = SLATE


def paragraph(doc, text: str, size: float = 10, space_after: float = 7) -> None:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.15
    rich(p, text, size)


def code_block(doc, lines: list[str]) -> None:
    """Monospaced, shaded block with a left rule."""
    t = doc.add_table(rows=1, cols=1)
    cell = t.rows[0].cells[0]
    cell.width = Inches(6.3)
    pre.shade(cell, "F4F6F9")
    tc_pr = cell._tc.get_or_add_tcPr()
    borders = pre.OxmlElement("w:tcBorders")
    for edge, sz, col in (("left", 18, "1F3864"), ("top", 2, "F4F6F9"),
                          ("bottom", 2, "F4F6F9"), ("right", 2, "F4F6F9")):
        el = pre.OxmlElement(f"w:{edge}")
        el.set(pre.qn("w:val"), "single")
        el.set(pre.qn("w:sz"), str(sz))
        el.set(pre.qn("w:color"), col)
        borders.append(el)
    tc_pr.append(borders)
    cell.text = ""
    first = True
    for line in lines:
        p = cell.paragraphs[0] if first else cell.add_paragraph()
        first = False
        p.paragraph_format.space_after = Pt(0)
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.line_spacing = 1.0
        r = p.add_run(line if line else " ")
        r.font.name = "Consolas"
        r.font.size = Pt(8)
        r.font.color.rgb = RGBColor(0x2B, 0x2B, 0x2B)
    doc.add_paragraph().paragraph_format.space_after = Pt(3)


def split_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def build() -> Path:
    raw = SOURCE.read_text().splitlines()
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Inches(8.27), Inches(11.69)
    sec.left_margin = sec.right_margin = Inches(0.95)
    sec.top_margin, sec.bottom_margin = Inches(0.8), Inches(0.75)
    add_page_number(sec)
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10)
    doc.core_properties.title = "DM-AHR-ViT — Code Guide"
    doc.core_properties.subject = "Code, design decisions and how to defend the study"

    # ---- cover page, replacing the markdown title block
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    r = p.add_run("CODE GUIDE")
    r.font.size = Pt(9); r.font.bold = True; r.font.color.rgb = ACCENT
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(4)
    r = p.add_run("How the pipeline works,\nand why it is built this way")
    r.font.size = Pt(30); r.font.bold = True; r.font.color.rgb = NAVY
    p = doc.add_paragraph(); p.paragraph_format.space_after = Pt(18)
    r = p.add_run("What each part does, what was used, and what it can defend")
    r.font.size = Pt(13); r.font.color.rgb = SLATE
    pre.para_border(p, "C05A33", 12, "bottom")
    body(doc, "DM-AHR-ViT · Explainable Vision Transformer for Skin Lesion "
              "Classification Using Dermoscopy Images", size=10.5)
    body(doc, "Research prototype only. Not clinically validated and not intended "
              "for diagnosis, triage or patient care.", size=9, italic=True)
    callout(doc, "How to read this document",
            "Sections 1–3 orient you: the study in a page, what was actually used, and the "
            "repository map. Section 4 walks each pipeline phase with its reason for existing. "
            "Sections 5–7 are the audit trail: the source of every reported number, each design "
            "decision, and every deviation from the proposal. Sections 8–10 prepare you for "
            "questions: a glossary, likely supervisor questions with grounded answers, and an "
            "explicit list of what is not claimed.")
    doc.add_page_break()

    i = 0
    # skip the markdown H1 title block and its preamble
    while i < len(raw) and not raw[i].startswith("## "):
        i += 1

    # Markdown soft-wraps paragraphs across lines; accumulate them so inline
    # emphasis that spans a line break is still recognised.
    pending: list[str] = []

    def flush() -> None:
        if pending:
            paragraph(doc, " ".join(pending))
            pending.clear()

    while i < len(raw):
        line = raw[i]
        stripped = line.strip()

        if stripped.startswith("```"):
            flush()
            block: list[str] = []
            i += 1
            while i < len(raw) and not raw[i].strip().startswith("```"):
                block.append(raw[i].rstrip())
                i += 1
            i += 1
            code_block(doc, block)
            continue

        if stripped.startswith("### "):
            flush()
            h2(doc, strip_md(stripped[4:]))
            i += 1
            continue

        if stripped.startswith("## "):
            flush()
            h1(doc, strip_md(stripped[3:]))
            i += 1
            continue

        if stripped.startswith("|") and i + 1 < len(raw) and set(
                raw[i + 1].strip()) <= set("|-: "):
            flush()
            headers = [strip_md(c) for c in split_row(stripped)]
            i += 2
            rows = []
            while i < len(raw) and raw[i].strip().startswith("|"):
                rows.append([strip_md(c) for c in split_row(raw[i])])
                i += 1
            n = len(headers)
            width = round(6.3 / n, 2)
            table(doc, headers, rows, widths=[width] * n, font=8.5)
            continue

        if stripped.startswith("- "):
            flush()
            bullet(doc, strip_md(stripped[2:]))
            i += 1
            continue

        if stripped in ("---", ""):
            flush()
            i += 1
            continue

        if stripped.startswith("**") and stripped.endswith("**"):
            flush()
            paragraph(doc, stripped, size=10.5)
            i += 1
            continue

        pending.append(stripped)
        i += 1

    flush()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc.save(OUT)
    return OUT


def main() -> int:
    if not SOURCE.exists():
        print(f"missing {SOURCE}", file=sys.stderr)
        return 1
    out = build()
    print(f"wrote {out} ({out.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
