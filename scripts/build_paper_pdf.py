#!/usr/bin/env python3
"""Build a readable PDF from REPORT.md without changing the paper source."""
from __future__ import annotations

import html
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "REPORT.md"
OUTPUT = ROOT / "output" / "pdf" / "gpt2_ioi_research_paper.pdf"


def register_fonts() -> tuple[str, str]:
    regular = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    if not Path(regular).exists():
        regular = "/Library/Fonts/Arial.ttf"
        bold = "/Library/Fonts/Arial Bold.ttf"
    if Path(regular).exists() and Path(bold).exists():
        pdfmetrics.registerFont(TTFont("PaperSans", regular))
        pdfmetrics.registerFont(TTFont("PaperSans-Bold", bold))
        return "PaperSans", "PaperSans-Bold"
    return "Helvetica", "Helvetica-Bold"


def inline_markup(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<font name='Courier'>\1</font>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^*]+)\*", r"<i>\1</i>", text)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^)]+)\)", r"<link href='\2' color='#1769aa'>\1</link>", text)
    return text


def make_styles(font: str, bold_font: str):
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle("PaperTitle", parent=styles["Title"], fontName=bold_font, fontSize=22, leading=27, alignment=TA_CENTER, textColor=colors.HexColor("#18324b"), spaceAfter=8))
    styles.add(ParagraphStyle("Subtitle", parent=styles["Normal"], fontName=font, fontSize=11, leading=15, alignment=TA_CENTER, textColor=colors.HexColor("#4b6478"), spaceAfter=18))
    styles.add(ParagraphStyle("H1Paper", parent=styles["Heading1"], fontName=bold_font, fontSize=16, leading=20, textColor=colors.HexColor("#18324b"), spaceBefore=15, spaceAfter=7, keepWithNext=True))
    styles.add(ParagraphStyle("H2Paper", parent=styles["Heading2"], fontName=bold_font, fontSize=12.5, leading=16, textColor=colors.HexColor("#245b7a"), spaceBefore=11, spaceAfter=5, keepWithNext=True))
    styles.add(ParagraphStyle("H3Paper", parent=styles["Heading3"], fontName=bold_font, fontSize=11, leading=14, textColor=colors.HexColor("#245b7a"), spaceBefore=8, spaceAfter=4, keepWithNext=True))
    styles.add(ParagraphStyle("BodyPaper", parent=styles["BodyText"], fontName=font, fontSize=9.6, leading=14, alignment=TA_LEFT, spaceAfter=7, textColor=colors.HexColor("#20252a")))
    styles.add(ParagraphStyle("QuotePaper", parent=styles["BodyText"], fontName=font, fontSize=9.5, leading=14, leftIndent=18, rightIndent=12, borderPadding=7, borderColor=colors.HexColor("#9db6c7"), borderWidth=0.7, borderLeft=True, spaceBefore=4, spaceAfter=9, textColor=colors.HexColor("#344b5b")))
    styles.add(ParagraphStyle("BulletPaper", parent=styles["BodyText"], fontName=font, fontSize=9.5, leading=13.5, leftIndent=15, firstLineIndent=-9, bulletIndent=3, spaceAfter=4))
    styles.add(ParagraphStyle("SmallPaper", parent=styles["BodyText"], fontName=font, fontSize=8.3, leading=11, textColor=colors.HexColor("#596b78"), spaceAfter=5))
    return styles


def parse_markdown(text: str, styles):
    story = []
    lines = text.replace("\r\n", "\n").split("\n")
    paragraph: list[str] = []
    i = 0

    def flush() -> None:
        if paragraph:
            content = " ".join(x.strip() for x in paragraph).strip()
            if content:
                story.append(Paragraph(inline_markup(content), styles["BodyPaper"]))
            paragraph.clear()

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            flush(); i += 1; continue
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                raw = [x.strip() for x in lines[i].strip().strip("|").split("|")]
                if not all(re.fullmatch(r":?-+:?", x) for x in raw):
                    rows.append([Paragraph(inline_markup(x), styles["SmallPaper"]) for x in raw])
                i += 1
            if rows:
                table = Table(rows, repeatRows=1, colWidths=[1.12 * inch, 2.05 * inch, 1.38 * inch, 1.38 * inch, 1.0 * inch])
                table.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dceaf3")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#18324b")),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#9db6c7")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]))
                story.extend([Spacer(1, 3), table, Spacer(1, 8)])
            continue
        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            flush()
            level = len(heading.group(1))
            style = styles["H1Paper" if level == 1 else "H2Paper" if level == 2 else "H3Paper"]
            story.append(Paragraph(inline_markup(heading.group(2)), style))
            i += 1; continue
        if line.startswith("> "):
            flush(); story.append(Paragraph(inline_markup(line[2:]), styles["QuotePaper"])); i += 1; continue
        if line.startswith("- "):
            flush(); story.append(Paragraph(inline_markup(line[2:]), styles["BulletPaper"], bulletText="•")); i += 1; continue
        if line.startswith("```"):
            flush()
            i += 1
            code = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code.append(lines[i]); i += 1
            story.append(Paragraph(inline_markup(" ".join(code)), styles["SmallPaper"]))
            i += 1; continue
        paragraph.append(line)
        i += 1
    flush()
    return story


def footer(canvas, doc):
    canvas.saveState()
    width, height = LETTER
    canvas.setStrokeColor(colors.HexColor("#c6d4dd"))
    canvas.setLineWidth(0.4)
    canvas.line(doc.leftMargin, 0.53 * inch, width - doc.rightMargin, 0.53 * inch)
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#607584"))
    canvas.drawString(doc.leftMargin, 0.34 * inch, "GPT 2 Small IOI intervention study")
    canvas.drawRightString(width - doc.rightMargin, 0.34 * inch, f"{doc.page}")
    canvas.restoreState()


def main():
    font, bold_font = register_fonts()
    styles = make_styles(font, bold_font)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUTPUT), pagesize=LETTER, rightMargin=0.72 * inch, leftMargin=0.72 * inch, topMargin=0.68 * inch, bottomMargin=0.72 * inch, title="Robustness of Attention Head Intervention Rankings in GPT 2 Small", author="Haider Imran", subject="Independent mechanistic interpretability research report")
    story = parse_markdown(SOURCE.read_text(encoding="utf-8"), styles)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT)


if __name__ == "__main__":
    main()
