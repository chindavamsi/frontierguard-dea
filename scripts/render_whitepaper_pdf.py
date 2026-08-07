"""Render the FrontierGuard white paper Markdown as a publication-ready PDF."""

from __future__ import annotations

import argparse
import html
import re
import textwrap
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "paper" / "whitepaper" / "README.md"
DEFAULT_OUTPUT = ROOT / "output" / "pdf" / "frontierguard-dea-white-paper-v0.1.pdf"

NAVY = colors.HexColor("#12263A")
TEAL = colors.HexColor("#22A699")
INK = colors.HexColor("#17212B")
MUTED = colors.HexColor("#52616B")
LIGHT = colors.HexColor("#EEF3F5")
LINE = colors.HexColor("#CCD7DC")


@dataclass(frozen=True)
class PaperMetadata:
    """Publication metadata read from the white paper frontmatter."""

    title: str
    author: str
    date: date
    report_version: str
    software_version: str
    status: str

    @property
    def display_date(self) -> str:
        """Return a human-readable, platform-independent publication date."""
        return f"{self.date.day} {self.date.strftime('%B %Y')}"

    @property
    def title_parts(self) -> tuple[str, str]:
        """Split the publication title into cover title and subtitle."""
        if ": " not in self.title:
            return self.title, ""
        title, subtitle = self.title.split(": ", maxsplit=1)
        return title, subtitle


def _parse_frontmatter(markdown: str) -> tuple[PaperMetadata, str]:
    """Parse required scalar metadata and return the remaining Markdown."""
    lines = markdown.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("white paper must begin with YAML frontmatter")
    try:
        end = lines.index("---", 1)
    except ValueError as error:
        raise ValueError("white paper frontmatter is not terminated") from error

    values: dict[str, str] = {}
    for line in lines[1:end]:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        key, separator, raw_value = line.partition(":")
        if not separator:
            raise ValueError(f"invalid frontmatter line: {line!r}")
        value = raw_value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        values[key.strip()] = value

    required = {
        "title",
        "author",
        "date",
        "report-version",
        "software-version",
        "status",
    }
    missing = sorted(required - values.keys())
    if missing:
        raise ValueError(f"white paper frontmatter is missing: {', '.join(missing)}")

    metadata = PaperMetadata(
        title=values["title"],
        author=values["author"],
        date=date.fromisoformat(values["date"]),
        report_version=values["report-version"],
        software_version=values["software-version"],
        status=values["status"],
    )
    return metadata, "\n".join(lines[end + 1 :])


def _ascii_text(value: str) -> str:
    replacements = {
        "\u00a0": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
        "\u2190": "<-",
        "\u2192": "->",
        "\u2194": "<->",
        "\u2212": "-",
        "\u2264": "<=",
        "\u2265": ">=",
        "\u00d7": "x",
        "\u03b1": "alpha",
        "\u03b2": "beta",
        "\u03b8": "theta",
        "\u03bb": "lambda",
        "\u03c1": "rho",
        "\u03c6": "phi",
    }
    return "".join(replacements.get(char, char) for char in value)


_MATH_COMMANDS = {
    "alpha": " alpha ",
    "beta": " beta ",
    "cdot": " x ",
    "ell": "l",
    "forall": "for all ",
    "ge": ">=",
    "geq": ">=",
    "in": " in ",
    "lambda": " lambda ",
    "ldots": "...",
    "le": "<=",
    "leq": "<=",
    "max": " max ",
    "min": " min ",
    "phi": " phi ",
    "prod": " product ",
    "rho": " rho ",
    "sum": " sum ",
    "theta": " theta ",
    "times": " x ",
    "top": "T",
}
_MATH_OPERATORS = {"max", "min", "prod", "sum"}
_MATH_WRAPPERS = {
    "mathbf",
    "mathbb",
    "mathrm",
    "mathit",
    "mathsf",
    "mathtt",
    "operatorname",
    "text",
}
_MATH_SPACING = {"!", ",", ":", ";", "enspace", "hspace", "quad", "qquad", " "}
_MATH_STRUCTURAL = {"begin", "end", "frac", "left", "right"}
_KNOWN_MATH_COMMANDS = (
    set(_MATH_COMMANDS) | _MATH_WRAPPERS | _MATH_SPACING | _MATH_STRUCTURAL
)


def _read_math_argument(value: str, start: int) -> tuple[str, int]:
    """Read one braced or atomic TeX argument starting at ``start``."""
    index = start
    while index < len(value) and value[index].isspace():
        index += 1
    if index >= len(value):
        return "", index

    if value[index] == "{":
        depth = 1
        cursor = index + 1
        while cursor < len(value) and depth:
            if value[cursor] == "{" and value[cursor - 1] != "\\":
                depth += 1
            elif value[cursor] == "}" and value[cursor - 1] != "\\":
                depth -= 1
            cursor += 1
        if depth:
            raise ValueError("unbalanced braces in mathematical expression")
        return value[index + 1 : cursor - 1], cursor

    if value[index] == "\\":
        command = re.match(r"\\(?:[A-Za-z]+|.)", value[index:])
        if command:
            return command.group(0), index + len(command.group(0))
    return value[index], index + 1


def _format_math_script(operator: str, argument: str) -> str:
    """Format a TeX subscript or superscript in readable linear notation."""
    rendered = _latex_to_plain(argument).strip()
    if not rendered:
        return operator
    if len(rendered) == 1 or rendered in {"T", "+", "-"}:
        return f"{operator}{rendered}"
    if rendered.startswith("(") and rendered.endswith(")"):
        return f"{operator}{rendered}"
    return f"{operator}({rendered})"


_RAW_MATH_CONTROL = re.compile(
    r"\\|"
    r"\b(?:begin|end|frac|mathbb|mathbf|mathrm|operatorname|text)"
    r"\s*\{"
)


def _validate_math_conversion(source: str, rendered: str) -> None:
    """Reject a conversion that leaks TeX control syntax into the PDF."""
    residual = _RAW_MATH_CONTROL.search(rendered)
    if residual:
        excerpt_start = max(0, residual.start() - 20)
        excerpt_end = min(len(rendered), residual.end() + 40)
        excerpt = rendered[excerpt_start:excerpt_end]
        raise ValueError(
            "mathematical conversion left raw TeX syntax in "
            f"{excerpt!r}; source was {source!r}"
        )


def _validate_math_source(source: str) -> None:
    """Reject unhandled TeX commands before silently degrading an equation."""
    commands = re.findall(r"\\([A-Za-z]+|.)", source)
    unknown = sorted(
        {
            command
            for command in commands
            if command not in _KNOWN_MATH_COMMANDS and command != "\\"
        }
    )
    if unknown:
        raise ValueError(f"unsupported mathematical TeX commands: {', '.join(unknown)}")


def _render_math_text(value: str) -> str:
    """Convert and validate a complete inline or display equation."""
    _validate_math_source(value)
    rendered = _latex_to_plain(value)
    _validate_math_conversion(value, rendered)
    return rendered


def _latex_to_plain(value: str) -> str:
    """Convert the white paper's TeX subset to unambiguous linear notation."""
    value = _ascii_text(value)
    output: list[str] = []
    index = 0
    pending_operator_operand = False
    while index < len(value):
        char = value[index]

        if pending_operator_operand and char not in {"_", "^"}:
            if not char.isspace():
                output.append(" ")
            pending_operator_operand = False

        if char == "\\":
            if index + 1 < len(value) and value[index + 1] == "\\":
                output.append("\n")
                index += 2
                continue

            command_match = re.match(r"\\([A-Za-z]+|.)", value[index:])
            if not command_match:
                index += 1
                continue
            command = command_match.group(1)
            index += len(command_match.group(0))

            if command in {"begin", "end"}:
                _, index = _read_math_argument(value, index)
            elif command == "frac":
                numerator, index = _read_math_argument(value, index)
                denominator, index = _read_math_argument(value, index)
                rendered_numerator = _latex_to_plain(numerator)
                rendered_denominator = _latex_to_plain(denominator)
                if not re.fullmatch(r"[\w.^_()|]+", rendered_numerator):
                    rendered_numerator = f"({rendered_numerator})"
                if not re.fullmatch(r"[\w.^_()|]+", rendered_denominator):
                    rendered_denominator = f"({rendered_denominator})"
                output.append(f"{rendered_numerator} / {rendered_denominator}")
            elif command == "text":
                argument, index = _read_math_argument(value, index)
                output.append(_ascii_text(argument).replace("-", "\ue000"))
            elif command in _MATH_WRAPPERS:
                argument, index = _read_math_argument(value, index)
                output.append(_latex_to_plain(argument))
            elif command in {"left", "right"}:
                continue
            elif command in _MATH_SPACING:
                output.append(" ")
            elif command in _MATH_OPERATORS:
                output.append(_MATH_COMMANDS[command])
                pending_operator_operand = True
            elif command in _MATH_COMMANDS:
                output.append(_MATH_COMMANDS[command])
            else:
                # Keep unknown commands legible without leaking TeX syntax.
                output.append(command.replace("_", " "))
            continue

        if char in {"_", "^"}:
            argument, index = _read_math_argument(value, index + 1)
            output.append(_format_math_script(char, argument))
            continue

        if char == "{":
            argument, index = _read_math_argument(value, index)
            output.append(_latex_to_plain(argument))
            continue
        if char == "}":
            index += 1
            continue
        if char == "&":
            index += 1
            continue
        if char == "~":
            output.append(" ")
            index += 1
            continue

        output.append(char)
        index += 1

    lines: list[str] = []
    for raw_line in "".join(output).splitlines():
        line = re.sub(r"\s+", " ", raw_line).strip()
        line = re.sub(r"\s+([_^])", r"\1", line)
        line = re.sub(r"\s+([,.;:])", r"\1", line)
        line = re.sub(r",(?=\S)", ", ", line)
        line = re.sub(r"\s*([=<>]+)\s*", r" \1 ", line)
        line = re.sub(r"(?<!\^)\s*\+\s*", " + ", line)
        line = re.sub(r"(?<!\^)\s*-\s*", " - ", line)
        line = re.sub(r"\s+\+\s*$", " +", line)
        line = line.replace("\ue000", "-")
        line = re.sub(r"\s+", " ", line).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _inline_markup(value: str) -> str:
    value = re.sub(r'<a\s+id="[^"]+"></a>\s*', "", value)
    value = re.sub(
        r"\\\((.+?)\\\)",
        lambda match: _render_math_text(match.group(1)),
        value,
    )
    escaped = html.escape(_ascii_text(value), quote=True)

    def replace_link(match: re.Match[str]) -> str:
        label, target = match.groups()
        if html.unescape(target).startswith(("https://", "http://", "mailto:")):
            return f'<link href="{target}" color="#137C76">{label}</link>'
        return f'<font color="#137C76">{label}</font>'

    escaped = re.sub(
        r"\[([^\]]+)\]\(([^)]+)\)",
        replace_link,
        escaped,
    )
    escaped = re.sub(r"`([^`]+)`", r'<font name="Courier">\1</font>', escaped)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", escaped)
    return escaped


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=28,
            leading=33,
            alignment=TA_LEFT,
            textColor=colors.white,
            spaceAfter=10 * mm,
        ),
        "cover_subtitle": ParagraphStyle(
            "CoverSubtitle",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=15,
            leading=21,
            textColor=colors.HexColor("#D9F3F0"),
            spaceAfter=20 * mm,
        ),
        "cover_meta": ParagraphStyle(
            "CoverMeta",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=10.5,
            leading=17,
            textColor=colors.white,
        ),
        "h1": ParagraphStyle(
            "Heading1",
            parent=base["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=18,
            leading=22,
            textColor=NAVY,
            spaceBefore=10 * mm,
            spaceAfter=4 * mm,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13.5,
            leading=17,
            textColor=colors.HexColor("#174F59"),
            spaceBefore=7 * mm,
            spaceAfter=3 * mm,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Heading3",
            parent=base["Heading3"],
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            textColor=INK,
            spaceBefore=5 * mm,
            spaceAfter=2 * mm,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.6,
            alignment=TA_JUSTIFY,
            textColor=INK,
            spaceAfter=2.8 * mm,
        ),
        "bullet": ParagraphStyle(
            "Bullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            leftIndent=7 * mm,
            firstLineIndent=-4 * mm,
            bulletIndent=1 * mm,
            textColor=INK,
            spaceAfter=1.6 * mm,
        ),
        "quote": ParagraphStyle(
            "Quote",
            parent=base["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=9,
            leading=13.2,
            leftIndent=7 * mm,
            rightIndent=5 * mm,
            borderColor=TEAL,
            borderWidth=0,
            borderPadding=(2 * mm, 3 * mm, 2 * mm, 4 * mm),
            textColor=MUTED,
            spaceAfter=3 * mm,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=base["Code"],
            fontName="Courier",
            fontSize=6.8,
            leading=9.2,
            textColor=INK,
        ),
        "equation": ParagraphStyle(
            "Equation",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=13,
            textColor=NAVY,
            leftIndent=2 * mm,
            rightIndent=2 * mm,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base["Normal"],
            fontName="Helvetica",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
        ),
        "table": ParagraphStyle(
            "Table",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=7.1,
            leading=9.5,
            textColor=INK,
        ),
        "table_header": ParagraphStyle(
            "TableHeader",
            parent=base["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=7.2,
            leading=9.7,
            textColor=colors.white,
        ),
    }


class WhitePaperDoc(BaseDocTemplate):
    """Document template that adds PDF outline entries for report headings."""

    def afterFlowable(self, flowable: Flowable) -> None:  # noqa: N802
        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        if style_name not in {"Heading1", "Heading2"}:
            return
        if style_name == "Heading1":
            self._has_level_zero_heading = True
            level = 0
        else:
            level = 1 if getattr(self, "_has_level_zero_heading", False) else 0
        key = f"heading-{self.seq.nextf('heading')}"
        title = flowable.getPlainText()
        self.canv.bookmarkPage(key)
        self.canv.addOutlineEntry(title, key, level=level, closed=False)


def _document_metadata(document: BaseDocTemplate) -> PaperMetadata:
    metadata = getattr(document, "paper_metadata", None)
    if not isinstance(metadata, PaperMetadata):
        raise TypeError("document is missing parsed paper metadata")
    return metadata


def _cover_page(canvas: Canvas, document: BaseDocTemplate) -> None:
    metadata = _document_metadata(document)
    width, height = A4
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, 0, width, height, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, height - 11 * mm, width, 11 * mm, fill=1, stroke=0)
    canvas.setFillColor(colors.HexColor("#3D5A6C"))
    canvas.circle(width - 22 * mm, 24 * mm, 45 * mm, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.circle(width - 18 * mm, 28 * mm, 18 * mm, fill=1, stroke=0)
    canvas.setTitle(metadata.title)
    canvas.setAuthor(metadata.author)
    canvas.setSubject(f"FrontierGuard DEA Technical Report {metadata.report_version}")
    canvas.setKeywords(
        "data envelopment analysis, efficiency, productivity, network DEA, "
        "reproducible research, research software"
    )
    canvas.restoreState()


def _body_page(canvas: Canvas, document: BaseDocTemplate) -> None:
    metadata = _document_metadata(document)
    width, height = A4
    canvas.saveState()
    canvas.setStrokeColor(LINE)
    canvas.setLineWidth(0.4)
    canvas.line(20 * mm, height - 16 * mm, width - 20 * mm, height - 16 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(MUTED)
    canvas.drawString(20 * mm, height - 12 * mm, "FRONTIERGUARD DEA")
    right_text = (
        f"TECHNICAL REPORT {metadata.report_version} - "
        f"SOFTWARE {metadata.software_version}"
    )
    canvas.drawRightString(width - 20 * mm, height - 12 * mm, right_text)
    canvas.line(20 * mm, 15 * mm, width - 20 * mm, 15 * mm)
    canvas.drawString(
        20 * mm,
        10.5 * mm,
        f"{metadata.author} - {metadata.display_date}",
    )
    page_text = str(document.page - 1)
    canvas.drawRightString(width - 20 * mm, 10.5 * mm, page_text)
    canvas.restoreState()


def _code_block(value: str, styles: dict[str, ParagraphStyle], width: float) -> Table:
    wrapped_lines: list[str] = []
    for line in _ascii_text(value).splitlines() or [""]:
        wrapped_lines.extend(
            textwrap.wrap(
                line,
                width=100,
                replace_whitespace=False,
                drop_whitespace=False,
            )
            or [""]
        )
    pre = Preformatted("\n".join(wrapped_lines), styles["code"])
    table = Table([[pre]], colWidths=[width], hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("BOX", (0, 0), (-1, -1), 0.35, LINE),
                ("LEFTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2 * mm),
            ]
        )
    )
    return table


def _equation_block(
    value: str, styles: dict[str, ParagraphStyle], width: float
) -> Table:
    """Render a display equation as polished, readable linear notation."""
    equation = _render_math_text(value)
    lines = equation.splitlines() or [""]
    paragraphs = [
        [Paragraph(html.escape(line, quote=True), styles["equation"])] for line in lines
    ]
    table = Table(paragraphs, colWidths=[width], hAlign="LEFT", splitByRow=0)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F7FAFA")),
                ("LINEBEFORE", (0, 0), (0, -1), 1.6, TEAL),
                ("LEFTPADDING", (0, 0), (-1, -1), 5 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4 * mm),
                ("TOPPADDING", (0, 0), (0, 0), 2.5 * mm),
                ("BOTTOMPADDING", (0, -1), (0, -1), 2.5 * mm),
            ]
        )
    )
    return table


def _table_block(
    lines: list[str], styles: dict[str, ParagraphStyle], width: float
) -> Table:
    parsed: list[list[str]] = []
    for line in lines:
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if cells and all(re.fullmatch(r":?-{3,}:?", cell) for cell in cells):
            continue
        parsed.append(cells)
    column_count = max(len(row) for row in parsed)
    normalized = [row + [""] * (column_count - len(row)) for row in parsed]
    rows: list[list[Paragraph]] = []
    for row_index, row in enumerate(normalized):
        style = styles["table_header"] if row_index == 0 else styles["table"]
        rows.append([Paragraph(_inline_markup(cell), style) for cell in row])
    if column_count == 2:
        widths = [width * 0.30, width * 0.70]
    elif column_count == 3:
        widths = [width * 0.18, width * 0.32, width * 0.50]
    else:
        widths = [width / column_count] * column_count
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("GRID", (0, 0), (-1, -1), 0.3, LINE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2 * mm),
                ("TOPPADDING", (0, 0), (-1, -1), 1.7 * mm),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 1.7 * mm),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
            ]
        )
    )
    return table


def _body_story(markdown: str, width: float) -> list[Flowable]:
    styles = _styles()
    lines = markdown.splitlines()
    while lines and (not lines[0].strip() or lines[0].startswith(("# ", "## ", "**"))):
        lines.pop(0)

    story: list[Flowable] = []
    paragraph_lines: list[str] = []
    quote_lines: list[str] = []
    code_lines: list[str] = []
    math_lines: list[str] = []
    table_lines: list[str] = []
    in_code = False
    in_math = False

    def flush_paragraph() -> None:
        if paragraph_lines:
            text = " ".join(part.strip() for part in paragraph_lines)
            story.append(Paragraph(_inline_markup(text), styles["body"]))
            paragraph_lines.clear()

    def flush_quote() -> None:
        if quote_lines:
            text = " ".join(part.strip() for part in quote_lines)
            story.append(Paragraph(_inline_markup(text), styles["quote"]))
            quote_lines.clear()

    def flush_table() -> None:
        if table_lines:
            story.extend(
                [
                    Spacer(1, 1.5 * mm),
                    _table_block(table_lines, styles, width),
                    Spacer(1, 3 * mm),
                ]
            )
            table_lines.clear()

    def flush_code() -> None:
        if code_lines:
            story.extend(
                [
                    Spacer(1, 1.2 * mm),
                    _code_block("\n".join(code_lines), styles, width),
                    Spacer(1, 3 * mm),
                ]
            )
            code_lines.clear()

    def flush_math() -> None:
        if math_lines:
            story.extend(
                [
                    Spacer(1, 1.2 * mm),
                    _equation_block("\n".join(math_lines), styles, width),
                    Spacer(1, 3 * mm),
                ]
            )
            math_lines.clear()

    for raw_line in lines:
        line = raw_line.rstrip()

        if line.startswith("```"):
            flush_paragraph()
            flush_quote()
            flush_table()
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line)
            continue

        if line.strip() == r"\[":
            flush_paragraph()
            if in_math:
                raise ValueError("nested display-math block")
            in_math = True
            continue
        if line.strip() == r"\]":
            if not in_math:
                raise ValueError("display-math block closed before it was opened")
            flush_math()
            in_math = False
            continue
        if in_math:
            math_lines.append(line)
            continue

        if line.startswith("|"):
            flush_paragraph()
            flush_quote()
            table_lines.append(line)
            continue
        flush_table()

        if line.startswith(">"):
            flush_paragraph()
            quote_lines.append(line.lstrip("> ").strip())
            continue
        flush_quote()

        if not line.strip():
            flush_paragraph()
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            story.append(
                Paragraph(_inline_markup(heading.group(2)), styles[f"h{level}"])
            )
            continue

        bullet = re.match(r"^\s*[-*]\s+(.+)$", line)
        numbered = re.match(r"^\s*(\d+)\.\s+(.+)$", line)
        if bullet or numbered:
            flush_paragraph()
            if bullet:
                marker = "-"
                content = bullet.group(1)
            else:
                assert numbered is not None
                marker = f"{numbered.group(1)}."
                content = numbered.group(2)
            story.append(
                Paragraph(
                    _inline_markup(content),
                    styles["bullet"],
                    bulletText=marker,
                )
            )
            continue

        if line.strip() == "---":
            flush_paragraph()
            story.append(Spacer(1, 2 * mm))
            continue

        paragraph_lines.append(line)

    if in_math:
        raise ValueError("display-math block was not closed")
    flush_paragraph()
    flush_quote()
    flush_table()
    flush_code()
    return story


def render(source: Path, output: Path) -> None:
    """Render ``source`` Markdown into ``output`` PDF."""
    metadata, markdown = _parse_frontmatter(source.read_text(encoding="utf-8"))
    output.parent.mkdir(parents=True, exist_ok=True)
    width, height = A4
    cover_frame = Frame(
        24 * mm,
        24 * mm,
        width - 48 * mm,
        height - 48 * mm,
        id="cover-frame",
        showBoundary=0,
    )
    body_frame = Frame(
        20 * mm,
        20 * mm,
        width - 40 * mm,
        height - 40 * mm,
        id="body-frame",
        showBoundary=0,
    )
    document = WhitePaperDoc(
        str(output),
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=20 * mm,
        pageTemplates=[
            PageTemplate(id="cover", frames=[cover_frame], onPage=_cover_page),
            PageTemplate(id="body", frames=[body_frame], onPage=_body_page),
        ],
        title=metadata.title,
        author=metadata.author,
        subject="Auditable and reproducible data envelopment analysis",
        invariant=1,
    )
    document.paper_metadata = metadata

    styles = _styles()
    cover_title, cover_subtitle = metadata.title_parts
    story: list[Flowable] = [
        Spacer(1, 48 * mm),
        Paragraph(_inline_markup(cover_title), styles["cover_title"]),
        Paragraph(_inline_markup(cover_subtitle), styles["cover_subtitle"]),
        Paragraph(
            f"<b>{html.escape(metadata.author)}</b><br/>"
            f"Technical Report {html.escape(metadata.report_version)}<br/>"
            f"Software version {html.escape(metadata.software_version)}<br/>"
            f"{html.escape(metadata.display_date)}<br/><br/>"
            f"<b>{html.escape(metadata.status)}</b>",
            styles["cover_meta"],
        ),
        Spacer(1, 32 * mm),
        Paragraph(
            "OPEN RESEARCH SOFTWARE  /  MIT LICENSE  /  "
            "GITHUB.COM/CHINDAVAMSI/FRONTIERGUARD-DEA",
            styles["cover_meta"],
        ),
        NextPageTemplate("body"),
        PageBreak(),
    ]
    body_width = width - 40 * mm
    story.extend(_body_story(markdown, body_width))
    document.build(story)


def main() -> int:
    """Parse command-line paths and render the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    render(args.source.resolve(), args.output.resolve())
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
