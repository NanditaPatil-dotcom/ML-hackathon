#!/usr/bin/env python3
"""Add the current raw-data quality/connectivity audit to the report HTML/PDF."""
from __future__ import annotations

import base64
import html
import io
from pathlib import Path

import pandas as pd
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "report"
EDA_DIR = ROOT / "results" / "eda"
HTML_REPORT = REPORT_DIR / "PathSnap_Technical_Report.html"
PDF_REPORT = REPORT_DIR / "PathSnap_Technical_Report.pdf"
START_MARKER = "<!-- EDA_AUDIT_APPENDIX_START -->"
END_MARKER = "<!-- EDA_AUDIT_APPENDIX_END -->"
PDF_MARKER = "APPENDIX A | DATASET QUALITY AND ROAD-NETWORK CONNECTIVITY"


def fmt(value: object) -> str:
    if isinstance(value, (float, int)):
        if isinstance(value, float) and value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
    return str(value)


def create_html_appendix(rows: pd.DataFrame) -> str:
    quality_cols = [
        ("partition", "Partition"), ("trips", "Trips"),
        ("raw_gps_observations", "GPS rows"), ("missing_gps_rows", "Missing"),
        ("non_numeric_gps_rows", "Non-numeric"), ("nonfinite_gps_rows", "Non-finite"),
        ("invalid_coordinate_rows", "Invalid coordinates"), ("duplicate_gps_rows", "Duplicates"),
        ("nonmonotonic_timestamp_steps", "Time backsteps"), ("gaps_over_2s", "Gaps >2 s"),
    ]
    network_cols = [
        ("partition", "Partition"), ("invalid_nodes", "Invalid nodes"),
        ("invalid_arcs", "Invalid arcs"), ("duplicate_arcs", "Duplicate arcs"),
        ("self_loop_arcs", "Self-loops"), ("isolated_nodes", "Isolated nodes"),
        ("trips_with_multiple_weak_components", "Trips with >1 component"),
        ("weak_components_median_per_trip", "Median components/trip"),
        ("weak_components_max_per_trip", "Maximum components"),
        ("largest_component_fraction_median", "Median largest-component share"),
    ]

    def table(cols):
        header = "".join(f"<th>{html.escape(label)}</th>" for _, label in cols)
        body = "".join(
            "<tr>" + "".join(f"<td>{html.escape(fmt(row[key]))}</td>" for key, _ in cols) + "</tr>"
            for row in rows.to_dict(orient="records")
        )
        return f'<table class="data-table eda-audit-table"><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>'

    train = rows.loc[rows.partition == "train"].iloc[0]
    test = rows.loc[rows.partition == "student_test"].iloc[0]
    gps_img = base64.b64encode((EDA_DIR / "gps_data_quality.png").read_bytes()).decode("ascii")
    graph_img = base64.b64encode((EDA_DIR / "road_network_connectivity.png").read_bytes()).decode("ascii")
    return f"""{START_MARKER}
<h2>Appendix A. Dataset Quality and Road-Network Connectivity</h2>
<p>This audit was generated from the available raw <code>train</code> and <code>student_test</code> partitions (80 and 10 trips). The reserved evaluator-test partition was not accessed. A GPS row is marked invalid when a coordinate is non-finite or outside longitude/latitude bounds. Road connectivity is reported as weak connectivity, treating directed arcs as undirected only for component membership.</p>
<h3>GPS data-quality checks</h3>
{table(quality_cols)}
<p>No missing, non-numeric, non-finite, out-of-range, duplicate, or non-increasing-timestamp GPS rows were found in either available partition. There were {int(train.gaps_over_2s + test.gaps_over_2s):,} timestamp gaps longer than two seconds: {int(train.gaps_over_2s):,} in training and {int(test.gaps_over_2s):,} in student testing. These are sampling gaps, not missing rows.</p>
<div class="figure-box"><img src="data:image/png;base64,{gps_img}" alt="Raw GPS data quality counts"><div class="caption">Raw GPS quality checks by partition. Zero-count categories are labeled explicitly.</div></div>
<h3>Road-network integrity and connectivity</h3>
{table(network_cols)}
<p>No invalid-coordinate nodes, out-of-range arc endpoints, or isolated nodes were found. Duplicate directed arcs numbered {int(train.duplicate_arcs):,} in training and {int(test.duplicate_arcs):,} in student testing; self-loops numbered {int(train.self_loop_arcs):,} and {int(test.self_loop_arcs):,}, respectively. Multiple weak components occurred in {int(train.trips_with_multiple_weak_components)}/{int(train.trips)} training trips and {int(test.trips_with_multiple_weak_components)}/{int(test.trips)} student-test trips. The median largest-component share was {train.largest_component_fraction_median:.1%} for training and {test.largest_component_fraction_median:.1%} for student testing. Component counts include all valid-coordinate nodes; the largest-component share helps distinguish substantial fragmentation from small disconnected pieces.</p>
<div class="figure-box"><img src="data:image/png;base64,{graph_img}" alt="Road-network weak-component counts and largest component shares"><div class="caption">Weak-component counts and largest-component node share across trips. Component count uses a logarithmic axis.</div></div>
<p>The interactive Folium map for representative training trip <code>00000001</code> is saved at <code>results/eda/00000001_map.html</code>. It provides separate toggles for raw GPS observations, nearby road segments, and the reference route.</p>
{END_MARKER}"""


def create_appendix_pdf(rows: pd.DataFrame) -> bytes:
    quality_cols = [
        ("partition", "Partition"), ("trips", "Trips"),
        ("raw_gps_observations", "GPS rows"), ("missing_gps_rows", "Missing"),
        ("non_numeric_gps_rows", "Non-numeric"), ("nonfinite_gps_rows", "Non-finite"),
        ("invalid_coordinate_rows", "Invalid coord."), ("duplicate_gps_rows", "Duplicates"),
        ("nonmonotonic_timestamp_steps", "Time backsteps"), ("gaps_over_2s", "Gaps >2s"),
    ]
    network_cols = [
        ("partition", "Partition"), ("invalid_nodes", "Invalid nodes"),
        ("invalid_arcs", "Invalid arcs"), ("duplicate_arcs", "Duplicate arcs"),
        ("self_loop_arcs", "Self-loops"), ("isolated_nodes", "Isolated"),
        ("trips_with_multiple_weak_components", "Trips >1 comp."),
        ("weak_components_median_per_trip", "Median comp."),
        ("weak_components_max_per_trip", "Max comp."),
        ("largest_component_fraction_median", "Median largest share"),
    ]
    # Embed a TrueType font: this avoids viewer-dependent fallback glyphs in the
    # generated appendix and keeps the report legible after PDF merging.
    font_path = Path("/usr/share/fonts/liberation-sans-fonts/LiberationSans-Regular.ttf")
    bold_path = Path("/usr/share/fonts/liberation-sans-fonts/LiberationSans-Bold.ttf")
    pdfmetrics.registerFont(TTFont("EDA", str(font_path)))
    pdfmetrics.registerFont(TTFont("EDA-Bold", str(bold_path)))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="EDA_Title", parent=styles["Title"], fontName="EDA-Bold", fontSize=15, leading=17, spaceAfter=2))
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontName="EDA", fontSize=7.1, leading=8.4))
    styles.add(ParagraphStyle(name="Section", parent=styles["Heading2"], fontName="EDA-Bold", fontSize=9, leading=10, spaceBefore=1, spaceAfter=1))
    styles.add(ParagraphStyle(name="TableHead", parent=styles["BodyText"], fontName="EDA-Bold", fontSize=6.0,
                              leading=6.5, textColor=colors.white, alignment=1))
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=11*mm, leftMargin=11*mm,
                            topMargin=9*mm, bottomMargin=9*mm, title=PDF_MARKER)
    story = [
        Paragraph(PDF_MARKER, styles["EDA_Title"]),
        Paragraph("Raw train and student_test data only (80 and 10 trips); reserved evaluator-test data was not accessed. Connectivity uses weak components, treating directed arcs as undirected only for component membership.", styles["Small"]),
        Spacer(1, 1.5*mm),
        Paragraph("GPS data-quality checks", styles["Section"]),
    ]

    def make_table(cols):
        data = [[Paragraph(label, styles["TableHead"]) for _, label in cols]]
        for row in rows.to_dict(orient="records"):
            vals = []
            for key, _ in cols:
                value = row[key]
                if key == "largest_component_fraction_median":
                    vals.append(f"{value:.1%}")
                else:
                    vals.append(fmt(value))
            data.append(vals)
        widths = [doc.width / len(cols)] * len(cols)
        table = Table(data, colWidths=widths, repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#24466a")),
            ("GRID", (0,0), (-1,-1), 0.35, colors.HexColor("#b9c5d2")),
            ("FONTNAME", (0,1), (-1,-1), "EDA"), ("FONTSIZE", (0,1), (-1,-1), 6.5), ("ALIGN", (1,1), (-1,-1), "CENTER"),
            ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f1f5f9")]),
            ("LEFTPADDING", (0,0), (-1,-1), 3), ("RIGHTPADDING", (0,0), (-1,-1), 3),
            ("TOPPADDING", (0,0), (-1,-1), 2), ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ]))
        return table

    story.extend([make_table(quality_cols), Spacer(1, 1*mm)])
    train = rows.loc[rows.partition == "train"].iloc[0]
    test = rows.loc[rows.partition == "student_test"].iloc[0]
    story.append(Paragraph(
        f"Across {int(train.raw_gps_observations + test.raw_gps_observations):,} raw GPS rows, no missing, non-numeric, non-finite, out-of-range, duplicate, or non-increasing-timestamp rows were found. There were {int(train.gaps_over_2s + test.gaps_over_2s):,} time gaps longer than two seconds ({int(train.gaps_over_2s):,} train; {int(test.gaps_over_2s):,} student_test); these are sampling gaps, not missing rows.", styles["Small"]
    ))
    story.extend([Spacer(1, 1*mm), Paragraph("Road-network integrity and weak connectivity", styles["Section"]),
                  make_table(network_cols), Spacer(1, 1*mm)])
    story.append(Paragraph(
        f"No invalid-coordinate nodes, invalid arc endpoints, or isolated nodes were found. Duplicate directed arcs: {int(train.duplicate_arcs):,} train and {int(test.duplicate_arcs):,} student_test; self-loops: {int(train.self_loop_arcs)} and {int(test.self_loop_arcs)}. Multiple weak components occurred in {int(train.trips_with_multiple_weak_components)}/{int(train.trips)} train trips and {int(test.trips_with_multiple_weak_components)}/{int(test.trips)} student_test trips. Median largest-component node share: {train.largest_component_fraction_median:.1%} train and {test.largest_component_fraction_median:.1%} student_test.", styles["Small"]
    ))
    story.extend([Spacer(1, 1*mm), Paragraph(
        "Gaps over two seconds are sampling intervals, not missing records. Isolated nodes have no incident arcs. Duplicate arcs are repeated directed endpoint pairs.", styles["Small"]),
        Spacer(1, 1*mm)])
    # Keep both evidence plots together in a single landscape appendix page.
    left = Image(str(EDA_DIR / "gps_data_quality.png"), width=127*mm, height=63.5*mm)
    right = Image(str(EDA_DIR / "road_network_connectivity.png"), width=127*mm, height=57.15*mm)
    plots = Table([[left, right]], colWidths=[doc.width/2, doc.width/2])
    plots.setStyle(TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"), ("LEFTPADDING", (0,0), (-1,-1), 1),
                               ("RIGHTPADDING", (0,0), (-1,-1), 1), ("TOPPADDING", (0,0), (-1,-1), 0),
                               ("BOTTOMPADDING", (0,0), (-1,-1), 0)]))
    story.append(plots)
    doc.build(story)
    return buffer.getvalue()


def main() -> None:
    summary = pd.read_csv(EDA_DIR / "eda_partition_summary.csv")
    if set(summary["partition"]) != {"train", "student_test"}:
        raise ValueError("Expected exactly the train and student_test EDA summaries.")
    appendix_html = create_html_appendix(summary)
    document = HTML_REPORT.read_text(encoding="utf-8")
    if START_MARKER in document and END_MARKER in document:
        start = document.index(START_MARKER)
        end = document.index(END_MARKER, start) + len(END_MARKER)
        document = document[:start] + appendix_html + document[end:]
    else:
        if "</body>" not in document:
            raise ValueError("Report HTML has no closing body tag.")
        document = document.replace("</body>", appendix_html + "\n</body>", 1)
    HTML_REPORT.write_text(document, encoding="utf-8")

    appendix = PdfReader(io.BytesIO(create_appendix_pdf(summary)))
    existing = PdfReader(str(PDF_REPORT))
    source_pages = list(existing.pages)
    appendix_start = next((i for i, page in enumerate(source_pages)
                           if PDF_MARKER in (page.extract_text() or "")), None)
    if appendix_start is not None:
        source_pages = source_pages[:appendix_start]
    writer = PdfWriter()
    for page in source_pages:
        writer.add_page(page)
    for page in appendix.pages:
        writer.add_page(page)
    writer.add_metadata({"/Title": "PathSnap Technical Report with EDA Quality Appendix"})
    with PDF_REPORT.open("wb") as stream:
        writer.write(stream)
    print(f"Updated HTML: {HTML_REPORT}")
    print(f"Updated PDF: {PDF_REPORT} ({len(writer.pages)} pages)")


if __name__ == "__main__":
    main()
