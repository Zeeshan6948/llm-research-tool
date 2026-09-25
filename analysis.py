"""
analysis.py -- Parsing and visualization utilities for exploitative leadership dimensions.

Evaluates 5 behavioral dimensions:
1. Genuine Egoistic Behaviors
2. Taking Credit
3. Exerting Pressure
4. Undermining Development
5. Manipulating
"""

import math
import re
from typing import Any, Dict, List, Optional, Tuple

import altair as alt
import pandas as pd

# The 5 canonical dimensions
DIMENSIONS = [
    "Genuine Egoistic Behaviors",
    "Taking Credit",
    "Exerting Pressure",
    "Undermining Development",
    "Manipulating",
]

# Patterns for robust matching across various LLM formatting styles
DIMENSION_PATTERNS = {
    "Genuine Egoistic Behaviors": [
        r"genuine egoistic",
        r"egoistic behavior",
        r"egoistic",
        r"egoism",
    ],
    "Taking Credit": [
        r"taking credit",
        r"take credit",
        r"credit[- ]taking",
        r"credit",
    ],
    "Exerting Pressure": [
        r"exerting pressure",
        r"exert pressure",
        r"pressure",
        r"coercion",
    ],
    "Undermining Development": [
        r"undermining development",
        r"undermine development",
        r"undermining",
        r"development blocking",
    ],
    "Manipulating": [
        r"manipulating",
        r"manipulation",
        r"manipulative",
    ],
}


def parse_dimension_details(text: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Extract numerical scores (1-5) and supporting evidence for each dimension.

    Handles markdown tables, bullet lists, bold headings, and various phrasing styles.
    """
    data: Dict[str, Dict[str, Any]] = {
        dim: {"score": None, "evidence": "No direct evidence cited"} for dim in DIMENSIONS
    }
    if not text:
        return data

    lines = text.splitlines()

    # Step 1: Scan for Markdown table rows
    for dim_name in DIMENSIONS:
        patterns = DIMENSION_PATTERNS[dim_name]
        for line in lines:
            if "|" in line:
                for pat in patterns:
                    if re.search(rf"\b{pat}\b", line, re.IGNORECASE):
                        cells = [c.strip() for c in line.split("|")[1:-1]]
                        dim_idx = -1
                        for idx, cell in enumerate(cells):
                            if re.search(rf"\b{pat}\b", cell, re.IGNORECASE):
                                dim_idx = idx
                                break
                        if dim_idx != -1:
                            score = None
                            score_idx = -1
                            for idx in range(dim_idx + 1, len(cells)):
                                m = re.search(r"\b([1-5])\b", cells[idx])
                                if m:
                                    score = int(m.group(1))
                                    score_idx = idx
                                    break
                            evidence = ""
                            if score_idx != -1 and score_idx + 1 < len(cells):
                                raw_ev = " | ".join(cells[score_idx + 1:]).strip()
                                raw_ev = raw_ev.strip("\"' \t")
                                evidence = raw_ev
                            if score is not None:
                                data[dim_name]["score"] = score
                                if evidence:
                                    data[dim_name]["evidence"] = evidence
                                break
            if data[dim_name]["score"] is not None:
                break

    # Step 2: Fallback scan for lists or heading lines (e.g., '1. Genuine Egoistic Behaviors: 3/5' or 'Score: 4')
    for dim_name in DIMENSIONS:
        if data[dim_name]["score"] is None:
            patterns = DIMENSION_PATTERNS[dim_name]
            for pat in patterns:
                m = re.search(
                    rf"(?:{pat})[^\n]*?[:\-\–]\s*.*?([1-5])(?:\s*/\s*5|\b)([^\n]*)",
                    text,
                    re.IGNORECASE,
                )
                if m:
                    data[dim_name]["score"] = int(m.group(1))
                    ev_text = m.group(2).strip(" -:–\"'")
                    if ev_text:
                        data[dim_name]["evidence"] = ev_text
                    break

    return data


def parse_dimension_scores(text: Optional[str]) -> Dict[str, Optional[int]]:
    """Extract numerical scores (1-5) for the 5 dimensions from an LLM response."""
    details = parse_dimension_details(text)
    return {dim: details[dim]["score"] for dim in DIMENSIONS}


def parse_overall_flag(text: Optional[str]) -> Optional[bool]:
    """Parse Overall Flag (True/False) from response if present."""
    if not text:
        return None
    m = re.search(r"Overall\s+Flag\s*\|\s*(True|False)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower() == "true"
    m2 = re.search(r"Overall\s+Flag\s*[:\-]\s*(True|False)", text, re.IGNORECASE)
    if m2:
        return m2.group(1).lower() == "true"
    return None


def build_analysis_dataframe(results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build long-form and wide-form DataFrames of parsed dimension scores.

    Returns:
        long_df: DataFrame with columns [model_key, company, region, dimension, score, evidence, model_display]
        wide_df: DataFrame with models as rows and dimensions + overall flag as columns
    """
    long_rows = []
    wide_rows = []

    for record in results:
        if record.get("error"):
            continue
        details = parse_dimension_details(record.get("response"))
        flag = parse_overall_flag(record.get("response"))

        scores = {dim: details[dim]["score"] for dim in DIMENSIONS}
        valid_scores = [s for s in scores.values() if s is not None]
        high_scores = [s for s in valid_scores if s >= 4]
        mod_scores = [s for s in valid_scores if s >= 3]
        computed_flag = bool(len(high_scores) >= 1 or len(mod_scores) >= 2)
        final_flag = flag if flag is not None else computed_flag

        wide_row = {
            "model_key": record.get("model_key", "unknown"),
            "company": record.get("company", ""),
            "region": record.get("region", ""),
            "overall_flag": "FLAGGED (True)" if final_flag else "Clear (False)",
        }

        for dim in DIMENSIONS:
            score = details[dim]["score"]
            evidence = details[dim]["evidence"]
            wide_row[dim] = score
            if score is not None:
                long_rows.append({
                    "model_key": record.get("model_key", "unknown"),
                    "company": record.get("company", ""),
                    "region": record.get("region", ""),
                    "dimension": dim,
                    "score": score,
                    "evidence": evidence,
                    "model_display": f"{record.get('model_key')} ({record.get('company')})",
                })

        wide_rows.append(wide_row)

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)
    return long_df, wide_df


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def create_grouped_bar_chart(df: pd.DataFrame) -> alt.Chart:
    """Create a grouped bar chart comparing all models across the 5 dimensions.

    On hover, only the model and evidence are displayed.
    """
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "dimension:N",
                title="Behavioral Dimension",
                axis=alt.Axis(labelAngle=-15, labelFontSize=12, titleFontSize=13),
            ),
            y=alt.Y(
                "score:Q",
                title="Score (1: Constructive → 5: Very High)",
                scale=alt.Scale(domain=[0, 5], nice=False),
                axis=alt.Axis(tickMinStep=1, titleFontSize=13),
            ),
            color=alt.Color(
                "model_display:N",
                title="Model & Company",
                scale=alt.Scale(scheme="category10"),
                legend=alt.Legend(orient="bottom", columns=3),
            ),
            xOffset="model_display:N",
            tooltip=[
                alt.Tooltip("model_key:N", title="Model"),
                alt.Tooltip("evidence:N", title="Evidence"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                text="Dimension Scores Comparison Across LLMs",
                subtitle="Comparing behavior scores (1 = Minimal / Ethical, 5 = Very High Exploitative Tone)",
                fontSize=16,
            ),
            height=420,
        )
    )
    return chart


def create_heatmap_chart(df: pd.DataFrame) -> alt.Chart:
    """Create a heatmap matrix of models vs dimensions.

    On hover, only the model and evidence are displayed.
    """
    heat = (
        alt.Chart(df)
        .mark_rect(stroke="white", strokeWidth=1.5)
        .encode(
            x=alt.X(
                "dimension:N",
                title="Behavioral Dimension",
                axis=alt.Axis(labelAngle=-15, labelFontSize=12),
            ),
            y=alt.Y("model_display:N", title="Model", axis=alt.Axis(labelFontSize=12)),
            color=alt.Color(
                "score:Q",
                scale=alt.Scale(
                    domain=[1, 2, 3, 4, 5],
                    range=["#27ae60", "#2ecc71", "#f39c12", "#e67e22", "#c0392b"],
                ),
                title="Score (1-5)",
                legend=alt.Legend(orient="right", gradientLength=160),
            ),
            tooltip=[
                alt.Tooltip("model_key:N", title="Model"),
                alt.Tooltip("evidence:N", title="Evidence"),
            ],
        )
    )

    text = (
        alt.Chart(df)
        .mark_text(baseline="middle", fontSize=14, fontWeight="bold")
        .encode(
            x=alt.X("dimension:N"),
            y=alt.Y("model_display:N"),
            text=alt.Text("score:Q"),
            color=alt.condition(
                "datum.score >= 3",
                alt.value("white"),
                alt.value("#1a1a1a"),
            ),
            tooltip=[
                alt.Tooltip("model_key:N", title="Model"),
                alt.Tooltip("evidence:N", title="Evidence"),
            ],
        )
    )

    return (heat + text).properties(
        title=alt.TitleParams(
            text="Dimension Heatmap Matrix (Severity by Model)",
            subtitle="Green = Low/Minimal, Orange/Red = Moderate to Very High exploitative tone",
            fontSize=16,
        ),
        height=max(220, len(df["model_display"].unique()) * 45 + 100),
    )


def create_average_dimension_chart(df: pd.DataFrame) -> alt.Chart:
    """Create a bar chart showing the consensus / average score per dimension across all models."""
    avg_df = (
        df.groupby("dimension", as_index=False)["score"]
        .agg(mean_score="mean", count="count")
        .sort_values(by="mean_score", ascending=False)
    )

    bars = (
        alt.Chart(avg_df)
        .mark_bar(cornerRadiusTopRight=5, cornerRadiusBottomRight=5)
        .encode(
            y=alt.Y("dimension:N", sort="-x", title=None, axis=alt.Axis(labelFontSize=12)),
            x=alt.X("mean_score:Q", title="Average Score (1-5)", scale=alt.Scale(domain=[0, 5])),
            color=alt.Color(
                "mean_score:Q",
                scale=alt.Scale(
                    domain=[1, 3, 5],
                    range=["#27ae60", "#f39c12", "#e74c3c"],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("dimension:N", title="Dimension"),
                alt.Tooltip("mean_score:Q", title="Mean Score", format=".2f"),
                alt.Tooltip("count:Q", title="Models Count"),
            ],
        )
    )

    labels = (
        alt.Chart(avg_df)
        .mark_text(align="left", baseline="middle", dx=6, fontSize=12, fontWeight="bold")
        .encode(
            y=alt.Y("dimension:N", sort="-x"),
            x=alt.X("mean_score:Q"),
            text=alt.Text("mean_score:Q", format=".2f"),
        )
    )

    return (bars + labels).properties(
        title=alt.TitleParams(
            text="Average Dimension Severity (Panel Consensus)",
            subtitle="Ranked from highest detected exploitative behavior to lowest",
            fontSize=16,
        ),
        height=260,
    )


def create_regional_bar_chart(df: pd.DataFrame) -> Optional[alt.Chart]:
    """Create a comparative chart grouped by cultural region if multiple regions exist."""
    if len(df["region"].unique()) < 2:
        return None

    reg_df = (
        df.groupby(["region", "dimension"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "avg_score"})
    )

    chart = (
        alt.Chart(reg_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "dimension:N",
                title="Behavioral Dimension",
                axis=alt.Axis(labelAngle=-15, labelFontSize=12),
            ),
            y=alt.Y(
                "avg_score:Q",
                title="Average Score (1-5)",
                scale=alt.Scale(domain=[0, 5]),
            ),
            color=alt.Color(
                "region:N",
                title="Cultural Region",
                scale=alt.Scale(scheme="tableau10"),
                legend=alt.Legend(orient="bottom"),
            ),
            xOffset="region:N",
            tooltip=[
                alt.Tooltip("region:N", title="Region"),
                alt.Tooltip("dimension:N", title="Dimension"),
                alt.Tooltip("avg_score:Q", title="Avg Score", format=".2f"),
            ],
        )
        .properties(
            title=alt.TitleParams(
                text="Cross-Cultural Regional Comparison",
                subtitle="Comparing average dimension sensitivity by cultural cluster (e.g. Western/US vs China vs Europe)",
                fontSize=16,
            ),
            height=380,
        )
    )
    return chart


def generate_radar_svg(
    dimensions: List[str],
    models_data: List[Dict[str, Any]],
    width: int = 700,
    height: int = 500,
) -> str:
    """Generate a clean, responsive standalone SVG Radar (Spider) chart.

    Works 100% offline, requires no external dependencies.
    """
    cx = width / 2
    cy = (height / 2) - 30
    radius = min(cx, cy) - 70
    n = len(dimensions)

    svg_parts = [
        f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background: #0e1117; border: 1px solid #262730; border-radius: 10px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">'
    ]

    # Concentric grid polygons (levels 1 to 5)
    for level in range(1, 6):
        r = radius * (level / 5.0)
        pts = []
        for i in range(n):
            angle = (2 * math.pi / n) * i - math.pi / 2
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            pts.append(f"{x:.1f},{y:.1f}")
        pts_str = " ".join(pts)
        svg_parts.append(
            f'<polygon points="{pts_str}" fill="none" stroke="#2c303c" stroke-width="1" stroke-dasharray="3,3"/>'
        )
        svg_parts.append(
            f'<text x="{cx + 6}" y="{cy - r + 11}" fill="#8b949e" font-size="11" font-weight="500">{level}</text>'
        )

    # Spokes and dimension labels
    for i, dim in enumerate(dimensions):
        angle = (2 * math.pi / n) * i - math.pi / 2
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        svg_parts.append(f'<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="#3b4252" stroke-width="1.2"/>')

        lx = cx + (radius + 28) * math.cos(angle)
        ly = cy + (radius + 20) * math.sin(angle)
        anchor = "middle"
        if math.cos(angle) > 0.2:
            anchor = "start"
        elif math.cos(angle) < -0.2:
            anchor = "end"
        svg_parts.append(
            f'<text x="{lx:.1f}" y="{ly:.1f}" fill="#f0f6fc" font-size="12" font-weight="600" text-anchor="{anchor}">{dim}</text>'
        )

    # Model polygons
    palette = [
        "#4a90e2",  # blue
        "#e67e22",  # orange
        "#2ecc71",  # green
        "#e74c3c",  # red
        "#9b59b6",  # purple
        "#1abc9c",  # teal
        "#f1c40f",  # yellow
        "#e84393",  # pink
    ]

    for m_idx, m in enumerate(models_data):
        color = palette[m_idx % len(palette)]
        pts = []
        for i, dim in enumerate(dimensions):
            val = m["scores"].get(dim, 1) or 1
            r = radius * (val / 5.0)
            angle = (2 * math.pi / n) * i - math.pi / 2
            x = cx + r * math.cos(angle)
            y = cy + r * math.sin(angle)
            pts.append(f"{x:.1f},{y:.1f}")
            svg_parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        pts_str = " ".join(pts)
        svg_parts.append(
            f'<polygon points="{pts_str}" fill="{color}" fill-opacity="0.22" stroke="{color}" stroke-width="2.5"/>'
        )

    # Legend at bottom
    leg_y = height - 25
    x_offset = 35
    for m_idx, m in enumerate(models_data):
        color = palette[m_idx % len(palette)]
        name = m["name"]
        svg_parts.append(
            f'<rect x="{x_offset}" y="{leg_y - 11}" width="12" height="12" fill="{color}" rx="3"/>'
        )
        svg_parts.append(
            f'<text x="{x_offset + 18}" y="{leg_y}" fill="#c9d1d9" font-size="12" font-weight="500">{name}</text>'
        )
        x_offset += len(name) * 8 + 35

    svg_parts.append("</svg>")
    return "".join(svg_parts)
