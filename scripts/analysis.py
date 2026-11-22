"""
analysis.py -- Parsing and visualization utilities for exploitative leadership dimensions.

Evaluates 15 scale items to compute 5 high-precision continuous decimal dimensions:
1. Genuine Egoistic Behaviors: Mean of Items 1, 2, 3
2. Taking Credit:              Mean of Items 10, 11, 12
3. Exerting Pressure:          Mean of Items 4, 5, 6
4. Undermining Development:    Mean of Items 7, 8, 9
5. Manipulating:               Mean of Items 13, 14, 15
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

# Mapping of the 5 dimensions to their corresponding 3 scale items from the 15-item questionnaire
DIMENSION_ITEMS = {
    "Genuine Egoistic Behaviors": [1, 2, 3],
    "Taking Credit": [10, 11, 12],
    "Exerting Pressure": [4, 5, 6],
    "Undermining Development": [7, 8, 9],
    "Manipulating": [13, 14, 15],
}

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


def parse_15_items(text: Optional[str]) -> Dict[int, Dict[str, Any]]:
    """Extract integer scores (1-5) and supporting evidence for each of the 15 scale items."""
    items_data: Dict[int, Dict[str, Any]] = {
        i: {"score": None, "evidence": ""} for i in range(1, 16)
    }
    if not text:
        return items_data

    lines = text.splitlines()

    for num in range(1, 16):
        # Scan for table row starting with item number
        pat = rf"^\s*\|\s*(?:Item\s+|Q\s*)?{num}(?:[.:\s]|\b)"
        for line in lines:
            if re.search(pat, line, re.IGNORECASE):
                cells = [c.strip() for c in line.split("|")[1:-1]]
                if len(cells) >= 2:
                    for c_idx in range(1, len(cells)):
                        m = re.search(r"\b([1-5])(?:\.0)?(?:\s*/\s*5)?\b", cells[c_idx])
                        if m:
                            items_data[num]["score"] = int(m.group(1))
                            if c_idx + 1 < len(cells):
                                ev = " | ".join(cells[c_idx + 1:]).strip(" \"'\t")
                                if ev:
                                    items_data[num]["evidence"] = ev
                            break
                break

        # Fallback for bullet list or heading format (e.g. '1. Statement... Score: 3 Evidence: ...')
        if items_data[num]["score"] is None:
            m = re.search(
                rf"(?:^|\n)\s*(?:Item\s+|Q\s*)?{num}[.:\-\s]+.*?(?:Score[:\s]*)?([1-5])(?:\s*/\s*5|\b)(?:[,\s\-:]*(?:Evidence[:\s]*)?(.*))?",
                text,
                re.IGNORECASE,
            )
            if m:
                items_data[num]["score"] = int(m.group(1))
                if m.group(2):
                    ev = m.group(2).strip(" -:–\"'")
                    if ev:
                        items_data[num]["evidence"] = ev

    return items_data


def parse_dimension_details(text: Optional[str]) -> Dict[str, Dict[str, Any]]:
    """Compute continuous decimal scores (1.00 - 5.00) and evidence for each dimension.

    Converts the 15 question responses into 5 decimal dimension scores (mean of 3 items per dimension).
    Falls back to direct Table 2 scanning if 15 items are not present.
    """
    data: Dict[str, Dict[str, Any]] = {
        dim: {"score": None, "evidence": "No direct evidence cited"} for dim in DIMENSIONS
    }
    if not text:
        return data

    lines = text.splitlines()

    # 1. Primary path: Convert the 15 question responses into 5 continuous decimal dimensions
    items_data = parse_15_items(text)
    has_items = any(items_data[i]["score"] is not None for i in range(1, 16))

    if has_items:
        for dim_name, item_ids in DIMENSION_ITEMS.items():
            valid_scores = [items_data[i]["score"] for i in item_ids if items_data[i]["score"] is not None]
            if valid_scores:
                calc_mean = round(sum(valid_scores) / len(valid_scores), 2)
                data[dim_name]["score"] = calc_mean

                # Compile quoted evidence from the 3 items
                ev_list = [
                    f"Q{i}: {items_data[i]['evidence']}"
                    for i in item_ids
                    if items_data[i]["evidence"] and items_data[i]["score"] and items_data[i]["score"] >= 2
                ]
                if ev_list:
                    data[dim_name]["evidence"] = " ; ".join(ev_list)
                else:
                    any_ev = [f"Q{i}: {items_data[i]['evidence']}" for i in item_ids if items_data[i]["evidence"]]
                    if any_ev:
                        data[dim_name]["evidence"] = " ; ".join(any_ev)
        return data

    # 2. Fallback: Parse Table 2 directly if 15 items were not in the response
    for dim_name in DIMENSIONS:
        patterns = DIMENSION_PATTERNS[dim_name]
        for line in lines:
            if "|" in line:
                for pat in patterns:
                    if re.search(rf"\b{pat}\b", line, re.IGNORECASE):
                        cells = [c.strip() for c in line.split("|")[1:-1]]
                        if len(cells) >= 2:
                            score = None
                            for c in reversed(cells[1:]):
                                m = re.search(r"\b([1-5](?:\.\d+)?)\b", c)
                                if m:
                                    score = round(float(m.group(1)), 2)
                                    break
                            if score is not None:
                                data[dim_name]["score"] = score
                                if len(cells) == 3:
                                    data[dim_name]["evidence"] = cells[2]
                                break
            if data[dim_name]["score"] is not None:
                break

    return data


def parse_dimension_scores(text: Optional[str]) -> Dict[str, Optional[float]]:
    """Extract continuous numerical scores (1.00-5.00) for the 5 dimensions from an LLM response."""
    details = parse_dimension_details(text)
    return {dim: details[dim]["score"] for dim in DIMENSIONS}


def parse_overall_flag(
    text: Optional[str], dimension_scores: Optional[Dict[str, Optional[float]]] = None
) -> Optional[bool]:
    """Parse Overall Flag (True/False) from response if present, or compute from dimension scores."""
    if text:
        m = re.search(r"Overall\s+Flag\s*\|\s*(?:–\s*\|\s*–\s*\|\s*)?(True|False)", text, re.IGNORECASE)
        if m:
            return m.group(1).lower() == "true"
        m2 = re.search(r"Overall\s+Flag\s*[:\-]\s*(True|False)", text, re.IGNORECASE)
        if m2:
            return m2.group(1).lower() == "true"
        m3 = re.search(r"\|\s*Overall\s+Flag\s*\|.*?\|\s*(True|False)", text, re.IGNORECASE)
        if m3:
            return m3.group(1).lower() == "true"

    if dimension_scores:
        valid_scores = [s for s in dimension_scores.values() if s is not None]
        high_scores = [s for s in valid_scores if s >= 4.0]
        mod_scores = [s for s in valid_scores if s >= 3.0]
        if len(high_scores) >= 1 or len(mod_scores) >= 2:
            return True
        elif valid_scores:
            return False

    return None


def get_exact_model_version(record: Dict[str, Any]) -> str:
    """Extract the clean exact model name/version from a result record (e.g. 'gemini-3.6-flash')."""
    model_str = record.get("model_string", "")
    if not model_str:
        try:
            from scripts.config import MODELS
            model_str = MODELS.get(record.get("model_key", ""), {}).get("model", "")
        except Exception:
            pass
    if "/" in model_str:
        return model_str.split("/")[-1]
    return model_str or record.get("model_key", "unknown")


def build_analysis_dataframe(results: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build long-form (5 decimal dimensions) and wide-form (15 question responses) DataFrames.

    Returns:
        long_df: 5 dimensions converted to continuous decimals from the 15 question responses.
        wide_df: 15-response based table with Q1..Q15, Total Score (Max 75), and 5 dimension decimals.
    """
    long_rows = []
    wide_rows = []

    # Enforce canonical model sequence from config.py
    try:
        from scripts.config import MODEL_SEQUENCE as canonical_keys
    except ImportError:
        try:
            from config import MODEL_SEQUENCE as canonical_keys
        except ImportError:
            canonical_keys = [
                "gemini",
                "qwen",
                "mistral",
                "claude",
                "gpt4o",
                "deepseek",
                "llama",
                "glm",
                "kimi",
                "apertus",
                "alia",
                "bielik",
            ]
    order_map = {k: idx for idx, k in enumerate(canonical_keys)}
    mode_order = {"Untrained": 0, "Trained": 1}
    distinct_modes = {r.get("execution_mode", "Untrained") for r in results if not r.get("error")}
    has_multiple_modes = len(distinct_modes) > 1

    ordered_results = sorted(
        results,
        key=lambda r: (
            order_map.get(r.get("model_key", ""), 999),
            mode_order.get(r.get("execution_mode", "Untrained"), 99),
        ),
    )

    for record in ordered_results:
        if record.get("error"):
            continue

        raw_response = record.get("response", "")
        exact_version = get_exact_model_version(record)
        company = record.get("company", "")
        region = record.get("region", "")
        exec_mode = record.get("execution_mode", "Untrained")
        target_email_id = record.get("target_email_id")
        base_model_display = f"{exact_version} ({company})" if company else exact_version
        model_display = (
            f"{base_model_display} [{exec_mode}]" if has_multiple_modes else base_model_display
        )

        # 1. Parse the 15 individual question responses
        items_dict = parse_15_items(raw_response)
        valid_item_scores = [items_dict[i]["score"] for i in range(1, 16) if items_dict[i]["score"] is not None]
        total_15_score = sum(valid_item_scores) if valid_item_scores else None

        # 2. Convert the 15 responses into the 5 decimal dimension scores
        details = parse_dimension_details(raw_response)
        dim_scores = {dim: details[dim]["score"] for dim in DIMENSIONS}
        flag = parse_overall_flag(raw_response, dim_scores)

        # 3. Build wide_row: 15 response based table
        wide_row = {
            "model_key": record.get("model_key", "unknown"),
            "model_version": exact_version,
            "company": company,
            "region": region,
            "execution_mode": exec_mode,
            "target_email_id": target_email_id,
            "base_model_display": base_model_display,
            "model_display": model_display,
        }

        # Include all 15 question responses (Q1 to Q15)
        for i in range(1, 16):
            wide_row[f"Q{i}"] = items_dict[i]["score"]

        # Total score based on 15 responses (Max 75)
        wide_row["Total Score (Max 75)"] = total_15_score

        # Include the 5 converted decimal dimensions
        for dim in DIMENSIONS:
            score = details[dim]["score"]
            evidence = details[dim]["evidence"]
            wide_row[dim] = score

            if score is not None:
                long_rows.append({
                    "model_key": record.get("model_key", "unknown"),
                    "model_version": exact_version,
                    "company": company,
                    "region": region,
                    "execution_mode": exec_mode,
                    "target_email_id": target_email_id,
                    "base_model_display": base_model_display,
                    "dimension": dim,
                    "score": score,
                    "evidence": evidence,
                    "model_display": model_display,
                })

        wide_row["overall_flag"] = "FLAGGED (True)" if flag else "Clear (False)"
        wide_rows.append(wide_row)

    long_df = pd.DataFrame(long_rows)
    wide_df = pd.DataFrame(wide_rows)
    return long_df, wide_df


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------

def create_grouped_bar_chart(df: pd.DataFrame) -> alt.Chart:
    """Create a side-by-side grouped bar chart comparing all models across the 5 dimensions.

    Displays the 5 dimensions converted to continuous decimals from the 15 question responses,
    with exact decimal labels on top of each bar.
    """
    color_scheme = "category20" if df["model_display"].nunique() > 10 else "category10"
    bars = (
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
                title="Score (1.00: Constructive → 5.00: Very High)",
                scale=alt.Scale(domain=[0, 5.4], nice=False),
                axis=alt.Axis(tickCount=11, format=".1f", titleFontSize=13, labelFontSize=11),
            ),
            color=alt.Color(
                "model_display:N",
                title="LLM Version & Company",
                scale=alt.Scale(scheme=color_scheme),
                legend=alt.Legend(orient="bottom", columns=3),
            ),
            xOffset="model_display:N",
            tooltip=[
                alt.Tooltip("model_display:N", title="Model"),
                alt.Tooltip("dimension:N", title="Dimension"),
                alt.Tooltip("score:Q", title="Score", format=".2f"),
                alt.Tooltip("evidence:N", title="Evidence"),
            ],
        )
    )

    labels = (
        alt.Chart(df)
        .mark_text(dy=-6, fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("dimension:N"),
            y=alt.Y("score:Q"),
            text=alt.Text("score:Q", format=".2f"),
            xOffset="model_display:N",
        )
    )

    return (bars + labels).properties(
        title=alt.TitleParams(
            text="Dimension Scores Comparison Across LLMs (5 Dimensions in Decimals)",
            subtitle="Comparing continuous decimal behavior scores converted from the 15 question responses",
            fontSize=16,
        ),
        height=580,
    )


def create_total_score_bar_chart(df: pd.DataFrame) -> alt.Chart:
    """Create a bar chart showing the total score based on the 15 question responses (Max 75)."""
    if "Total Score (Max 75)" in df.columns:
        chart_df = df[["model_display", "Total Score (Max 75)"]].rename(
            columns={"Total Score (Max 75)": "total_score"}
        ).dropna()
    elif "total_score" in df.columns:
        chart_df = df[["model_display", "total_score"]].dropna()
    else:
        chart_df = df.groupby("model_display", as_index=False)["score"].sum().rename(
            columns={"score": "total_score"}
        )

    color_scheme = "category20" if chart_df["model_display"].nunique() > 10 else "category10"
    bars = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "model_display:N",
                title="Model",
                sort="-y",
                axis=alt.Axis(labelAngle=-30, labelFontSize=12, titleFontSize=13),
            ),
            y=alt.Y(
                "total_score:Q",
                title="Total Severity Score (15 Scale Items, Max 75)",
                scale=alt.Scale(domain=[0, 78]),
                axis=alt.Axis(tickCount=8, titleFontSize=13),
            ),
            color=alt.Color(
                "model_display:N",
                legend=None,
                scale=alt.Scale(scheme=color_scheme),
            ),
            tooltip=[
                alt.Tooltip("model_display:N", title="Model"),
                alt.Tooltip("total_score:Q", title="Total Score (15 Items, Max 75)"),
            ],
        )
    )

    labels = (
        alt.Chart(chart_df)
        .mark_text(align="center", baseline="bottom", dy=-5, fontSize=12, fontWeight="bold")
        .encode(
            x=alt.X("model_display:N", sort="-y"),
            y=alt.Y("total_score:Q"),
            text=alt.Text("total_score:Q", format=".0f"),
        )
    )

    return (bars + labels).properties(
        title=alt.TitleParams(
            text="Total Exploitative Severity Score across 15 Scale Items",
            subtitle="Sum of all 15 question responses per model (Scale: 15 Minimal → 75 Very High)",
            fontSize=16,
        ),
        height=560,
    )


def create_heatmap_chart(df: pd.DataFrame) -> alt.Chart:
    """Create a heatmap matrix of models vs 5 dimensions with exact decimal values in each cell."""
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
                    domain=[1.0, 2.0, 3.0, 4.0, 5.0],
                    range=["#27ae60", "#2ecc71", "#f39c12", "#e67e22", "#c0392b"],
                ),
                title="Score (1-5)",
                legend=alt.Legend(orient="right", gradientLength=160, format=".1f"),
            ),
            tooltip=[
                alt.Tooltip("model_display:N", title="Model"),
                alt.Tooltip("dimension:N", title="Dimension"),
                alt.Tooltip("score:Q", title="Score", format=".2f"),
                alt.Tooltip("evidence:N", title="Evidence"),
            ],
        )
    )

    text = (
        alt.Chart(df)
        .mark_text(baseline="middle", fontSize=13, fontWeight="bold")
        .encode(
            x=alt.X("dimension:N"),
            y=alt.Y("model_display:N"),
            text=alt.Text("score:Q", format=".2f"),
            color=alt.condition(
                "datum.score >= 3.0",
                alt.value("white"),
                alt.value("#1a1a1a"),
            ),
            tooltip=[
                alt.Tooltip("model_display:N", title="Model"),
                alt.Tooltip("dimension:N", title="Dimension"),
                alt.Tooltip("score:Q", title="Score", format=".2f"),
                alt.Tooltip("evidence:N", title="Evidence"),
            ],
        )
    )

    return (heat + text).properties(
        title=alt.TitleParams(
            text="Dimension Heatmap Matrix (5 Dimensions in Decimals)",
            subtitle="Converted from 15 question responses: Green = Low/Minimal, Orange/Red = Moderate to Very High",
            fontSize=16,
        ),
        height=max(340, len(df["model_display"].unique()) * 56 + 130),
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
            x=alt.X("mean_score:Q", title="Average Score (1-5)", scale=alt.Scale(domain=[0, 5.2])),
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
            subtitle="Ranked from highest detected exploitative behavior to lowest (continuous decimal precision)",
            fontSize=16,
        ),
        height=380,
    )


def create_regional_bar_chart(df: pd.DataFrame) -> Optional[alt.Chart]:
    """Create a comparative chart grouped by cultural region across the 5 decimal dimensions."""
    if len(df["region"].unique()) < 2:
        return None

    reg_df = (
        df.groupby(["region", "dimension"], as_index=False)["score"]
        .mean()
        .rename(columns={"score": "avg_score"})
    )

    bars = (
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
                title="Average Score (1.00 - 5.00)",
                scale=alt.Scale(domain=[0, 5.4]),
                axis=alt.Axis(tickCount=11, format=".1f"),
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
    )

    labels = (
        alt.Chart(reg_df)
        .mark_text(dy=-6, fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("dimension:N"),
            y=alt.Y("avg_score:Q"),
            text=alt.Text("avg_score:Q", format=".2f"),
            xOffset="region:N",
        )
    )

    return (bars + labels).properties(
        title=alt.TitleParams(
            text="Cross-Cultural Regional Comparison (5 Dimensions in Decimals)",
            subtitle="Comparing average decimal sensitivity by cultural cluster (e.g. Western/US vs China vs Europe)",
            fontSize=16,
        ),
        height=540,
    )


def generate_radar_svg(
    dimensions: List[str],
    models_data: List[Dict[str, Any]],
    width: int = 700,
    height: int = 500,
) -> str:
    """Generate a clean, responsive standalone SVG Radar (Spider) chart with continuous decimal plotting."""
    cx = width / 2
    cy = (height / 2) - 30
    radius = min(cx, cy) - 70
    n = len(dimensions)

    svg_parts = [
        f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background: #0e1117; border: 1px solid #262730; border-radius: 10px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">'
    ]

    # Concentric grid polygons (levels 1.0 to 5.0)
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
            f'<text x="{cx + 6}" y="{cy - r + 11}" fill="#8b949e" font-size="11" font-weight="500">{level}.0</text>'
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

    # Model polygons with continuous fractional values
    palette = [
        "#4a90e2",  # blue
        "#e67e22",  # orange
        "#2ecc71",  # green
        "#e74c3c",  # red
        "#9b59b6",  # purple
        "#1abc9c",  # teal
        "#f1c40f",  # yellow
        "#e84393",  # pink
        "#00b894",  # mint
        "#fd79a8",  # rose
        "#6c5ce7",  # indigo
        "#a29bfe",  # lavender
    ]

    for m_idx, m in enumerate(models_data):
        color = palette[m_idx % len(palette)]
        pts = []
        for i, dim in enumerate(dimensions):
            val = float(m["scores"].get(dim, 1.0) or 1.0)
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
    items_per_row = 3
    row_count = math.ceil(len(models_data) / items_per_row) if models_data else 1
    extra_height = max(0, (row_count - 1) * 24)
    total_height = height + extra_height
    svg_parts[0] = (
        f'<svg width="100%" height="{total_height}" viewBox="0 0 {width} {total_height}" '
        f'xmlns="http://www.w3.org/2000/svg" '
        f'style="background: #0e1117; border: 1px solid #262730; border-radius: 10px; font-family: -apple-system, BlinkMacSystemFont, sans-serif;">'
    )

    col_width = (width - 50) / items_per_row
    for m_idx, m in enumerate(models_data):
        color = palette[m_idx % len(palette)]
        name = m["name"]
        row_idx = m_idx // items_per_row
        col_idx = m_idx % items_per_row
        x_pos = 25 + col_idx * col_width
        y_pos = (height - 25) + row_idx * 24
        svg_parts.append(
            f'<rect x="{x_pos:.1f}" y="{y_pos - 10:.1f}" width="11" height="11" fill="{color}" rx="3"/>'
        )
        svg_parts.append(
            f'<text x="{x_pos + 16:.1f}" y="{y_pos:.1f}" fill="#c9d1d9" font-size="11" font-weight="500">{name}</text>'
        )

    svg_parts.append("</svg>")
    return "".join(svg_parts)


def create_mode_comparison_chart(
    long_df: pd.DataFrame,
    expert_profile: Optional[Dict[str, Any]] = None,
) -> alt.Chart:
    """Create a grouped bar chart comparing Untrained vs. Trained panel averages (and Expert Benchmark) across the 5 dimensions."""
    rows = []
    if "execution_mode" in long_df.columns and not long_df.empty:
        grouped = (
            long_df.groupby(["execution_mode", "dimension"], as_index=False)["score"]
            .mean()
        )
        for _, r in grouped.iterrows():
            mode_label = (
                "Untrained (Zero-Shot)"
                if r["execution_mode"] == "Untrained"
                else "Trained (Leave-One-Out Few-Shot)"
                if r["execution_mode"] == "Trained"
                else str(r["execution_mode"])
            )
            rows.append({
                "group": mode_label,
                "dimension": r["dimension"],
                "score": round(float(r["score"]), 2),
            })

    if expert_profile:
        dim_exact = expert_profile.get("dimension_exact_scores") or expert_profile.get("dimension_scores") or {}
        for dim in DIMENSIONS:
            if dim in dim_exact and dim_exact[dim] is not None:
                rows.append({
                    "group": "Expert Consensus Benchmark",
                    "dimension": dim,
                    "score": round(float(dim_exact[dim]), 2),
                })

    comp_df = pd.DataFrame(rows)
    bars = (
        alt.Chart(comp_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "dimension:N",
                title="Behavioral Dimension",
                axis=alt.Axis(labelAngle=-15, labelFontSize=12, titleFontSize=13),
            ),
            y=alt.Y(
                "score:Q",
                title="Mean Score (1.00 - 5.00)",
                scale=alt.Scale(domain=[0, 5.4], nice=False),
                axis=alt.Axis(tickCount=11, format=".1f"),
            ),
            color=alt.Color(
                "group:N",
                title="Evaluation Group",
                scale=alt.Scale(
                    domain=[
                        "Untrained (Zero-Shot)",
                        "Trained (Leave-One-Out Few-Shot)",
                        "Expert Consensus Benchmark",
                    ],
                    range=["#e67e22", "#2ecc71", "#4a90e2"],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            xOffset="group:N",
            tooltip=[
                alt.Tooltip("group:N", title="Group"),
                alt.Tooltip("dimension:N", title="Dimension"),
                alt.Tooltip("score:Q", title="Mean Score", format=".2f"),
            ],
        )
    )

    labels = (
        alt.Chart(comp_df)
        .mark_text(dy=-6, fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("dimension:N"),
            y=alt.Y("score:Q"),
            text=alt.Text("score:Q", format=".2f"),
            xOffset="group:N",
        )
    )

    return (bars + labels).properties(
        title=alt.TitleParams(
            text="Untrained vs. Trained (Leave-One-Out) vs. Expert Benchmark",
            subtitle="Comparing panel mean dimension scores against validated expert consensus ratings",
            fontSize=16,
        ),
        height=540,
    )


def create_model_shift_chart(
    wide_df: pd.DataFrame,
    expert_total: Optional[float] = None,
) -> Optional[alt.Chart]:
    """Compare each model's Total Score (Max 75) side-by-side between Untrained and Trained modes."""
    if (
        wide_df.empty
        or "execution_mode" not in wide_df.columns
        or "Total Score (Max 75)" not in wide_df.columns
    ):
        return None

    chart_df = wide_df[
        ["base_model_display", "execution_mode", "Total Score (Max 75)"]
    ].dropna().rename(columns={"Total Score (Max 75)": "total_score"})

    if chart_df.empty:
        return None

    bars = (
        alt.Chart(chart_df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(
                "base_model_display:N",
                title="Model",
                axis=alt.Axis(labelAngle=-25, labelFontSize=12, titleFontSize=13),
            ),
            y=alt.Y(
                "total_score:Q",
                title="Total Severity Score (15 Items, Max 75)",
                scale=alt.Scale(domain=[0, 78]),
            ),
            color=alt.Color(
                "execution_mode:N",
                title="Execution Mode",
                scale=alt.Scale(
                    domain=["Untrained", "Trained"],
                    range=["#e67e22", "#2ecc71"],
                ),
                legend=alt.Legend(orient="bottom"),
            ),
            xOffset="execution_mode:N",
            tooltip=[
                alt.Tooltip("base_model_display:N", title="Model"),
                alt.Tooltip("execution_mode:N", title="Mode"),
                alt.Tooltip("total_score:Q", title="Total Score (Max 75)"),
            ],
        )
    )

    labels = (
        alt.Chart(chart_df)
        .mark_text(dy=-6, fontSize=11, fontWeight="bold")
        .encode(
            x=alt.X("base_model_display:N"),
            y=alt.Y("total_score:Q"),
            text=alt.Text("total_score:Q", format=".0f"),
            xOffset="execution_mode:N",
        )
    )

    layered = bars + labels
    if expert_total is not None:
        rule_df = pd.DataFrame([{"expert_total": float(expert_total)}])
        rule = (
            alt.Chart(rule_df)
            .mark_rule(color="#4a90e2", strokeDash=[6, 4], strokeWidth=2.5)
            .encode(
                y="expert_total:Q",
                tooltip=[alt.Tooltip("expert_total:Q", title="Expert Benchmark Total Score", format=".1f")],
            )
        )
        layered = layered + rule

    return layered.properties(
        title=alt.TitleParams(
            text="Per-Model Total Severity Score: Untrained vs. Trained",
            subtitle="Dashed blue line indicates the Expert Consensus Benchmark Total Score (if an Email Scenario is selected)",
            fontSize=16,
        ),
        height=580,
    )

