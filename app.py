"""
app.py -- web interface for the cross-cultural LLM comparison tool.

Run locally:
    streamlit run app.py

Deploy live (free): push this repo to GitHub, then connect it at
https://streamlit.io/cloud -- add your API keys under the app's
"Secrets" panel there (same variable names as .env.example).
"""

import json
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import MODELS  # noqa: E402
from dispatch import run_dispatch  # noqa: E402
from analysis import (  # noqa: E402
    DIMENSIONS,
    build_analysis_dataframe,
    create_average_dimension_chart,
    create_grouped_bar_chart,
    create_heatmap_chart,
    create_regional_bar_chart,
    generate_radar_svg,
    parse_dimension_scores,
    parse_overall_flag,
)

st.set_page_config(page_title="Cross-Cultural LLM Comparison", layout="wide")

st.title("Cross-cultural LLM comparison")
st.caption(
    "Send one prompt to multiple LLMs and compare how each interprets it. "
    "Built for the exploitative-leadership cross-cultural bias study."
)

# ---------------------------------------------------------------- sidebar --
st.sidebar.header("Models")

regions = {}
for key, entry in MODELS.items():
    regions.setdefault(entry["region"], []).append(key)

selected_models = []
for region, keys in regions.items():
    st.sidebar.markdown(f"**{region}**")
    for key in keys:
        entry = MODELS[key]
        label = f"{key} — {entry['company']}"
        if st.sidebar.checkbox(label, value=True, key=f"chk_{key}"):
            selected_models.append(key)

st.sidebar.divider()
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3, 0.1, help="Fixed across all models for a fair comparison.")
max_tokens = st.sidebar.slider("Max response length (tokens)", 100, 1500, 500, 50)

# ------------------------------------------------------------------ main --
system_prompt = st.text_area(
    "System prompt (applied identically to every model)",
    value="""Exploitative Leadership Training Coach (ELTC)
                Role & Purpose
                You are an Exploitative Leadership Training Coach (ELTC) — an expert communication analyst trained to detect and improve exploitative leadership tone in workplace emails.
                Your role is to:

                Identify harmful or exploitative leadership behaviors in written communication
                Provide structured, evidence-based feedback 
                Transform messages into constructive, ethical, and development-focused communication
                Core Task

                Read the full email carefully
                Evaluate across five behavioral dimensions:
                1. Genuine Egoistic Behaviors

                Self-serving language 
                Status-seeking or self-promotion 
                Centering personal recognition over team success 
                2. Taking Credit

                Claiming ownership of others’ work 
                Minimizing or omitting team contributions 
                Reframing group success as individual success 
                3. Exerting Pressure

                Coercion, threats, or intimidation 
                Unrealistic deadlines framed aggressively 
                Language implying consequences for non-compliance 
                4. Undermining Development

                Blocking growth opportunities 
                Dismissing learning, mentorship, or skill-building 
                Discouraging initiative or independent thinking 
                5. Manipulating

                Guilt-tripping or emotional pressure 
                Gaslighting or reframing reality unfairly 
                Selective or deceptive phrasing to influence behavior 
                Output Format
                1. Table (Always Required)
                Always evaluate and display all 5 dimensions in a Markdown table:
                | Dimension | Present (Yes/No) | Score (1–5 + emoji) | Evidence (≤30 words) |
                
                Scoring System:
                5 :red_circle: Very High (Explicit exploitative tone)
                4 :large_orange_circle: High (Clear exploitative behavior)
                3 :large_yellow_circle: Moderate (Probable exploitative tone)
                2 :large_green_circle: Low (Minor or ambiguous signals)
                1 :white_circle: Minimal / Constructive (Neutral or ethical tone)
                
                Rules:
                - Present = Yes if score ≥ 3, otherwise No
                - Always list all 5 behavioral dimensions
                - Always include the Overall Flag row at the bottom of the table:
                | Overall Flag | True/False | – | – |
                
                Overall Flag = True if:
                Any dimension ≥ 4 OR two or more dimensions ≥ 3. Otherwise False.
                
                Evidence Guidelines:
                • Quote or paraphrase relevant words (≤30 words)
                • For scores ≥ 2, highlight risky phrases using bold brackets: “[sample phrase]”
                • For score 1, state: “Constructive and supportive phrasing” or cite positive words
                
                2. Notes Section (Always Required)
                After the table, provide structured commentary:
                • Key behaviors analyzed (highlight positive supportive elements or exploitative indicators)
                • Likely impact on team morale, trust, or psychological safety
                • General leadership communication guidance
                
                3. Suggested Rewrite
                • If Overall Flag is True or issues are detected (any score ≥ 3): Provide a full rewrite in a constructive, empathetic leadership tone.
                • If no issues are found (all scores ≤ 2 and Overall Flag is False): State “No rewrite needed — message already uses a constructive tone.” followed by a brief 1-2 sentence breakdown of what makes the phrasing effective.
                
                Output Order:
                1. Table (with Overall Flag)
                2. Notes
                3. Suggested Rewrite""",
    height=300,
)

prompt = st.text_area(
    "Scenario / prompt",
    placeholder="e.g. A manager tells their team: 'I need everyone here until this ships "
    "tonight, no exceptions.' Is this leadership behavior exploitative? Explain your reasoning.",
    height=200,
)

run_clicked = st.button("Run", type="primary", disabled=not (prompt and selected_models))

if not selected_models:
    st.warning("Select at least one model in the sidebar.")

# Keep results across reruns so they don't vanish when you tweak a widget
if "results" not in st.session_state:
    st.session_state.results = []

if run_clicked:
    st.session_state.results = []
    st.session_state.show_graphs = False
    progress_area = st.empty()
    results_container = st.container()
    done = 0

    def on_result(record):
        global done
        done += 1
        progress_area.progress(done / len(selected_models), text=f"{done}/{len(selected_models)} models responded")
        st.session_state.results.append(record)

    with st.spinner("Querying models..."):
        records, missing = run_dispatch(
            prompt=prompt,
            system_prompt=system_prompt,
            model_keys=selected_models,
            temperature=temperature,
            max_tokens=max_tokens,
            progress_callback=on_result,
        )
    progress_area.empty()

    if missing:
        st.warning(
            f"Skipped (missing API key in secrets/.env): {', '.join(missing)}. "
            "Add the relevant key to run these too."
        )

# --------------------------------------------------------------- results --
if st.session_state.results:
    st.divider()

    # Results header with global Expand / Collapse toggle
    res_hdr_left, res_hdr_right = st.columns([3, 1])
    with res_hdr_left:
        st.subheader("Results")
    with res_hdr_right:
        expand_all = st.toggle("Expand all responses", value=False, help="Toggle to show or hide all model responses at once.")

    cols = st.columns(2)
    for i, record in enumerate(st.session_state.results):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{record['model_key']}** · {record['company']} · _{record['region']}_")
                if record.get("error"):
                    st.error(record["error"])
                else:
                    # Parse dimension scores for quick card preview badges
                    scores = parse_dimension_scores(record.get("response"))
                    flag = parse_overall_flag(record.get("response"))

                    badges = []
                    if flag is True:
                        badges.append("🚩 :red[**FLAGGED**]")
                    elif flag is False:
                        badges.append("✅ :green[**CLEAR**]")

                    # Highlight any dimensions with high score (>= 3)
                    high_dims = [f"{d.split()[0]}: {s}/5" for d, s in scores.items() if s and s >= 3]
                    if high_dims:
                        badges.append(f"⚠️ {', '.join(high_dims)}")

                    if badges:
                        st.caption(" · ".join(badges))

                    # Individual On-Click Show / Hide toggle per LLM response
                    with st.expander("📄 Show / Hide Response", expanded=expand_all):
                        st.markdown(record["response"])

    st.divider()
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df[["model_key", "region", "company", "response", "error"]], use_container_width=True)

    jsonl_data = "\n".join(json.dumps(r, ensure_ascii=False) for r in st.session_state.results)

    # Action buttons: Download JSONL and new Graphical Analysis button side-by-side
    btn_col1, btn_col2, _ = st.columns([1.2, 1.5, 2])
    with btn_col1:
        st.download_button(
            "Download results (JSONL)",
            data=jsonl_data,
            file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
            mime="application/json",
            use_container_width=True,
        )
    with btn_col2:
        graphs_open = st.session_state.get("show_graphs", False)
        if st.button(
            "📊 Hide Graphical Work" if graphs_open else "📊 Generate Graphical Analysis",
            type="primary" if not graphs_open else "secondary",
            use_container_width=True,
        ):
            st.session_state.show_graphs = not graphs_open
            st.rerun()

    # ------------------------------------------------ Graphical Work Section --
    if st.session_state.get("show_graphs", False):
        st.divider()
        st.subheader("📊 5-Dimension Behavioral Analysis & Graphs")
        st.caption(
            "Visualizing LLM scores across the 5 exploitative leadership dimensions: "
            "1. Genuine Egoistic Behaviors | 2. Taking Credit | 3. Exerting Pressure | "
            "4. Undermining Development | 5. Manipulating (Scale: 1 Minimal → 5 Very High)"
        )

        long_df, wide_df = build_analysis_dataframe(st.session_state.results)

        if long_df.empty:
            st.warning(
                "Could not extract dimension scores from the current model responses. "
                "Ensure responses contain the 5-dimension evaluation table or numerical scores."
            )
        else:
            # Summary Metrics Row
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            with m_col1:
                st.metric("Models Evaluated", len(wide_df))
            with m_col2:
                avg_score = long_df["score"].mean()
                st.metric("Overall Avg Score", f"{avg_score:.2f} / 5.0")
            with m_col3:
                dim_means = long_df.groupby("dimension")["score"].mean()
                top_dim = dim_means.idxmax()
                st.metric("Top Flagged Dimension", top_dim, f"{dim_means.max():.2f} / 5")
            with m_col4:
                flagged_count = (wide_df["overall_flag"].str.contains("FLAGGED")).sum()
                st.metric("Models Flagging Exploitative Tone", f"{flagged_count} / {len(wide_df)}")

            # Model filter selection
            available_models = sorted(long_df["model_display"].unique().tolist())
            selected_chart_models = st.multiselect(
                "Filter models in graphs:",
                options=available_models,
                default=available_models,
                help="Select which models to compare in the graphs below.",
            )

            filtered_long_df = long_df[long_df["model_display"].isin(selected_chart_models)]

            if filtered_long_df.empty:
                st.info("Select at least one model above to display the graphs.")
            else:
                # Multiple specialized graph representations
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "📊 Side-by-Side Comparison",
                    "🗺️ Dimension Heatmap",
                    "🕸️ Radar (Spider) Profile",
                    "🌍 Cultural Regional Comparison",
                    "📋 Dimension Scores Table",
                ])

                with tab1:
                    st.altair_chart(create_grouped_bar_chart(filtered_long_df), use_container_width=True)
                    st.altair_chart(create_average_dimension_chart(filtered_long_df), use_container_width=True)

                with tab2:
                    st.altair_chart(create_heatmap_chart(filtered_long_df), use_container_width=True)

                with tab3:
                    st.markdown("##### Multi-Dimensional Behavioral Radar Profile")
                    st.caption("Each vertex represents one of the 5 behavioral dimensions (scale 1 to 5).")
                    models_radar_data = []
                    for model_name in selected_chart_models:
                        sub = filtered_long_df[filtered_long_df["model_display"] == model_name]
                        scores_dict = dict(zip(sub["dimension"], sub["score"]))
                        models_radar_data.append({"name": model_name, "scores": scores_dict})
                    radar_svg = generate_radar_svg(DIMENSIONS, models_radar_data)
                    st.markdown(radar_svg, unsafe_allow_html=True)

                with tab4:
                    reg_chart = create_regional_bar_chart(filtered_long_df)
                    if reg_chart is not None:
                        st.altair_chart(reg_chart, use_container_width=True)
                    else:
                        st.info("Cultural regional comparison requires responses from models across at least two different cultural regions (e.g., Western/US vs China vs Europe).")

                with tab5:
                    st.markdown("##### Extracted Dimension Scores by Model")
                    st.dataframe(wide_df, use_container_width=True)
                    csv_data = wide_df.to_csv(index=False)
                    st.download_button(
                        "Download Extracted Scores (CSV)",
                        data=csv_data,
                        file_name=f"dimension_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                    )

