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
    value="You are evaluating workplace leadership scenarios. Judge the behavior described "
    "and explain your reasoning clearly and concisely.",
    height=80,
)

prompt = st.text_area(
    "Scenario / prompt",
    placeholder="e.g. A manager tells their team: 'I need everyone here until this ships "
    "tonight, no exceptions.' Is this leadership behavior exploitative? Explain your reasoning.",
    height=140,
)

run_clicked = st.button("Run", type="primary", disabled=not (prompt and selected_models))

if not selected_models:
    st.warning("Select at least one model in the sidebar.")

# Keep results across reruns so they don't vanish when you tweak a widget
if "results" not in st.session_state:
    st.session_state.results = []

if run_clicked:
    st.session_state.results = []
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
    st.subheader("Results")

    cols = st.columns(2)
    for i, record in enumerate(st.session_state.results):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{record['model_key']}** · {record['company']} · _{record['region']}_")
                if record["error"]:
                    st.error(record["error"])
                else:
                    st.write(record["response"])

    st.divider()
    df = pd.DataFrame(st.session_state.results)
    st.dataframe(df[["model_key", "region", "company", "response", "error"]], use_container_width=True)

    jsonl_data = "\n".join(json.dumps(r, ensure_ascii=False) for r in st.session_state.results)
    st.download_button(
        "Download results (JSONL)",
        data=jsonl_data,
        file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        mime="application/json",
    )
