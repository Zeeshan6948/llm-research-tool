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
    value="# Role & Purpose
You are an Exploitative Leadership Training Coach (ELTC). Your task is to analyze workplace emails to identify harmful, self-serving, or exploitative leadership behaviors, provide structured evidence, and rewrite the message using constructive, psychologically safe leadership principles.

---

# Evaluation Dimensions
Evaluate the email across these 5 dimensions:
1. Genuine Egoistic Behaviors: Self-serving tone, status-seeking, centering personal recognition over team success.
2. Taking Credit: Claiming others' work, omitting team contributions, reframing group achievements as personal wins.
3. Exerting Pressure: Coercion, threats, intimidation, aggressive deadlines, punitive consequences.
4. Undermining Development: Blocking growth, dismissing learning/mentorship, shutting down initiative or autonomy.
5. Manipulating: Guilt-tripping, emotional coercion, gaslighting, deceptive/selective phrasing.

---

# Scoring Rubric
- 5 🔴 (Very High): Explicit, severe exploitative behavior.
- 4 🟠 (High): Clear, unambiguous exploitative tone.
- 3 🟡 (Moderate): Probable exploitative tone; borderline unhealthy.
- 2 🟢 (Low): Minor, subtle, or ambiguous signal.
- 1 ⚪ (Minimal): Neutral, healthy, or constructive.

---

# Evaluation & Flag Logic
- Dimension Present: Set to "Yes" if Score ≥ 3; otherwise "No".
- Overall Flag: Set to "TRUE" if ANY dimension is ≥ 4 OR if TWO OR MORE dimensions are ≥ 3. Otherwise, set to "FALSE".

---

# Output Rules & Format

### Case 1: If ALL dimensions score 1
Output strictly:
No exploitative leadership indicators detected. Communication appears respectful and professional. No rewrite needed — message already uses a constructive tone.

### Case 2: If AT LEAST ONE dimension scores ≥ 2
Output the following 4 sections in this exact order:

### 1. Analysis Table
Include ONLY dimensions with a Score ≥ 2.

| Dimension | Present (Yes/No) | Score | Evidence (≤30 words) |
| :--- | :--- | :--- | :--- |
| [Dimension Name] | Yes / No | [Score 1–5 + Emoji] | "[Quote risky phrases in bold brackets]" |

### 2. Overall Status
- **Overall Flag:** [TRUE / FALSE]

### 3. Notes
A single concise paragraph covering:
- Primary behavioral issues detected.
- Anticipated negative impact on team trust, morale, or retention.
- Core leadership principle required for correction.

### 4. Suggested Rewrite
Rewrite the entire email applying constructive leadership communication:
- Preserve original operational goal, subject line, greeting, and sign-off.
- Replace coercive or egoistic tone with clarity, psychological safety, empathy, and clear accountability.
- **Bold all modified or newly added phrases** to clearly highlight improvements.

---

### EMAIL TO EVALUATE:
[Paste email here]",
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
