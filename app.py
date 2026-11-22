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

import importlib
import os
import sys
from pathlib import Path

# Add scripts directory to sys.path so modules can be imported directly or via scripts package
_scripts_dir = str(Path(__file__).parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

import scripts.config as _scripts_config  # noqa: E402
import scripts.analysis as _scripts_analysis  # noqa: E402

# Ensure Streamlit reruns always pick up updates to config.py / analysis.py without requiring a full restart
importlib.reload(_scripts_config)
importlib.reload(_scripts_analysis)

from scripts.config import MODEL_SEQUENCE, MODELS  # noqa: E402
from scripts.dispatch import run_dispatch, run_dispatch_both  # noqa: E402
from scripts.benchmark_data import (  # noqa: E402
    assemble_system_prompt,
    detect_email_id_from_text,
    load_benchmark_dataset,
)
from scripts.analysis import (  # noqa: E402
    DIMENSIONS,
    build_analysis_dataframe,
    create_average_dimension_chart,
    create_grouped_bar_chart,
    create_heatmap_chart,
    create_mode_comparison_chart,
    create_model_shift_chart,
    create_regional_bar_chart,
    create_total_score_bar_chart,
    get_exact_model_version,
    parse_dimension_scores,
    parse_overall_flag,
)

LOGO_PATH = "logo.png"

# Ensure Streamlit runtime theme options use the positive blue palette even before server restart
try:
    st._config.set_option("theme.base", "light")
    st._config.set_option("theme.primaryColor", "#2563EB")
    st._config.set_option("theme.backgroundColor", "#F8FAFC")
    st._config.set_option("theme.secondaryBackgroundColor", "#EFF6FF")
    st._config.set_option("theme.textColor", "#0F172A")
except Exception:
    pass

st.set_page_config(
    page_title="Cross-Cultural LLM Comparison",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Hide Streamlit Deploy button, top-right three dots menu, and footer, plus Blue Positive Vibe Theme
st.markdown(
    """
    <style>
    /* Positive Blue Theme Accents */
    :root {
        --primary-color: #2563EB !important;
    }
    .stApp {
        background: linear-gradient(180deg, #F0F7FF 0%, #F8FAFC 35%, #FFFFFF 100%);
    }
    section[data-testid="stSidebar"] {
        background-color: #EFF6FF !important;
        border-right: 1px solid #BFDBFE !important;
    }
    h1, h2, h3 {
        color: #1E3A8A !important;
    }
    /* Primary buttons in vibrant positive blue */
    button[kind="primary"], [data-testid="baseButton-primary"], [data-testid="stBaseButton-primary"] {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
        color: #FFFFFF !important;
        box-shadow: 0 2px 6px rgba(37, 99, 235, 0.25) !important;
    }
    button[kind="primary"]:hover, [data-testid="baseButton-primary"]:hover, [data-testid="stBaseButton-primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #1D4ED8 !important;
    }
    /* Model Selection Checkboxes (Checked State -> Blue instead of Red) */
    div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:has(input:checked) > span:first-child,
    label[data-baseweb="checkbox"]:has(input:checked) > span:first-child,
    label[data-baseweb="checkbox"] input:checked ~ span:first-of-type {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
    }
    div[data-testid="stCheckbox"] label[data-baseweb="checkbox"]:hover > span:first-child {
        border-color: #2563EB !important;
    }
    /* Execution Mode Radio Buttons (Selected State -> Blue instead of Red) */
    div[data-testid="stRadio"] label[data-baseweb="radio"]:has(input:checked) > div:first-child,
    label[data-baseweb="radio"]:has(input:checked) > div:first-child {
        background-color: #2563EB !important;
        border-color: #2563EB !important;
    }
    div[data-testid="stRadio"] label[data-baseweb="radio"]:hover > div:first-child {
        border-color: #2563EB !important;
    }
    /* Filter Models in Graphs Multiselect Tags / Pills (Blue instead of Red) */
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"],
    span[data-baseweb="tag"] {
        background-color: #2563EB !important;
        border-color: #1D4ED8 !important;
        color: #FFFFFF !important;
        border-radius: 6px !important;
    }
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] span,
    div[data-testid="stMultiSelect"] span[data-baseweb="tag"] svg,
    span[data-baseweb="tag"] span,
    span[data-baseweb="tag"] svg {
        color: #FFFFFF !important;
        fill: #FFFFFF !important;
    }
    /* Focus borders on inputs, textareas, and multiselect dropdowns (Blue instead of Red) */
    div[data-baseweb="select"] > div:focus-within,
    div[data-baseweb="textarea"]:focus-within,
    div[data-baseweb="base-input"]:focus-within,
    div[data-baseweb="input"]:focus-within {
        border-color: #2563EB !important;
        box-shadow: 0 0 0 1px #2563EB !important;
    }
    /* Active tab underline & text in blue */
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #2563EB !important;
        border-bottom-color: #2563EB !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="tab-highlight"] {
        background-color: #2563EB !important;
    }
    /* Progress bar in blue */
    div[data-testid="stProgress"] [role="progressbar"] > div,
    div[data-testid="stProgress"] > div > div > div > div {
        background-color: #2563EB !important;
    }
    /* Metric cards subtle blue frame */
    div[data-testid="stMetric"] {
        background-color: #F0F7FF;
        border: 1px solid #DBEAFE;
        border-radius: 10px;
        padding: 12px 16px;
    }
    /* Enlarge Streamlit logo beyond native limits */
    [data-testid="stLogo"] {
        height: 6rem !important;
        max-height: 100% !important;
        width: auto !important;
    }
    [data-testid="stLogo"] img {
        height: 6rem !important;
        max-height: 100% !important;
        width: auto !important;
    }
    /* Hide Streamlit Deploy button */
    .stDeployButton, [data-testid="stDeployButton"] {
        display: none !important;
    }
    /* Hide top-right three dots menu (hamburger) */
    #MainMenu, [data-testid="stMainMenu"], [data-testid="stToolbarActions"] {
        visibility: hidden !important;
        display: none !important;
    }
    /* Hide footer */
    footer {
        visibility: hidden !important;
    }
    /* Full-screen opaque backdrop for the Purpose of Study modal so nothing behind it is visible */
    div[data-testid="stDialog"],
    .stDialog,
    div[data-testid="stModal"] {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 50%, #BFDBFE 100%) !important;
        background-color: #EFF6FF !important;
        opacity: 1 !important;
        z-index: 999999 !important;
    }
    div[data-testid="stDialog"] > div:first-child,
    .stDialog > div:first-child,
    div[data-testid="stModal"] > div:first-child {
        background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 50%, #BFDBFE 100%) !important;
        background-color: #EFF6FF !important;
        opacity: 1 !important;
    }
    div[data-testid="stDialog"] [role="dialog"],
    .stDialog [role="dialog"] {
        background-color: #FFFFFF !important;
        border: 1px solid #BFDBFE !important;
        box-shadow: 0 20px 50px rgba(30, 58, 138, 0.18) !important;
    }
    /* Completely hide background app content & sidebar while the dialog modal is open */
    body:has(div[data-testid="stDialog"]) section[data-testid="stSidebar"],
    body:has(div[data-testid="stDialog"]) [data-testid="stMainBlockContainer"],
    body:has(div[data-testid="stDialog"]) header[data-testid="stHeader"] {
        visibility: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.dialog("🎓 Purpose of Study & How to Use", width="large")
def show_study_intro():
    st.markdown(
        """
        ### 🌍 Cross-Cultural Exploitative Leadership Bias Study
        This research platform investigates how Large Language Models (LLMs) developed across different cultural and geopolitical regions (**Western/US**, **China**, and **Europe**) perceive, evaluate, and score **exploitative leadership behaviors** in workplace communication, and how **in-context expert calibration (Leave-One-Out training)** aligns model evaluations with human expert consensus.

        All models evaluate workplace communication using a standardized **15-item psychometric scale (Q1–Q15, Likert 1–5)** grouped into **5 core behavioral dimensions**:
        1. **Genuine Egoistic Behaviors** *(Q1–Q3)* — Prioritizing personal goals and self-benefit over employee needs.
        2. **Exerting Pressure** *(Q4–Q6)* — Coercion, unreasonable workload increases, or aggressive demands.
        3. **Undermining Development** *(Q7–Q9)* — Assigning routine/tedious work and blocking professional growth.
        4. **Taking Credit** *(Q10–Q12)* — Claiming recognition or ownership of subordinates' work.
        5. **Manipulating** *(Q13–Q15)* — Deception, emotional guilt-tripping, or playing colleagues against each other.

        ---

        ### 🚀 How to Use This Application
        1. **Choose an Execution Mode (Left Sidebar, under the Logo):**
           - **Baseline LLMs:** Evaluates the message using only the baseline rubric without prior expert benchmark examples.
           - **Expert Augmented LLMs:** Calibrates the LLMs by injecting validated expert benchmark ratings into the system prompt using a **Leave-One-Out** strategy (when `Email N` is selected, expert ratings from the other **8 scenarios** are injected into the system prompt while `Email N` is evaluated).
           - **Both Above:** Runs both **Baseline LLMs** and **Expert Augmented LLMs** side-by-side to measure how expert calibration shifts each model toward the human expert benchmark.
        2. **Select LLMs (Left Sidebar):** Choose which models and cultural clusters (**Western/US**, **China**, **Europe**) you want to compare.
        3. **Select a Scenario or Write a Custom Message:**
           - In **Expert Augmented LLMs** or **Both Above** mode, click **`Email 1`–`Email 9`** to lock in one of the 9 benchmark workplace emails, or click **`✏️ Custom`** (or leave unselected) to write your own custom message in the **"Write Your message to be evaluted by LLMs"** box.
        4. **Run & Explore the Results Hierarchy:**
           - **🌍 1) Cultural Comparison:** Compare average dimension scores across cultural regions (**Western/US**, **China**, **Europe**).
           - **📈 2) Overall Comparison:** Compare each model's total exploitative severity score across all 15 scale items (out of 75).
           - **📊 3) Dimension Comparison:** Inspect per-model 5-dimension bar charts, panel consensus rankings, and the **🗺️ Dimension Heatmap** matrix inside the same tab.
           - **📋 4) Dimension Scores Table:** View and download (CSV) the full table of Q1–Q15 item scores, total scores, and 5-dimension decimal scores.
           - **⚖️ Untrained vs. Trained (Leave-One-Out) vs. Expert Benchmark** *(shown in **Both Above** mode)*: Compare Baseline vs. Expert Augmented scores directly against the validated **Expert Consensus Benchmark**.

        *Click the **✕** button in the top-right corner of this window (or the button below) to open the application.*
        """
    )
    if st.button("Start Exploring the Application", type="primary", use_container_width=True):
        st.rerun()


if "intro_shown" not in st.session_state:
    st.session_state.intro_shown = True
    show_study_intro()

# Display application logo
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=220)

title_col, info_col = st.columns([5, 1], vertical_alignment="center")
with title_col:
    st.title("Cross-cultural LLM comparison")
with info_col:
    if st.button("ℹ️ Purpose & Guide", use_container_width=True):
        show_study_intro()

st.caption(
    "Send one prompt to multiple LLMs and compare how each interprets it. "
    "Built for the exploitative-leadership cross-cultural bias study."
)

@st.cache_data(show_spinner=False)
def get_cached_benchmark_dataset():
    """Load and cache the 9 emails and 15-item expert benchmark evaluations from data/."""
    return load_benchmark_dataset()


benchmark_dataset = get_cached_benchmark_dataset()

# ---------------------------------------------------------------- sidebar --
st.sidebar.subheader("Execution Mode")

MODE_UNTRAINED = "Baseline LLMs"
MODE_TRAINED = "Expert Augmented LLMs"
MODE_BOTH = "Both Above"

execution_mode = st.sidebar.radio(
    "Evaluation strategy:",
    options=[MODE_UNTRAINED, MODE_TRAINED, MODE_BOTH],
    index=0,
    key="execution_mode_radio",
    help=(
        "• Baseline LLMs: Baseline prompt without prior expert examples.\n"
        "• Expert Augmented LLMs: Leave-One-Out injection of the 8 other workplace scenarios' "
        "validated 15-item expert ratings into the system prompt.\n"
        "• Both Above: Runs Baseline LLMs and Expert Augmented LLMs side-by-side for direct comparison."
    ),
)

st.sidebar.divider()
st.sidebar.header("Models")

regions = {}
for key, entry in MODELS.items():
    regions.setdefault(entry["region"], []).append(key)

selected_models = []
for region, keys in regions.items():
    st.sidebar.markdown(f"**{region}**")
    for key in keys:
        entry = MODELS[key]
        exact_model = entry["model"].split("/")[-1]
        label = f"{exact_model} — {entry['company']}"
        if st.sidebar.checkbox(label, value=True, key=f"chk_{key}"):
            selected_models.append(key)

st.sidebar.divider()
temperature = 0.3
max_tokens = 4096

# ------------------------------------------------------------------ main --
if "selected_email_id" not in st.session_state:
    st.session_state.selected_email_id = None
if "scenario_prompt_input" not in st.session_state:
    st.session_state.scenario_prompt_input = ""

is_trained_or_both = execution_mode in (MODE_TRAINED, MODE_BOTH)

system_prompt = st.text_area(
    "System prompt (applied identically to every model)",
    value="""[SYSTEM INSTRUCTION]
You are an Exploitative Leadership Training Coach (ELTC). Your job is to score workplace communication strictly using a Markdown table.

CRITICAL FORMAT CONSTRAINT:
- Your response must begin IMMEDIATELY with `| Question | Answer | Reason |`.
- Do NOT output greetings, preamble, explanations, disclaimers, or notes before or after the table.
- Do NOT use code blocks (```) or numbered text lists.
- You MUST output the table header row, the alignment separator row (`| :---: | :---: | :--- |`), and ALL 15 numbered question rows (`| 1 |` through `| 15 |`).
- Do NOT skip or merge any rows. Every question from 1 to 15 must have its own row.
- Do NOT calculate means, sums, averages, or flags.

EVALUATION SCALE (Answer column must contain ONLY a single integer from 1 to 5):
5 = Very High (Explicit/severe evidence)
4 = High (Clear direct evidence)
3 = Moderate (Meaningful indirect concern)
2 = Low (Weak/ambiguous)
1 = Minimal (No evidence / neutral / constructive)

REASON COLUMN RULES:
- Maximum 30 words per row, single line, escape any pipe symbol as \|.
- Scores 2–5: Include an exact short quote from the email in bold (e.g., **"exact quote"**) with brief context.
- Score 1: State "No supporting evidence in the email."

EVALUATION ITEMS (You MUST score all 15 items in rows 1 to 15):
1. Leader assumes employee's work can be used for personal benefit.
2. Leader views employee as a means to achieve personal goals.
3. Leader values own goals over employee needs.
4. Leader puts employee under pressure to achieve personal goals.
5. Leader increases workload without considering employee needs.
6. Leader disregards current workload when assigning tasks.
7. Leader assigns tedious tasks for personal benefit.
8. Leader blocks professional development for own goals.
9. Leader assigns routine/monotonous tasks for personal benefit.
10. Leader uses employee's work for personal recognition.
11. Leader presents employee's work as their own.
12. Leader exploits employee's work for personal gain.
13. Leader plays employee against colleagues.
14. Leader manipulates employee to achieve personal goals.
15. Leader deceives employee to achieve personal goals.

MANDATORY OUTPUT FORMAT (17 lines total: 1 header + 1 separator + 15 numbered rows):
| Question | Answer | Reason |
| :---: | :---: | :--- |
| 1 | <integer 1-5> | <Reason ≤30 words> |
| 2 | <integer 1-5> | <Reason ≤30 words> |
| 3 | <integer 1-5> | <Reason ≤30 words> |
| 4 | <integer 1-5> | <Reason ≤30 words> |
| 5 | <integer 1-5> | <Reason ≤30 words> |
| 6 | <integer 1-5> | <Reason ≤30 words> |
| 7 | <integer 1-5> | <Reason ≤30 words> |
| 8 | <integer 1-5> | <Reason ≤30 words> |
| 9 | <integer 1-5> | <Reason ≤30 words> |
| 10 | <integer 1-5> | <Reason ≤30 words> |
| 11 | <integer 1-5> | <Reason ≤30 words> |
| 12 | <integer 1-5> | <Reason ≤30 words> |
| 13 | <integer 1-5> | <Reason ≤30 words> |
| 14 | <integer 1-5> | <Reason ≤30 words> |
| 15 | <integer 1-5> | <Reason ≤30 words> |

[EMAIL TO EVALUATE]
""",
    height=300,
)

# -------------------------------------------- Email Scenario Selector Bar --
if is_trained_or_both:
    st.markdown("##### 📧 Select Target Email Scenario (Leave-One-Out Few-Shot Evaluation)")
    st.caption(
        "By default, you can write your own custom message below. "
        "Or click any Email 1–9 button to lock in that scenario and use the remaining 8 emails' expert benchmark ratings for training."
    )
    email_cols = st.columns(10)
    for idx, eid in enumerate(range(1, 10)):
        is_active = st.session_state.selected_email_id == eid
        btn_type = "primary" if is_active else "secondary"
        with email_cols[idx]:
            if st.button(
                f"Email {eid}",
                key=f"email_btn_{eid}",
                type=btn_type,
                use_container_width=True,
                help=f"Select Email {eid}: {benchmark_dataset.get(eid, {}).get('subject', '')}",
            ):
                if st.session_state.selected_email_id == eid:
                    # Clicking the already active email deselects it and makes the box editable again
                    st.session_state.selected_email_id = None
                    st.session_state.scenario_prompt_input = ""
                else:
                    st.session_state.selected_email_id = eid
                    st.session_state.scenario_prompt_input = benchmark_dataset[eid]["email_text"]
                st.rerun()

    with email_cols[9]:
        if st.button(
            "✏️ Custom",
            key="email_btn_custom",
            type="primary" if st.session_state.selected_email_id is None else "secondary",
            use_container_width=True,
            help="Clear selected email and write a custom editable message.",
        ):
            st.session_state.selected_email_id = None
            st.session_state.scenario_prompt_input = ""
            st.rerun()

    active_eid = st.session_state.selected_email_id
    if active_eid is not None and active_eid in benchmark_dataset:
        active_info = benchmark_dataset.get(active_eid, {})
        loo_ids = [i for i in range(1, 10) if i != active_eid]
        st.info(
            f"**Active Target:** `Email {active_eid}` (*{active_info.get('subject', '')}*) — *Message box locked to scenario text*  |  "
            f"**In-Context Reference Expert Ratings (8 Scenarios):** `{', '.join(f'Email {i}' for i in loo_ids)}`"
        )
    else:
        st.info(
            "**Custom Message Mode (Editable):** Write your own message below, or select one of `Email 1`–`Email 9` above."
        )

is_prompt_locked = is_trained_or_both and (st.session_state.selected_email_id is not None)

prompt = st.text_area(
    "Write Your message to be evaluted by LLMs",
    key="scenario_prompt_input",
    placeholder="e.g. A manager tells their team: 'I need everyone here until this ships "
    "tonight, no exceptions.' Is this leadership behavior exploitative? Explain your reasoning.",
    height=200,
    disabled=is_prompt_locked,
)

run_clicked = st.button("Run", type="primary", disabled=not (prompt and selected_models))

if not selected_models:
    st.warning("Select at least one model in the sidebar.")

# Keep results across reruns so they don't vanish when you tweak a widget
if "results" not in st.session_state:
    st.session_state.results = []
if "last_run_mode" not in st.session_state:
    st.session_state.last_run_mode = MODE_UNTRAINED
if "last_target_email_id" not in st.session_state:
    st.session_state.last_target_email_id = None

if run_clicked:
    st.session_state.results = []
    st.session_state.show_graphs = False
    st.session_state.last_run_mode = execution_mode

    # Only associate a target email ID in Trained or Both mode (never in Untrained mode)
    if is_trained_or_both:
        detected_eid = detect_email_id_from_text(prompt, benchmark_dataset)
        target_eid = st.session_state.selected_email_id if is_prompt_locked else detected_eid
    else:
        target_eid = None
    st.session_state.last_target_email_id = target_eid

    progress_area = st.empty()
    results_container = st.container()
    done = 0

    if execution_mode == MODE_UNTRAINED:
        total_tasks = len(selected_models)

        def on_result_untrained(record):
            global done
            done += 1
            progress_area.progress(
                min(done / max(total_tasks, 1), 1.0),
                text=f"{done}/{total_tasks} models responded (Untrained Zero-Shot)",
            )
            st.session_state.results.append(record)

        with st.spinner("Querying models (Untrained Zero-Shot)..."):
            records, missing = run_dispatch(
                prompt=prompt,
                system_prompt=system_prompt,
                model_keys=selected_models,
                temperature=temperature,
                max_tokens=max_tokens,
                progress_callback=on_result_untrained,
                execution_mode="Untrained",
                target_email_id=None,
            )

    elif execution_mode == MODE_TRAINED:
        total_tasks = len(selected_models)
        trained_sys_prompt = assemble_system_prompt(
            base_system_prompt=system_prompt,
            target_email_id=target_eid,
            trained=True,
            dataset=benchmark_dataset,
        )
        target_label = f"Email {target_eid}" if target_eid else "Custom Message"

        def on_result_trained(record):
            global done
            done += 1
            progress_area.progress(
                min(done / max(total_tasks, 1), 1.0),
                text=f"{done}/{total_tasks} models responded (Trained on {target_label})",
            )
            st.session_state.results.append(record)

        with st.spinner(f"Querying models with Expert Benchmark Ratings (Target: {target_label})..."):
            records, missing = run_dispatch(
                prompt=prompt,
                system_prompt=trained_sys_prompt,
                model_keys=selected_models,
                temperature=temperature,
                max_tokens=max_tokens,
                progress_callback=on_result_trained,
                execution_mode="Trained",
                target_email_id=target_eid,
            )

    else:  # MODE_BOTH
        total_tasks = len(selected_models) * 2
        trained_sys_prompt = assemble_system_prompt(
            base_system_prompt=system_prompt,
            target_email_id=target_eid,
            trained=True,
            dataset=benchmark_dataset,
        )
        target_label = f"Email {target_eid}" if target_eid else "Custom Message"

        def on_result_both(record):
            global done
            done += 1
            progress_area.progress(
                min(done / max(total_tasks, 1), 1.0),
                text=f"{done}/{total_tasks} evaluations completed (Untrained & Trained on {target_label})",
            )
            st.session_state.results.append(record)

        with st.spinner(f"Running Both Untrained & Leave-One-Out Trained evaluations (Target: {target_label})..."):
            records, missing = run_dispatch_both(
                prompt=prompt,
                untrained_system_prompt=system_prompt,
                trained_system_prompt=trained_sys_prompt,
                model_keys=selected_models,
                temperature=temperature,
                max_tokens=max_tokens,
                target_email_id=target_eid,
                progress_callback=on_result_both,
            )

    st.session_state.results = records
    progress_area.empty()

    if missing:
        st.warning(
            f"Skipped (missing API key in secrets/.env): {', '.join(missing)}. "
            "Add the relevant key to run these too."
        )

# --------------------------------------------------------------- results --
if st.session_state.results:
    _order_map = {k: idx for idx, k in enumerate(MODEL_SEQUENCE)}
    _mode_order = {"Untrained": 0, "Trained": 1}
    st.session_state.results = sorted(
        st.session_state.results,
        key=lambda r: (
            _order_map.get(r.get("model_key", ""), 999),
            _mode_order.get(r.get("execution_mode", "Untrained"), 99),
        ),
    )

    distinct_modes_in_results = {
        r.get("execution_mode", "Untrained") for r in st.session_state.results if not r.get("error")
    }
    is_both_results = len(distinct_modes_in_results) > 1
    has_trained_results = "Trained" in distinct_modes_in_results and execution_mode != MODE_UNTRAINED

    active_target_eid = st.session_state.get("last_target_email_id") if has_trained_results else None
    active_expert_profile = (
        benchmark_dataset.get(active_target_eid)
        if (has_trained_results and active_target_eid in benchmark_dataset)
        else None
    )

    # 1. First: 5-Dimension Behavioral Analysis & Graphs
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
        if is_both_results:
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            untrained_mean = long_df[long_df["execution_mode"] == "Untrained"]["score"].mean()
            trained_mean = long_df[long_df["execution_mode"] == "Trained"]["score"].mean()
            delta_mean = trained_mean - untrained_mean if pd.notna(untrained_mean) and pd.notna(trained_mean) else 0.0
            with m_col1:
                st.metric("Evaluations Completed", f"{len(wide_df)} ({len(wide_df) // 2} Models × 2)")
            with m_col2:
                st.metric("Untrained Avg Score", f"{untrained_mean:.2f} / 5.0" if pd.notna(untrained_mean) else "N/A")
            with m_col3:
                st.metric(
                    "Trained Avg Score",
                    f"{trained_mean:.2f} / 5.0" if pd.notna(trained_mean) else "N/A",
                    delta=f"{delta_mean:+.2f} vs Untrained",
                )
            with m_col4:
                if active_expert_profile:
                    exp_mean = sum(active_expert_profile["dimension_exact_scores"].values()) / 5.0
                    st.metric(
                        f"Expert Benchmark (Email {active_target_eid})",
                        f"{exp_mean:.2f} / 5.0",
                        f"Total: {active_expert_profile['total_mean_score']:.1f} / 75",
                    )
                else:
                    flagged_count = (wide_df["overall_flag"].str.contains("FLAGGED")).sum()
                    st.metric("Flagged Evaluations", f"{flagged_count} / {len(wide_df)}")
        else:
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
                if active_expert_profile:
                    exp_mean = sum(active_expert_profile["dimension_exact_scores"].values()) / 5.0
                    st.metric(
                        f"Expert Benchmark (Email {active_target_eid})",
                        f"{exp_mean:.2f} / 5.0",
                        f"Total: {active_expert_profile['total_mean_score']:.1f} / 75",
                    )
                else:
                    flagged_count = (wide_df["overall_flag"].str.contains("FLAGGED")).sum()
                    st.metric("Models Flagging Exploitative Tone", f"{flagged_count} / {len(wide_df)}")

        # Model filter selection (preserving canonical model sequence)
        available_models = wide_df["model_display"].tolist()
        selected_chart_models = st.multiselect(
            "Filter models in graphs:",
            options=available_models,
            default=available_models,
            help="Select which models to compare in the graphs below.",
        )

        filtered_long_df = long_df[long_df["model_display"].isin(selected_chart_models)]
        filtered_wide_df = wide_df[wide_df["model_display"].isin(selected_chart_models)].reset_index(drop=True)

        if filtered_long_df.empty:
            st.info("Select at least one model above to display the graphs.")
        else:
            # Hierarchy:
            # 1) Cultural Comparison
            # 2) Overall Comparison
            # 3) Dimension Comparison (also has inside Dimension Heatmap)
            # 4) Dimension Scores Table
            # (+ Untrained vs. Trained (Leave-One-Out) vs. Expert Benchmark ONLY in Both mode)
            if is_both_results:
                tab_cultural, tab_overall, tab_dim, tab_table, tab_both_comp = st.tabs([
                    "🌍 Cultural Comparison",
                    "📈 Overall Comparison",
                    "📊 Dimension Comparison",
                    "📋 Dimension Scores Table",
                    "⚖️ Untrained vs. Trained (Leave-One-Out) vs. Expert Benchmark",
                ])
            else:
                tab_cultural, tab_overall, tab_dim, tab_table = st.tabs([
                    "🌍 Cultural Comparison",
                    "📈 Overall Comparison",
                    "📊 Dimension Comparison",
                    "📋 Dimension Scores Table",
                ])

            # 1) Cultural Comparison
            with tab_cultural:
                reg_chart = create_regional_bar_chart(filtered_long_df)
                if reg_chart is not None:
                    st.altair_chart(reg_chart, use_container_width=True)
                else:
                    st.info(
                        "Cultural comparison requires responses from models across at least two different cultural regions "
                        "(e.g., Western/US vs China vs Europe)."
                    )

            # 2) Overall Comparison
            with tab_overall:
                st.altair_chart(create_total_score_bar_chart(filtered_wide_df), use_container_width=True)

            # 3) Dimension Comparison (also has inside Dimension Heatmap)
            with tab_dim:
                st.altair_chart(create_grouped_bar_chart(filtered_long_df), use_container_width=True)
                st.altair_chart(create_average_dimension_chart(filtered_long_df), use_container_width=True)
                st.divider()
                st.markdown("##### 🗺️ Dimension Heatmap")
                st.altair_chart(create_heatmap_chart(filtered_long_df), use_container_width=True)

            # 4) Dimension Scores Table
            with tab_table:
                st.markdown("##### 15-Item Response Scores & Dimension Analysis")
                st.caption("Question responses (Q1–Q15), Total Score (out of 75), and converted 5-dimension decimal scores.")
                dim_table_height = min(max(520, (len(filtered_wide_df) + 1) * 38 + 20), 980)
                st.dataframe(filtered_wide_df, use_container_width=True, height=dim_table_height)
                csv_data = filtered_wide_df.to_csv(index=False)
                st.download_button(
                    "Download 15-Item Scores (CSV)",
                    data=csv_data,
                    file_name=f"15_item_dimension_scores_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

            # Only calculated and rendered in Both mode
            if is_both_results:
                with tab_both_comp:
                    st.altair_chart(
                        create_mode_comparison_chart(filtered_long_df, active_expert_profile),
                        use_container_width=True,
                    )
                    shift_chart = create_model_shift_chart(
                        filtered_wide_df,
                        expert_total=active_expert_profile["total_mean_score"] if active_expert_profile else None,
                    )
                    if shift_chart is not None:
                        st.altair_chart(shift_chart, use_container_width=True)

                    if active_expert_profile:
                        st.markdown(
                            f"##### 🎯 Validated Expert Consensus Benchmark for Email {active_target_eid} "
                            f"(*{active_expert_profile['subject']}* — {active_expert_profile['rater_count']} Expert Raters)"
                        )
                        exp_row = {
                            "Benchmark": f"Expert Consensus (Email {active_target_eid})",
                            **{f"Q{i}": active_expert_profile["likert_scores"][str(i)] for i in range(1, 16)},
                            "Total Score (Likert 15-75)": active_expert_profile["total_likert_score"],
                            "Total Score (Exact Mean)": active_expert_profile["total_mean_score"],
                            **active_expert_profile["dimension_exact_scores"],
                        }
                        st.dataframe(pd.DataFrame([exp_row]), use_container_width=True, hide_index=True)

    # 2. Second: Tabular results of all
    st.divider()
    st.subheader("Results Overview")
    df = pd.DataFrame(st.session_state.results)
    df["model_version"] = df.apply(get_exact_model_version, axis=1)
    if "execution_mode" not in df.columns:
        df["execution_mode"] = "Untrained"
    overview_cols = ["model_key", "model_version", "company", "region", "execution_mode", "response", "error"]
    overview_table_height = min(max(520, (len(df) + 1) * 38 + 20), 980)
    st.dataframe(
        df[[c for c in overview_cols if c in df.columns]],
        use_container_width=True,
        height=overview_table_height,
    )

    jsonl_data = "\n".join(json.dumps(r, ensure_ascii=False) for r in st.session_state.results)
    st.download_button(
        "Download results (JSONL)",
        data=jsonl_data,
        file_name=f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        mime="application/json",
    )

    # 3. Last: All LLM Show/Hide Results
    st.divider()
    res_hdr_left, res_hdr_right = st.columns([3, 1])
    with res_hdr_left:
        st.subheader("Model Responses")
    with res_hdr_right:
        expand_all = st.toggle("Expand all responses", value=False, help="Toggle to show or hide all model responses at once.")

    def _render_response_card(record, show_mode_badge=False):
        with st.container(border=True):
            exact_ver = get_exact_model_version(record)
            mode_str = record.get("execution_mode", "Untrained")
            mode_tag = f" · **`[{mode_str}]`**" if show_mode_badge else ""
            st.markdown(f"**{exact_ver}** (`{record['model_key']}`) · {record['company']} · _{record['region']}_{mode_tag}")
            if record.get("error"):
                st.error(record["error"])
            else:
                scores = parse_dimension_scores(record.get("response"))
                flag = parse_overall_flag(record.get("response"), scores)

                badges = []
                if flag is True:
                    badges.append("🚩 :red[**FLAGGED**]")
                elif flag is False:
                    badges.append("✅ :green[**CLEAR**]")

                high_dims = [f"{d.split()[0]}: {s:.2f}/5" for d, s in scores.items() if s and s >= 3.0]
                if high_dims:
                    badges.append(f"⚠️ {', '.join(high_dims)}")

                if badges:
                    st.caption(" · ".join(badges))

                with st.expander("📄 Show / Hide Response", expanded=expand_all):
                    st.markdown(record["response"])

    if is_both_results:
        st.caption("Left column: **Untrained (Zero-Shot)** · Right column: **Trained (Leave-One-Out In-Context)**")
        grouped_by_model = {}
        for rec in st.session_state.results:
            grouped_by_model.setdefault(rec["model_key"], {})[rec.get("execution_mode", "Untrained")] = rec

        for m_key in MODEL_SEQUENCE:
            if m_key not in grouped_by_model:
                continue
            pair = grouped_by_model[m_key]
            col_u, col_t = st.columns(2)
            with col_u:
                if "Untrained" in pair:
                    _render_response_card(pair["Untrained"], show_mode_badge=True)
            with col_t:
                if "Trained" in pair:
                    _render_response_card(pair["Trained"], show_mode_badge=True)
    else:
        cols = st.columns(2)
        for i, record in enumerate(st.session_state.results):
            with cols[i % 2]:
                _render_response_card(record, show_mode_badge=True)

