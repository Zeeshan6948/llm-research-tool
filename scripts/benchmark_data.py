"""
benchmark_data.py -- Data ingestion, local storage/caching, and Leave-One-Out (LOO)
prompt assembly service for the 9 workplace emails and 15-item expert benchmark ratings.

Ingests:
  1. data/Email Data.xlsx      (Column 1: Scenario 1..9, Column 2: Full Email Text / Vignette)
  2. data/Expert Results.xlsx  (Column 1: Scenario 1..9, Columns 2..16: Scale Item 1..15 Likert scores [1-5])

Caches the structured dataset locally in `data/benchmark_dataset.json` and exposes
utilities for UI state management and Leave-One-Out few-shot prompt injection.
"""

import json
import os
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
EMAIL_EXCEL_PATH = DATA_DIR / "Email Data.xlsx"
EXPERT_EXCEL_PATH = DATA_DIR / "Expert Results.xlsx"
CACHE_JSON_PATH = DATA_DIR / "benchmark_dataset.json"

SCALE_ITEM_DESCRIPTIONS: Dict[int, str] = {
    1: "Leader assumes employee's work can be used for personal benefit.",
    2: "Leader views employee as a means to achieve personal goals.",
    3: "Leader values own goals over employee needs.",
    4: "Leader puts employee under pressure to achieve personal goals.",
    5: "Leader increases workload without considering employee needs.",
    6: "Leader disregards current workload when assigning tasks.",
    7: "Leader assigns tedious tasks for personal benefit.",
    8: "Leader blocks professional development for own goals.",
    9: "Leader assigns routine/monotonous tasks for personal benefit.",
    10: "Leader uses employee's work for personal recognition.",
    11: "Leader presents employee's work as their own.",
    12: "Leader exploits employee's work for personal gain.",
    13: "Leader plays employee against colleagues.",
    14: "Leader manipulates employee to achieve personal goals.",
    15: "Leader deceives employee to achieve personal goals.",
}

DIMENSION_ITEMS_MAP: Dict[str, List[int]] = {
    "Genuine Egoistic Behaviors": [1, 2, 3],
    "Taking Credit": [10, 11, 12],
    "Exerting Pressure": [4, 5, 6],
    "Undermining Development": [7, 8, 9],
    "Manipulating": [13, 14, 15],
}


def _clean_excel_text(text: str) -> str:
    """Clean Excel XML escape sequences (e.g. _x000b_ vertical tabs) and non-breaking spaces."""
    if not text:
        return ""
    cleaned = re.sub(r"_x000[bB]_", "\n", str(text))
    cleaned = cleaned.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    # Collapse 3+ consecutive newlines to 2 newlines while preserving paragraph structure
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _col_letters_to_index(col_letters: str) -> int:
    """Convert Excel column letters (e.g. 'A' -> 0, 'B' -> 1, 'P' -> 15)."""
    idx = 0
    for ch in col_letters:
        if ch.isalpha():
            idx = idx * 26 + (ord(ch.upper()) - ord("A") + 1)
    return idx - 1


def _read_xlsx_rows(xlsx_path: Path) -> List[List[Any]]:
    """Read the first worksheet of an .xlsx file into a list of rows.

    Uses openpyxl if installed, with a zero-dependency fallback using Python's
    standard zipfile + xml.etree.ElementTree.
    """
    try:
        import openpyxl  # type: ignore

        wb = openpyxl.load_workbook(xlsx_path, data_only=True)
        ws = wb[wb.sheetnames[0]]
        rows = []
        for row in ws.iter_rows(values_only=True):
            if any(cell is not None for cell in row):
                rows.append(list(row))
        return rows
    except ImportError:
        pass

    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(xlsx_path, "r") as z:
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            ss_root = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in ss_root.findall(f"{ns}si"):
                parts = [t.text or "" for t in si.iter(f"{ns}t")]
                shared_strings.append("".join(parts))

        sheet_files = sorted(
            f for f in z.namelist() if f.startswith("xl/worksheets/sheet") and f.endswith(".xml")
        )
        if not sheet_files:
            raise ValueError(f"No worksheet found in {xlsx_path}")

        sheet_root = ET.fromstring(z.read(sheet_files[0]))
        parsed_rows: List[List[Any]] = []

        for row_el in sheet_root.iter(f"{ns}row"):
            row_dict: Dict[int, Any] = {}
            max_col = -1
            for c_el in row_el.findall(f"{ns}c"):
                ref = c_el.attrib.get("r", "")
                col_letters = "".join(ch for ch in ref if ch.isalpha())
                col_idx = _col_letters_to_index(col_letters) if col_letters else (max_col + 1)
                max_col = max(max_col, col_idx)

                cell_type = c_el.attrib.get("t", "")
                v_el = c_el.find(f"{ns}v")
                is_el = c_el.find(f"{ns}is")

                val: Any = None
                if v_el is not None and v_el.text is not None:
                    if cell_type == "s":
                        val = shared_strings[int(v_el.text)]
                    else:
                        val = v_el.text
                elif is_el is not None:
                    val = "".join(t.text or "" for t in is_el.iter(f"{ns}t"))

                row_dict[col_idx] = val

            if max_col >= 0:
                row_list = [row_dict.get(i) for i in range(max_col + 1)]
                if any(v is not None for v in row_list):
                    parsed_rows.append(row_list)

        return parsed_rows


def parse_emails_excel(xlsx_path: Path = EMAIL_EXCEL_PATH) -> Dict[int, str]:
    """Parse `Email Data.xlsx` into {email_id (1..9): full_email_text}."""
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Emails dataset not found at {xlsx_path}")

    rows = _read_xlsx_rows(xlsx_path)
    emails: Dict[int, str] = {}

    for row in rows:
        if len(row) < 2 or row[0] is None or row[1] is None:
            continue
        # Skip header row if first cell is non-numeric (e.g. 'Scenario')
        try:
            email_id = int(float(str(row[0]).strip()))
        except ValueError:
            continue

        if 1 <= email_id <= 9:
            emails[email_id] = _clean_excel_text(str(row[1]))

    return dict(sorted(emails.items()))


def parse_expert_ratings_excel(xlsx_path: Path = EXPERT_EXCEL_PATH) -> Dict[int, Dict[str, Any]]:
    """Parse `Expert Results.xlsx` into consensus and per-rater evaluations for Emails 1..9.

    Supports both:
      - Multi-rater rows per scenario (computes consensus integer Likert [1-5] and exact mean [1.00-5.00])
      - Single consensus row per scenario
    """
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Expert ratings dataset not found at {xlsx_path}")

    rows = _read_xlsx_rows(xlsx_path)
    grouped_raters: Dict[int, List[List[float]]] = {}

    for row in rows:
        if len(row) < 16 or row[0] is None:
            continue
        try:
            email_id = int(float(str(row[0]).strip()))
        except ValueError:
            continue

        if not (1 <= email_id <= 9):
            continue

        try:
            scores = [float(str(row[col_idx]).strip()) for col_idx in range(1, 16)]
        except (ValueError, TypeError):
            continue

        grouped_raters.setdefault(email_id, []).append(scores)

    expert_profiles: Dict[int, Dict[str, Any]] = {}
    for email_id in sorted(grouped_raters.keys()):
        rater_matrix = grouped_raters[email_id]
        num_raters = len(rater_matrix)

        # Compute mean and rounded integer Likert [1-5] consensus for each of the 15 items
        item_means: Dict[str, float] = {}
        item_likert: Dict[str, int] = {}

        for item_idx in range(15):
            q_num = item_idx + 1
            col_vals = [r[item_idx] for r in rater_matrix]
            mean_val = sum(col_vals) / num_raters
            rounded_val = int(round(mean_val))
            rounded_val = max(1, min(5, rounded_val))

            item_means[str(q_num)] = round(mean_val, 2)
            item_likert[str(q_num)] = rounded_val

        # Compute 5 behavioral dimension scores from both the consensus Likert integers and exact means
        dimension_likert_means: Dict[str, float] = {}
        dimension_exact_means: Dict[str, float] = {}
        for dim_name, q_ids in DIMENSION_ITEMS_MAP.items():
            likert_vals = [item_likert[str(q)] for q in q_ids]
            exact_vals = [item_means[str(q)] for q in q_ids]
            dimension_likert_means[dim_name] = round(sum(likert_vals) / len(likert_vals), 2)
            dimension_exact_means[dim_name] = round(sum(exact_vals) / len(exact_vals), 2)

        expert_profiles[email_id] = {
            "email_id": email_id,
            "rater_count": num_raters,
            "likert_scores": item_likert,  # { "1": 2, ..., "15": 2 } (1-5 integer scale)
            "mean_scores": item_means,     # { "1": 2.09, ..., "15": 2.18 }
            "total_likert_score": sum(item_likert.values()),
            "total_mean_score": round(sum(item_means.values()), 2),
            "dimension_scores": dimension_likert_means,
            "dimension_exact_scores": dimension_exact_means,
            "rater_rows": [[int(round(v)) for v in r] for r in rater_matrix],
        }

    return expert_profiles


def load_benchmark_dataset(force_reload: bool = False) -> Dict[int, Dict[str, Any]]:
    """Load the combined 9-email + 15-item expert benchmark dataset.

    Automatically parses the two Excel files in `data/` and caches the combined
    structured dataset locally to `data/benchmark_dataset.json`.
    """
    if not force_reload and CACHE_JSON_PATH.exists():
        try:
            # Check if Excel files are newer than cache
            cache_mtime = CACHE_JSON_PATH.stat().st_mtime
            email_mtime = EMAIL_EXCEL_PATH.stat().st_mtime if EMAIL_EXCEL_PATH.exists() else 0
            expert_mtime = EXPERT_EXCEL_PATH.stat().st_mtime if EXPERT_EXCEL_PATH.exists() else 0
            if cache_mtime >= max(email_mtime, expert_mtime):
                with open(CACHE_JSON_PATH, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                return {int(k): v for k, v in raw.get("scenarios", {}).items()}
        except Exception:
            pass

    emails = parse_emails_excel(EMAIL_EXCEL_PATH)
    expert_profiles = parse_expert_ratings_excel(EXPERT_EXCEL_PATH)

    combined: Dict[int, Dict[str, Any]] = {}
    for email_id in range(1, 10):
        email_text = emails.get(email_id, "")
        profile = expert_profiles.get(email_id, {})
        # Extract subject line for display convenience
        subject = f"Email {email_id}"
        for line in email_text.splitlines():
            if line.strip().lower().startswith("subject:"):
                subject = line.strip()[len("subject:"):].strip()
                break

        combined[email_id] = {
            "email_id": email_id,
            "subject": subject,
            "email_text": email_text,
            "rater_count": profile.get("rater_count", 0),
            "likert_scores": profile.get("likert_scores", {str(i): 1 for i in range(1, 16)}),
            "mean_scores": profile.get("mean_scores", {str(i): 1.0 for i in range(1, 16)}),
            "total_likert_score": profile.get("total_likert_score", 15),
            "total_mean_score": profile.get("total_mean_score", 15.0),
            "dimension_scores": profile.get("dimension_scores", {}),
            "dimension_exact_scores": profile.get("dimension_exact_scores", {}),
            "rater_rows": profile.get("rater_rows", []),
        }

    # Persist local JSON cache in data/
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(CACHE_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "description": "Cached 9 Workplace Emails and Validated 15-Item Expert Benchmark Ratings",
                    "source_files": [str(EMAIL_EXCEL_PATH.name), str(EXPERT_EXCEL_PATH.name)],
                    "scenarios": {str(k): v for k, v in combined.items()},
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
    except Exception:
        pass

    return combined


def get_leave_one_out_examples(
    target_email_id: Optional[int],
    dataset: Optional[Dict[int, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Return the reference emails {1..9} \\ {target_email_id} with their expert ratings."""
    if dataset is None:
        dataset = load_benchmark_dataset()
    if target_email_id is None:
        return [dataset[i] for i in sorted(dataset.keys())]
    return [dataset[i] for i in sorted(dataset.keys()) if i != target_email_id]


def format_leave_one_out_benchmark_block(
    target_email_id: Optional[int],
    dataset: Optional[Dict[int, Dict[str, Any]]] = None,
) -> str:
    """Format the leave-one-out expert ratings (without email text) into a compact calibration block.

    Avoids using a competing 3-column Markdown table so LLMs (such as Llama, GLM, and Kimi)
    do not confuse the calibration reference layout with the mandatory '| Question | Answer | Reason |'
    output table format.
    """
    if dataset is None:
        dataset = load_benchmark_dataset()

    ref_examples = get_leave_one_out_examples(target_email_id, dataset)
    ref_ids_str = ", ".join(f"Email {ex['email_id']}" for ex in ref_examples)

    lines = [
        "[VALIDATED EXPERT BENCHMARK REFERENCE DATA (LEAVE-ONE-OUT CALIBRATION)]",
        f"Reference expert ratings for {len(ref_examples)} scenarios ({ref_ids_str}) across Questions 1 to 15 (Scale 1=Minimal to 5=Very High).",
        "Use these expert rating distributions purely to calibrate your scoring sensitivity.",
        "DO NOT copy this reference format in your response — your response MUST strictly follow the MANDATORY OUTPUT FORMAT table (`| Question | Answer | Reason |`).",
        "",
    ]

    for ex in ref_examples:
        eid = ex["email_id"]
        likert = ex["likert_scores"]
        means = ex["mean_scores"]
        rater_count = ex.get("rater_count", 0)
        rater_rows = ex.get("rater_rows", [])

        consensus_pairs = ", ".join(
            f"Q{q}={likert.get(str(q), 1)} (mean {means.get(str(q), float(likert.get(str(q), 1))):.2f})"
            for q in range(1, 16)
        )
        lines.append(f"• Reference Scenario Email {eid} ({rater_count} Expert Raters):")
        lines.append(f"  - Expert Consensus (Q1..Q15): {consensus_pairs}")
        if rater_rows:
            rater_strs = [
                f"R{r_idx}:[{','.join(str(s) for s in r_scores)}]"
                for r_idx, r_scores in enumerate(rater_rows, start=1)
            ]
            lines.append(f"  - Individual Raters (Q1..Q15): {' | '.join(rater_strs)}")
        lines.append("")

    return "\n".join(lines).strip()


def assemble_system_prompt(
    base_system_prompt: str,
    target_email_id: Optional[int],
    trained: bool = False,
    dataset: Optional[Dict[int, Dict[str, Any]]] = None,
) -> str:
    """Assemble the final system prompt for Untrained (Zero-Shot) or Trained (Leave-One-Out Few-Shot) mode.

    In Untrained mode (`trained=False`), returns `base_system_prompt` unchanged.
    In Trained mode (`trained=True`), injects the Leave-One-Out expert rating profiles
    BEFORE 'MANDATORY OUTPUT FORMAT' (or before '[EMAIL TO EVALUATE]') so that the
    17-line Markdown table template remains the final instruction right before the email.
    """
    if not trained:
        return base_system_prompt

    benchmark_block = format_leave_one_out_benchmark_block(target_email_id, dataset)

    # Prefer placing the benchmark data right before MANDATORY OUTPUT FORMAT so the
    # 17-line table template is the very last thing the model sees before [EMAIL TO EVALUATE].
    format_marker = "MANDATORY OUTPUT FORMAT"
    eval_marker = "[EMAIL TO EVALUATE]"

    if format_marker in base_system_prompt:
        idx = base_system_prompt.rfind(format_marker)
        prefix = base_system_prompt[:idx].rstrip()
        suffix = base_system_prompt[idx:]
        return f"{prefix}\n\n{benchmark_block}\n\n{suffix}"

    if eval_marker in base_system_prompt:
        idx = base_system_prompt.rfind(eval_marker)
        prefix = base_system_prompt[:idx].rstrip()
        suffix = base_system_prompt[idx:]
        return f"{prefix}\n\n{benchmark_block}\n\n{suffix}"

    return f"{base_system_prompt.rstrip()}\n\n{benchmark_block}\n\n[EMAIL TO EVALUATE]\n"


def detect_email_id_from_text(
    prompt_text: str,
    dataset: Optional[Dict[int, Dict[str, Any]]] = None,
) -> Optional[int]:
    """Detect if the current prompt text matches one of the 9 benchmark emails."""
    if not prompt_text or not prompt_text.strip():
        return None
    if dataset is None:
        dataset = load_benchmark_dataset()
    norm_prompt = " ".join(prompt_text.split())
    for email_id, item in dataset.items():
        norm_email = " ".join(item["email_text"].split())
        if norm_prompt == norm_email:
            return email_id
    return None
