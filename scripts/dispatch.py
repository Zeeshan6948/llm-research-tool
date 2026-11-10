"""
dispatch.py -- send one prompt to all (or selected) models and save results.

Provider routing (no LiteLLM):
  - gemini/*          → google-generativeai SDK  (native system_instruction)
  - together_ai/*     → OpenAI SDK + Together AI base URL
  - mistral/*         → OpenAI SDK + Mistral AI base URL
  - scaleway/*        → OpenAI SDK + Scaleway Generative APIs base URL
  - premai/*          → OpenAI SDK + Prem AI base URL
  - gpt-* / o1-*      → OpenAI SDK (default)

Usage:
    python dispatch.py --prompt "Is this leadership behavior exploitative? ..."
    python dispatch.py --prompt-file scenarios/scenario1.txt
    python dispatch.py --prompt "..." --models mistral,qwen,deepseek
    python dispatch.py --prompt "..." --temperature 0.3

Results are appended to results/run_<timestamp>.jsonl -- one JSON line
per model response, so you can load them later with pandas for the
comparison/evaluation stage.
"""

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from dotenv import load_dotenv
from tqdm import tqdm

try:
    from config import MODELS
except ImportError:
    from scripts.config import MODELS

load_dotenv()

# ---------------------------------------------------------------------------
# Provider configuration: maps the model-string prefix to its base URL.
# Gemini is handled separately via the google-generativeai SDK.
# ---------------------------------------------------------------------------
_OPENAI_COMPAT_PROVIDERS = {
    "together_ai": "https://api.together.xyz/v1",
    "mistral":     "https://api.mistral.ai/v1",
    "scaleway":    "https://api.scaleway.ai/v1",
    "premai":      "https://gateway.prem.io/v1",
    # openai has no base_url override (uses SDK default)
}


def get_secret(key: str, default: str = "") -> str:
    """Retrieve an API key or config variable.

    Checks:
    1. Streamlit Secrets (st.secrets) if running within Streamlit Cloud or local Streamlit.
    2. Environment variables / .env (os.environ).
    """
    try:
        import streamlit as st
        if hasattr(st, "secrets") and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass
    return os.getenv(key, default)


def _parse_model_string(model_str: str):
    """Split 'provider/model-name' into (provider, model_id).

    Examples
    --------
    'gemini/gemini-3.6-flash'                        -> ('gemini', 'gemini-3.6-flash')
    'anthropic/claude-3-7-sonnet-20250219'           -> ('anthropic', 'claude-3-7-sonnet-20250219')
    'together_ai/meta-llama/Llama-3.3-70B-...'       -> ('together_ai', 'meta-llama/Llama-3.3-70B-...')
    'mistral/open-mistral-7b'                        -> ('mistral', 'open-mistral-7b')
    'gpt-4o-mini'                                    -> ('openai', 'gpt-4o-mini')
    """
    known_prefixes = ("gemini/", "anthropic/", "claude/", "together_ai/", "mistral/", "scaleway/", "premai/")
    for prefix in known_prefixes:
        if model_str.startswith(prefix):
            provider = prefix.rstrip("/")
            if provider == "claude":
                provider = "anthropic"
            model_id = model_str[len(prefix):]
            return provider, model_id
    # No recognised prefix → assume OpenAI
    return "openai", model_str


def _call_gemini(model_id, system_prompt, prompt, temperature, max_tokens):
    """Call Gemini via the google-genai SDK (the current, non-deprecated SDK).

    system_instruction is a first-class parameter -- no workarounds needed.

    Key considerations:
    1. Token Allocation: In thinking/reasoning models or complex table generation,
       a low token limit (e.g. <= 500) causes Gemini to halt mid-table after 1-2 lines.
       We enforce a generous output token budget (at least 2048) so the full table,
       evaluations, notes, and rewrite are completely generated.
    2. Thinking Budget: Set thinking_budget=0 to ensure 100% of tokens go to the
       visible response and none are wasted on internal thoughts.
    3. Transient 503 Retries: If Google servers report temporary high demand (503),
       retry with brief backoff.
    """
    import time
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=get_secret("GEMINI_API_KEY"))

    # Allocate enough tokens for the complete 15-question breakdown and 5-dimension summary
    gemini_max_tokens = max(max_tokens, 4096)

    config = types.GenerateContentConfig(
        system_instruction=system_prompt or None,
        temperature=temperature,
        max_output_tokens=gemini_max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),  # disable thinking
    )

    models_to_try = [model_id]
    # If standard 3.6 is facing regional 503 spikes, fallback gracefully
    if "3.6-flash" in model_id:
        models_to_try.append("gemini-flash-latest")

    last_exc = None
    for attempt_model in models_to_try:
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model=attempt_model,
                    contents=prompt,
                    config=config,
                )
                if response.candidates and response.candidates[0].content:
                    parts = response.candidates[0].content.parts or []
                    text = "".join(
                        part.text
                        for part in parts
                        if getattr(part, "text", None) and not getattr(part, "thought", False)
                    )
                    if text:
                        return text
                return response.text or ""
            except Exception as exc:
                last_exc = exc
                # If high demand (503) or rate limit (429), pause and retry
                if "503" in str(exc) or "429" in str(exc):
                    time.sleep(1.5 * (attempt + 1))
                    continue
                break

    if last_exc:
        raise last_exc
    return "No response received from Gemini."


def _call_claude(model_id, api_key, system_prompt, prompt, temperature, max_tokens):
    """Call Anthropic Claude via the official anthropic Python SDK.

    system is a first-class parameter on client.messages.create.
    """
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    claude_max_tokens = max(max_tokens, 4096)
    kwargs = {
        "model": model_id,
        "max_tokens": claude_max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if temperature is not None:
        kwargs["temperature"] = float(temperature)
    if system_prompt:
        kwargs["system"] = system_prompt

    try:
        response = client.messages.create(**kwargs)
    except TypeError as e:
        # Fallback if an older wrapper or custom environment rejects 'temperature'
        if "temperature" in str(e):
            kwargs.pop("temperature", None)
            response = client.messages.create(**kwargs)
        else:
            raise e

    return response.content[0].text


def _has_complete_15_item_table(text: str, min_items: int = 12) -> bool:
    """Return True if `text` contains at least `min_items` valid scored rows (1..15) in a Markdown table."""
    if not text:
        return False
    import re

    normalized = _normalize_markdown_table(text)
    lines = normalized.splitlines()
    found = set()
    for num in range(1, 16):
        pat = rf"^\s*\|\s*(?:Item\s+|Q\s*)?{num}\s*\|\s*[1-5](?:\.0)?\s*\|"
        for line in lines:
            if re.search(pat, line, re.IGNORECASE):
                found.add(num)
                break
    return len(found) >= min_items


def _call_openai_compat(model_id, provider, api_key, system_prompt, prompt, temperature, max_tokens):
    """Call any OpenAI-compatible endpoint (OpenAI, Together AI, Mistral).

    All three speak the same chat/completions API; only base_url and api_key differ.
    Includes reasoning-suppression and automatic table-enforcement retry for models
    like Llama (Muse-Spark), GLM, and Kimi when long Trained prompts are used.
    """
    from openai import OpenAI

    base_url = _OPENAI_COMPAT_PROVIDERS.get(provider)  # None → SDK uses OpenAI default
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    # Project-scoped OpenAI keys (sk-proj-...) may need the project header.
    if provider == "openai":
        project_id = get_secret("OPENAI_PROJECT_ID", "")
        if project_id:
            kwargs["default_headers"] = {"OpenAI-Project": project_id}

    client = OpenAI(**kwargs)

    # Append a concise format reminder at the end of the user prompt so long system prompts
    # in Trained mode don't cause Llama, GLM, or Kimi to drift into prose/non-tabular output.
    enforced_user_prompt = (
        f"{prompt.rstrip()}\n\n"
        "[IMPORTANT: Respond ONLY with the 17-line Markdown table starting immediately with "
        "`| Question | Answer | Reason |` for all 15 questions (1 to 15). "
        "Do NOT output any thinking, preamble, or non-table text.]"
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": enforced_user_prompt})

    # Allocate generous token budget (8192 for Together AI reasoning models, fallback to 4096)
    # so models like GLM-5.3-Flash, Kimi-K3, and Muse-Spark never exhaust max_tokens inside reasoning_content.
    effective_max_tokens = max(max_tokens, 8192 if provider == "together_ai" else 4096)

    def _invoke(msgs, tok_limit, disable_thinking=True):
        req_kwargs = {
            "model": model_id,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": tok_limit,
        }
        if provider == "together_ai" and disable_thinking:
            req_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        try:
            return client.chat.completions.create(**req_kwargs)
        except Exception:
            # Fallback if provider rejects extra_body or 8192 token limit
            req_kwargs.pop("extra_body", None)
            try:
                return client.chat.completions.create(**req_kwargs)
            except Exception:
                req_kwargs["max_tokens"] = 4096
                return client.chat.completions.create(**req_kwargs)

    response = _invoke(messages, effective_max_tokens, disable_thinking=True)
    choice_msg = response.choices[0].message
    content = (choice_msg.content or "").strip()
    reasoning_text = getattr(choice_msg, "reasoning_content", None) or ""

    # If content is empty or doesn't contain a valid 15-item table (e.g. model output prose or
    # spent its budget in reasoning_content), run a fast format-repair pass.
    if not _has_complete_15_item_table(content):
        draft_source = content if content else reasoning_text
        retry_messages = [
            {
                "role": "system",
                "content": (
                    "You are an Exploitative Leadership Training Coach (ELTC). "
                    "You MUST output ONLY a 17-line Markdown table starting immediately with "
                    "`| Question | Answer | Reason |` followed by `| :---: | :---: | :--- |` "
                    "and rows `| 1 |` through `| 15 |`. Answer must be an integer 1 to 5. "
                    "Reason must be <=30 words per row. No prose before or after the table."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Evaluate the following workplace email across all 15 scale items (1 to 15) "
                    f"and output ONLY the 17-line Markdown table (`| Question | Answer | Reason |`).\n\n"
                    f"EMAIL TO EVALUATE:\n{prompt.strip()}\n\n"
                    + (
                        f"Your preliminary analysis notes (convert these into the strict 15-row table):\n{draft_source[:3500]}\n\n"
                        if draft_source
                        else ""
                    )
                    + "Begin your response immediately with:\n| Question | Answer | Reason |\n| :---: | :---: | :--- |"
                ),
            },
        ]
        try:
            retry_resp = _invoke(retry_messages, 4096, disable_thinking=True)
            retry_msg = retry_resp.choices[0].message
            retry_content = (retry_msg.content or "").strip()
            if _has_complete_15_item_table(retry_content) or (not content and retry_content):
                content = retry_content
        except Exception:
            pass

    if not content and reasoning_text:
        content = reasoning_text
    return content


def _normalize_markdown_table(text: str) -> str:
    """Ensure the LLM output is a clean, valid 3-column Markdown table with all 15 rows."""
    if not text:
        return text

    import re

    # Strip <think>...</think> blocks if any reasoning model embedded them in content
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    # Also strip unclosed <think> prefix if a model started with <think> and then output a table
    if "<think>" in cleaned.lower() and "|" in cleaned:
        first_pipe_line = cleaned.find("| Question")
        if first_pipe_line == -1:
            first_pipe_line = cleaned.find("|")
        if first_pipe_line != -1:
            cleaned = cleaned[first_pipe_line:]

    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:markdown|md)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    raw_lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    table_lines = []

    for line in raw_lines:
        if line.startswith("|"):
            # Normalize '| Q1 | 4 | ...' or '| Item 1 | 4 | ...' -> '| 1 | 4 | ...'
            line = re.sub(
                r"^\|\s*(?:Question|Item|Q)\s*(\d{1,2})\s*\|",
                r"| \1 |",
                line,
                flags=re.IGNORECASE,
            )
            table_lines.append(line)
            continue

        # Convert non-leading-pipe rows like '1 | 4 | Reason' or 'Q1 | 4 | Reason'
        m_pipe = re.match(r"^(?:Question\s*|Item\s*|Q)?(\d{1,2})\s*\|\s*([1-5])(?:\.0)?\s*\|\s*(.+)$", line, re.IGNORECASE)
        if m_pipe:
            q_num = int(m_pipe.group(1))
            if 1 <= q_num <= 15:
                table_lines.append(f"| {q_num} | {m_pipe.group(2)} | {m_pipe.group(3).rstrip('|').strip()} |")
                continue

        # Convert numbered list rows like '1. 4 - Reason' or '1: Score 4 - Reason' or 'Q1: 4 - Reason'
        m_list = re.match(
            r"^(?:\*\*)?(?:Question\s*|Item\s*|Q)?(\d{1,2})(?:\*\*)?[.):\-]\s*(?:Score[:\s]*)?\*\*?([1-5])(?:\.0)?(?:/5)?\*\*?\s*[\-–:|]\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        if m_list:
            q_num = int(m_list.group(1))
            if 1 <= q_num <= 15:
                reason_clean = m_list.group(3).replace("|", "\\|").strip()
                table_lines.append(f"| {q_num} | {m_list.group(2)} | {reason_clean} |")
                continue

    if not table_lines:
        return cleaned

    repaired = []
    has_header = False
    has_sep = False
    seen_questions = set()

    for line in table_lines:
        # Check if header row
        if re.search(r"\|\s*(?:Question|Item)\s*\|", line, re.IGNORECASE):
            if not has_header:
                repaired.append("| Question | Answer | Reason |")
                has_header = True
            continue

        # Check if a model merged the separator row with row 1: e.g. '| :---: | 5 | reason |'
        m_merged = re.match(r"^\|\s*:?-+:?\s*\|\s*([1-5])\s*\|(.*)$", line)
        if m_merged:
            if not has_header:
                repaired.append("| Question | Answer | Reason |")
                has_header = True
            if not has_sep:
                repaired.append("| :---: | :---: | :--- |")
                has_sep = True
            if 1 not in seen_questions:
                repaired.append(f"| 1 | {m_merged.group(1)} |{m_merged.group(2)}")
                seen_questions.add(1)
            continue

        # Check if normal separator row
        if re.match(r"^\|\s*:?-+:?\s*\|", line):
            if not has_header:
                repaired.append("| Question | Answer | Reason |")
                has_header = True
            if not has_sep:
                repaired.append("| :---: | :---: | :--- |")
                has_sep = True
            continue

        # Check if data row has a valid question number 1..15 (ignore extraneous non-1..15 tables)
        m_q = re.match(r"^\|\s*(\d{1,2})\s*\|\s*([1-5])(?:\.0)?(?:\s*/\s*5)?\s*\|(.*)$", line)
        if m_q:
            q_num = int(m_q.group(1))
            if 1 <= q_num <= 15:
                if q_num in seen_questions:
                    continue
                if not has_header:
                    repaired.append("| Question | Answer | Reason |")
                    has_header = True
                if not has_sep:
                    repaired.append("| :---: | :---: | :--- |")
                    has_sep = True
                reason_col = m_q.group(3).strip()
                if not reason_col.endswith("|"):
                    reason_col = f"{reason_col} |"
                repaired.append(f"| {q_num} | {m_q.group(2)} | {reason_col}")
                seen_questions.add(q_num)
                continue

        # Fallback data row
        if not has_header:
            repaired.append("| Question | Answer | Reason |")
            has_header = True
        if not has_sep:
            repaired.append("| :---: | :---: | :--- |")
            has_sep = True
        repaired.append(line)

    return "\n".join(repaired)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_available_models(requested_keys):
    """Return the subset of MODELS that (a) were requested and (b) have
    their required API key set in the environment. Prints a warning for
    any requested model that's missing its key."""
    available = {}
    for key in requested_keys:
        if key not in MODELS:
            print(f"  [skip] '{key}' is not in config.py MODELS -- typo?")
            continue
        entry = MODELS[key]
        if not get_secret(entry["env_key"]):
            print(f"  [skip] '{key}' needs {entry['env_key']} in your .env file or Streamlit secrets")
            continue
        available[key] = entry
    return available


def query_model(
    key,
    entry,
    prompt,
    system_prompt,
    temperature,
    max_tokens,
    execution_mode="Untrained",
    target_email_id=None,
):
    """Call one model via its native SDK and return a result dict.

    Never raises -- failures are captured in the 'error' field so one bad
    model doesn't stop the whole run.
    """
    record = {
        "model_key":       key,
        "model_string":    entry["model"],
        "region":          entry["region"],
        "company":         entry["company"],
        "execution_mode":  execution_mode,
        "target_email_id": target_email_id,
        "prompt":          prompt,
        "temperature":     temperature,
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "response":        None,
        "error":           None,
    }

    provider, model_id = _parse_model_string(entry["model"])
    api_key = get_secret(entry["env_key"], "")

    try:
        if provider == "gemini":
            record["response"] = _call_gemini(
                model_id, system_prompt, prompt, temperature, max_tokens
            )
        elif provider == "anthropic":
            record["response"] = _call_claude(
                model_id, api_key, system_prompt, prompt, temperature, max_tokens
            )
        else:
            record["response"] = _call_openai_compat(
                model_id, provider, api_key, system_prompt, prompt, temperature, max_tokens
            )
        if record["response"]:
            record["response"] = _normalize_markdown_table(record["response"])
    except Exception as exc:  # noqa: BLE001 -- capture ANY provider error
        record["error"] = str(exc)

    return record


def run_dispatch(
    prompt,
    system_prompt,
    model_keys,
    temperature,
    max_tokens,
    progress_callback=None,
    execution_mode="Untrained",
    target_email_id=None,
):
    """Reusable entry point (used by both the CLI and the Streamlit app).

    Queries all available models in parallel using ThreadPoolExecutor.
    As each model finishes, progress_callback is called immediately so
    Streamlit or CLI users see progress in real time.
    """
    available = get_available_models(model_keys)
    if not available:
        return [], [k for k in model_keys if k not in available]

    records_map = {}
    # Use max_workers up to the number of models to query all simultaneously
    with ThreadPoolExecutor(max_workers=min(len(available), 12)) as executor:
        future_to_key = {
            executor.submit(
                query_model,
                key,
                entry,
                prompt,
                system_prompt,
                temperature,
                max_tokens,
                execution_mode,
                target_email_id,
            ): key
            for key, entry in available.items()
        }

        for future in as_completed(future_to_key):
            record = future.result()
            records_map[record["model_key"]] = record
            if progress_callback:
                progress_callback(record)

    # Maintain the original requested order in the final records list
    ordered_records = [records_map[k] for k in available if k in records_map]
    return ordered_records, [k for k in model_keys if k not in available]


def run_dispatch_both(
    prompt,
    untrained_system_prompt,
    trained_system_prompt,
    model_keys,
    temperature,
    max_tokens,
    target_email_id=None,
    progress_callback=None,
):
    """Run both Untrained (Zero-Shot) and Trained (Leave-One-Out Few-Shot) prompts in parallel."""
    available = get_available_models(model_keys)
    if not available:
        return [], [k for k in model_keys if k not in available]

    tasks = []
    for key, entry in available.items():
        tasks.append((key, entry, untrained_system_prompt, "Untrained"))
        tasks.append((key, entry, trained_system_prompt, "Trained"))

    records_map = {}
    with ThreadPoolExecutor(max_workers=min(len(tasks), 16)) as executor:
        future_to_task = {
            executor.submit(
                query_model,
                key,
                entry,
                prompt,
                sys_p,
                temperature,
                max_tokens,
                mode_label,
                target_email_id,
            ): (key, mode_label)
            for key, entry, sys_p, mode_label in tasks
        }

        for future in as_completed(future_to_task):
            record = future.result()
            records_map[(record["model_key"], record["execution_mode"])] = record
            if progress_callback:
                progress_callback(record)

    ordered_records = []
    for key in available:
        for mode_label in ("Untrained", "Trained"):
            if (key, mode_label) in records_map:
                ordered_records.append(records_map[(key, mode_label)])

    return ordered_records, [k for k in model_keys if k not in available]


def main():
    parser = argparse.ArgumentParser(description="Send one prompt to selected LLMs.")
    parser.add_argument("--prompt", type=str, help="The prompt text.")
    parser.add_argument("--prompt-file", type=str, help="Path to a text file containing the prompt.")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are evaluating workplace leadership scenarios. Judge the behavior described "
        "and explain your reasoning clearly and concisely.",
        help="System prompt applied identically to every model (edit the default in this script "
        "or pass your own with this flag).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help="Comma-separated model keys from config.py (e.g. 'mistral,qwen,deepseek'), or 'all'.",
    )
    parser.add_argument("--temperature", type=float, default=0.3, help="Sampling temperature (fixed across all models for consistency).")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--out", type=str, default=None, help="Output JSONL path (default: results/run_<timestamp>.jsonl)")
    args = parser.parse_args()

    if not args.prompt and not args.prompt_file:
        sys.exit("Error: provide --prompt or --prompt-file")

    prompt = args.prompt
    if args.prompt_file:
        with open(args.prompt_file, "r", encoding="utf-8") as f:
            prompt = f.read().strip()

    requested_keys = list(MODELS.keys()) if args.models == "all" else [k.strip() for k in args.models.split(",")]

    print(f"Requested models: {', '.join(requested_keys)}")
    available = get_available_models(requested_keys)
    if not available:
        sys.exit("\nNo models available -- check your .env file has the right API keys set.")

    print(f"\nQuerying {len(available)} model(s) with temperature={args.temperature} ...\n")

    out_path = args.out or f"results/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    records = []
    for key, entry in tqdm(available.items(), desc="Models"):
        record = query_model(key, entry, prompt, args.system_prompt, args.temperature, args.max_tokens)
        records.append(record)
        status = "OK" if record["error"] is None else f"ERROR: {record['error'][:80]}"
        tqdm.write(f"  {key:10s} ({entry['region']:20s}) -> {status}")

    with open(out_path, "a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nSaved {len(records)} result(s) to {out_path}")


if __name__ == "__main__":
    main()
