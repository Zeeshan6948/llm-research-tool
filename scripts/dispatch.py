"""
dispatch.py -- send one prompt to all (or selected) models and save results.

Provider routing (no LiteLLM):
  - gemini/*          → google-genai SDK (native system_instruction)
  - anthropic/*       → anthropic SDK (native system prompt)
  - together_ai/*     → OpenAI SDK + Together AI base URL
  - mistral/*         → OpenAI SDK + Mistral AI base URL
  - publicai/*        → OpenAI SDK + Public AI / Swiss AI Apertus base URL
  - huggingface/*     → OpenAI SDK + Hugging Face Inference Providers router
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
import contextlib
import json
import os
import sys
import threading
import time
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
    "publicai":    "https://api.publicai.co/v1",
    "huggingface": "https://router.huggingface.co/v1",
    "scaleway":    "https://api.scaleway.ai/v1",
    "premai":      "https://gateway.prem.io/v1",
    # openai has no base_url override (uses SDK default)
}

# Public AI (`api.publicai.co`) serves Apertus directly and also backs ALIA/Bielik
# behind Hugging Face Router. Cap simultaneous requests to this shared origin to
# prevent Cloudflare 504 gateway timeouts during multi-model / "Both Above" bursts.
_PUBLICAI_ORIGIN_SEMAPHORE = threading.Semaphore(2)
# Serialize Gemini calls (e.g. Untrained + Trained in "Both Above" mode) to avoid
# colliding on free-tier per-minute quota limits (429 RESOURCE_EXHAUSTED).
_GEMINI_SEMAPHORE = threading.Semaphore(1)

_KNOWN_PREFIXES = (
    "gemini/",
    "anthropic/",
    "claude/",
    "together_ai/",
    "mistral/",
    "publicai/",
    "huggingface/",
    "hf/",
    "scaleway/",
    "premai/",
)


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


def _get_first_secret(keys, default: str = "") -> str:
    """Return the first non-empty secret across a sequence of environment/secret key names."""
    for k in keys or ():
        val = get_secret(k, "").strip()
        if val:
            return val
    return default


def _get_entry_api_key(entry: dict) -> str:
    """Resolve an entry's API key using its primary env_key and any env_key_aliases."""
    keys = [entry["env_key"]] + list(entry.get("env_key_aliases", []))
    return _get_first_secret(keys, "")


def _parse_model_string(model_str: str):
    """Split 'provider/model-name' into (provider, model_id).

    Examples
    --------
    'gemini/gemini-3.6-flash'                                -> ('gemini', 'gemini-3.6-flash')
    'anthropic/claude-3-7-sonnet-20250219'                   -> ('anthropic', 'claude-3-7-sonnet-20250219')
    'together_ai/meta-llama/Llama-3.3-70B-...'               -> ('together_ai', 'meta-llama/Llama-3.3-70B-...')
    'mistral/open-mistral-7b'                                -> ('mistral', 'open-mistral-7b')
    'publicai/swiss-ai/apertus-v1.5-70b'                     -> ('publicai', 'swiss-ai/apertus-v1.5-70b')
    'huggingface/utter-project/EuroLLM-22B-Instruct-2512'    -> ('huggingface', 'utter-project/EuroLLM-22B-Instruct-2512')
    'gpt-4o-mini'                                            -> ('openai', 'gpt-4o-mini')
    """
    for prefix in _KNOWN_PREFIXES:
        if model_str.startswith(prefix):
            provider = prefix.rstrip("/")
            if provider == "claude":
                provider = "anthropic"
            elif provider == "hf":
                provider = "huggingface"
            model_id = model_str[len(prefix):]
            return provider, model_id
    # No recognised prefix → assume OpenAI
    return "openai", model_str


def _call_gemini(model_id, system_prompt, prompt, temperature, max_tokens, return_model_id=False):
    """Call Gemini via the google-genai SDK (the current, non-deprecated SDK).

    system_instruction is a first-class parameter -- no workarounds needed.

    Key considerations:
    1. Token Allocation: Enforce a generous output token budget (at least 4096) so the
       full 15-item table is never truncated.
    2. Thinking Budget: Set thinking_budget=0 so 100% of tokens go to the visible response.
    3. Resilience: Serialize concurrent calls (e.g. Both Above mode), retry on transient
       503/429 spikes, and fall back only to versioned Flash models supported on both
       v1 and v1beta (never '-latest' aliases, which 404 on API version v1).
    """
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
    # Versioned Flash fallbacks valid on both v1 and v1beta (avoid '-latest' which 404s on v1)
    if "flash" in model_id.lower():
        for fb in ("gemini-3.5-flash", "gemini-3.7-flash", "gemini-3.5-flash-lite"):
            if fb != model_id:
                models_to_try.append(fb)

    primary_exc = None
    last_exc = None
    with _GEMINI_SEMAPHORE:
        for idx, attempt_model in enumerate(models_to_try):
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=attempt_model,
                        contents=prompt,
                        config=config,
                    )
                    text = ""
                    if response.candidates and response.candidates[0].content:
                        parts = response.candidates[0].content.parts or []
                        text = "".join(
                            part.text
                            for part in parts
                            if getattr(part, "text", None) and not getattr(part, "thought", False)
                        )
                    if not text:
                        text = response.text or ""
                    if return_model_id:
                        return text, attempt_model
                    return text
                except Exception as exc:
                    last_exc = exc
                    if idx == 0:
                        primary_exc = exc
                    err_str = str(exc)
                    # If high demand (503) or rate limit (429), pause and retry
                    if "503" in err_str or "429" in err_str:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    break

    # Surface the primary model's error if all fallbacks failed (avoid masking 429/503 behind a fallback 404)
    if primary_exc is not None:
        raise primary_exc
    if last_exc is not None:
        raise last_exc
    if return_model_id:
        return "No response received from Gemini.", model_id
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


def _is_model_unavailable_error(exc: Exception) -> bool:
    """Return True if an API error indicates the requested model ID is missing or not served."""
    msg = str(exc).lower()
    markers = (
        "404",
        "model_not_found",
        "model not found",
        "does not exist",
        "not supported",
        "no inference provider",
        "not deployed",
        "unknown model",
        "invalid model",
    )
    return any(m in msg for m in markers)


def _is_token_limit_error(exc: Exception) -> bool:
    """Return True if an API error indicates max_tokens exceeds the model's context window."""
    msg = str(exc).lower()
    markers = (
        "max_tokens",
        "max_new_tokens",
        "context length",
        "maximum context",
        "context window",
        "input validation error",
        "exceeds",
    )
    return any(m in msg for m in markers)


def _is_transient_error(exc: Exception) -> bool:
    """Return True if an API error is a transient gateway/timeout/rate-limit failure."""
    msg = str(exc).lower()
    markers = (
        "502",
        "503",
        "504",
        "429",
        "gateway time-out",
        "gateway timeout",
        "origin_gateway_timeout",
        "bad gateway",
        "service unavailable",
        "overloaded",
        "rate limit",
        "timed out",
        "timeout",
        "temporarily unavailable",
        "connection error",
    )
    return any(m in msg for m in markers)


def _call_openai_compat(
    model_id,
    provider,
    api_key,
    system_prompt,
    prompt,
    temperature,
    max_tokens,
    base_url_override=None,
    fallback_models=None,
    return_model_id=False,
):
    """Call any OpenAI-compatible endpoint (OpenAI, Together AI, Mistral, Public AI, Hugging Face).

    All speak the same chat/completions API; only base_url, headers, and api_key differ.
    Includes streaming keep-alive for Public AI / Hugging Face (preventing Cloudflare 504
    gateway timeouts), transient 5xx/429 retries, reasoning-suppression, context-length /
    model-ID fallbacks, and automatic table-enforcement retry when long Trained prompts are used.
    """
    from openai import OpenAI

    base_url = base_url_override or _OPENAI_COMPAT_PROVIDERS.get(provider)  # None → SDK uses OpenAI default
    kwargs = {"api_key": api_key, "timeout": 180.0}
    if base_url:
        kwargs["base_url"] = base_url

    headers = {}
    # Project-scoped OpenAI keys (sk-proj-...) may need the project header.
    if provider == "openai":
        project_id = get_secret("OPENAI_PROJECT_ID", "")
        if project_id:
            headers["OpenAI-Project"] = project_id
    # Public AI gateway requires a non-empty identifying User-Agent header.
    elif provider == "publicai":
        headers["User-Agent"] = get_secret("PUBLICAI_USER_AGENT", "llm-research-tool/1.0")

    if headers:
        kwargs["default_headers"] = headers

    client = OpenAI(**kwargs)

    # Append a concise format reminder at the end of the user prompt so long system prompts
    # in Trained mode don't cause open models to drift into prose/non-tabular output.
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

    # Allocate generous token budget (8192 for Together AI reasoning models, 4096 otherwise)
    # so reasoning models never exhaust max_tokens inside reasoning_content.
    effective_max_tokens = max(max_tokens, 8192 if provider == "together_ai" else 4096)

    # Stream responses for Public AI / Hugging Face so Cloudflare receives bytes immediately
    # and never terminates slow 40B/70B generations with a 100s 504 Gateway Timeout.
    use_stream = provider in ("publicai", "huggingface")

    def _run_completion(req_kwargs):
        if not use_stream:
            resp = client.chat.completions.create(**req_kwargs)
            choice_msg = resp.choices[0].message
            return (choice_msg.content or "").strip(), (getattr(choice_msg, "reasoning_content", None) or "")

        stream = client.chat.completions.create(**{**req_kwargs, "stream": True})
        content_parts = []
        reasoning_parts = []
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if not delta:
                continue
            if getattr(delta, "content", None):
                content_parts.append(delta.content)
            if getattr(delta, "reasoning_content", None):
                reasoning_parts.append(delta.reasoning_content)
        return "".join(content_parts).strip(), "".join(reasoning_parts)

    def _invoke_single(target_model, msgs, tok_limit, disable_thinking=True):
        req_kwargs = {
            "model": target_model,
            "messages": msgs,
            "temperature": temperature,
            "max_tokens": tok_limit,
        }
        if provider in ("together_ai", "huggingface", "publicai") and disable_thinking:
            req_kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}

        origin_lock = (
            _PUBLICAI_ORIGIN_SEMAPHORE
            if provider in ("publicai", "huggingface")
            else contextlib.nullcontext()
        )

        last_err = None
        for attempt in range(3):
            try:
                with origin_lock:
                    try:
                        return _run_completion(req_kwargs)
                    except Exception as first_exc:
                        # Retry without extra_body if provider rejected it
                        if "extra_body" in req_kwargs:
                            req_kwargs.pop("extra_body", None)
                            try:
                                return _run_completion(req_kwargs)
                            except Exception as second_exc:
                                first_exc = second_exc

                        # Retry with smaller max_tokens if the model has a tighter context/output cap
                        if _is_token_limit_error(first_exc) or req_kwargs["max_tokens"] > 4096:
                            for fallback_limit in (4096, 1536, 1024):
                                if fallback_limit >= req_kwargs["max_tokens"]:
                                    continue
                                req_kwargs["max_tokens"] = fallback_limit
                                try:
                                    return _run_completion(req_kwargs)
                                except Exception as limit_exc:
                                    first_exc = limit_exc
                                    if not _is_token_limit_error(limit_exc):
                                        break
                        raise first_exc
            except Exception as exc:
                last_err = exc
                if attempt < 2 and _is_transient_error(exc):
                    time.sleep(3.0 * (attempt + 1))
                    continue
                raise

        if last_err is not None:
            raise last_err
        return "", ""

    model_candidates = [model_id] + [m for m in (fallback_models or []) if m and m != model_id]
    active_model_id = model_id
    content = ""
    reasoning_text = ""
    last_exc = None
    succeeded = False

    for idx, cand_model in enumerate(model_candidates):
        try:
            content, reasoning_text = _invoke_single(
                cand_model, messages, effective_max_tokens, disable_thinking=True
            )
            active_model_id = cand_model
            succeeded = True
            break
        except Exception as exc:
            last_exc = exc
            if idx + 1 < len(model_candidates) and _is_model_unavailable_error(exc):
                continue
            raise

    if not succeeded and last_exc is not None:
        raise last_exc

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
            retry_content, _ = _invoke_single(active_model_id, retry_messages, 4096, disable_thinking=True)
            if _has_complete_15_item_table(retry_content) or (not content and retry_content):
                content = retry_content
        except Exception:
            pass

    if not content and reasoning_text:
        content = reasoning_text
    if return_model_id:
        return content, active_model_id
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
    numbered_rows = {}
    fallback_rows = []

    for line in table_lines:
        # Check if header row
        if re.search(r"\|\s*(?:Question|Item)\s*\|", line, re.IGNORECASE):
            has_header = True
            continue

        # Check if a model merged the separator row with row 1: e.g. '| :---: | 5 | reason |'
        m_merged = re.match(r"^\|\s*:?-+:?\s*\|\s*([1-5])\s*\|(.*)$", line)
        if m_merged:
            has_header = True
            has_sep = True
            if 1 not in numbered_rows:
                reason_col = m_merged.group(2).strip()
                if not reason_col.endswith("|"):
                    reason_col = f"{reason_col} |"
                numbered_rows[1] = f"| 1 | {m_merged.group(1)} | {reason_col}"
            continue

        # Check if normal separator row
        if re.match(r"^\|\s*:?-+:?\s*\|", line):
            has_header = True
            has_sep = True
            continue

        # Check if data row has a valid question number 1..15 (ignore duplicates / extraneous tables)
        m_q = re.match(r"^\|\s*(\d{1,2})\s*\|\s*([1-5])(?:\.0)?(?:\s*/\s*5)?\s*\|(.*)$", line)
        if m_q:
            q_num = int(m_q.group(1))
            if 1 <= q_num <= 15:
                if q_num not in numbered_rows:
                    reason_col = m_q.group(3).strip()
                    if not reason_col.endswith("|"):
                        reason_col = f"{reason_col} |"
                    numbered_rows[q_num] = f"| {q_num} | {m_q.group(2)} | {reason_col}"
                continue

        # Collect fallback row only if we don't end up with a structured 1..15 table
        fallback_rows.append(line)

    if numbered_rows:
        repaired = [
            "| Question | Answer | Reason |",
            "| :---: | :---: | :--- |",
        ] + [numbered_rows[q] for q in sorted(numbered_rows.keys())]
        return "\n".join(repaired)

    if fallback_rows:
        repaired = [
            "| Question | Answer | Reason |",
            "| :---: | :---: | :--- |",
        ] + fallback_rows
        return "\n".join(repaired)

    return cleaned


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
        if not _get_entry_api_key(entry):
            aliases = entry.get("env_key_aliases", [])
            alias_hint = f" (or {' / '.join(aliases)})" if aliases else ""
            print(f"  [skip] '{key}' needs {entry['env_key']}{alias_hint} in your .env file or Streamlit secrets")
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
    api_key = _get_entry_api_key(entry)

    # Allow optional per-entry model ID and base URL overrides via .env / Streamlit secrets
    model_override = _get_first_secret(entry.get("model_env_keys", []))
    if model_override:
        if any(model_override.startswith(p) for p in _KNOWN_PREFIXES):
            provider, model_id = _parse_model_string(model_override)
            record["model_string"] = model_override
        else:
            model_id = model_override
            record["model_string"] = f"{provider}/{model_id}" if provider != "openai" else model_id

    base_url_override = _get_first_secret(entry.get("base_url_env_keys", [])) or None
    fallback_models = [] if model_override else entry.get("fallback_models", [])

    try:
        if provider == "gemini":
            response_text, resolved_model_id = _call_gemini(
                model_id, system_prompt, prompt, temperature, max_tokens, return_model_id=True
            )
            record["response"] = response_text
            if resolved_model_id and resolved_model_id != model_id:
                record["model_string"] = f"gemini/{resolved_model_id}"
        elif provider == "anthropic":
            record["response"] = _call_claude(
                model_id, api_key, system_prompt, prompt, temperature, max_tokens
            )
        else:
            response_text, resolved_model_id = _call_openai_compat(
                model_id,
                provider,
                api_key,
                system_prompt,
                prompt,
                temperature,
                max_tokens,
                base_url_override=base_url_override,
                fallback_models=fallback_models,
                return_model_id=True,
            )
            record["response"] = response_text
            if resolved_model_id and resolved_model_id != model_id:
                record["model_string"] = (
                    f"{provider}/{resolved_model_id}" if provider != "openai" else resolved_model_id
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
