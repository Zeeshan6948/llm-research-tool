"""
dispatch.py -- send one prompt to all (or selected) models and save results.

Provider routing (no LiteLLM):
  - gemini/*          → google-generativeai SDK  (native system_instruction)
  - together_ai/*     → OpenAI SDK + Together AI base URL
  - mistral/*         → OpenAI SDK + Mistral AI base URL
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
from datetime import datetime, timezone

from dotenv import load_dotenv
from tqdm import tqdm

from config import MODELS

load_dotenv()

# ---------------------------------------------------------------------------
# Provider configuration: maps the model-string prefix to its base URL.
# Gemini is handled separately via the google-generativeai SDK.
# ---------------------------------------------------------------------------
_OPENAI_COMPAT_PROVIDERS = {
    "together_ai": "https://api.together.xyz/v1",
    "mistral":     "https://api.mistral.ai/v1",
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
    known_prefixes = ("gemini/", "anthropic/", "claude/", "together_ai/", "mistral/")
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

    # Allocate enough tokens for the complete 5-dimension analysis + suggested rewrite
    gemini_max_tokens = max(max_tokens, 2048)

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

    kwargs = {
        "model": model_id,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system_prompt:
        kwargs["system"] = system_prompt

    response = client.messages.create(**kwargs)
    return response.content[0].text


def _call_openai_compat(model_id, provider, api_key, system_prompt, prompt, temperature, max_tokens):
    """Call any OpenAI-compatible endpoint (OpenAI, Together AI, Mistral).

    All three speak the same chat/completions API; only base_url and api_key differ.
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

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


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


def query_model(key, entry, prompt, system_prompt, temperature, max_tokens):
    """Call one model via its native SDK and return a result dict.

    Never raises -- failures are captured in the 'error' field so one bad
    model doesn't stop the whole run.
    """
    record = {
        "model_key":    key,
        "model_string": entry["model"],
        "region":       entry["region"],
        "company":      entry["company"],
        "prompt":       prompt,
        "temperature":  temperature,
        "timestamp":    datetime.now(timezone.utc).isoformat(),
        "response":     None,
        "error":        None,
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
    except Exception as exc:  # noqa: BLE001 -- capture ANY provider error
        record["error"] = str(exc)

    return record


def run_dispatch(prompt, system_prompt, model_keys, temperature, max_tokens, progress_callback=None):
    """Reusable entry point (used by both the CLI and the Streamlit app).

    progress_callback, if given, is called with each record as soon as it's
    ready -- lets the Streamlit app show results streaming in one by one
    instead of waiting for every model to finish.
    """
    available = get_available_models(model_keys)
    records = []
    for key, entry in available.items():
        record = query_model(key, entry, prompt, system_prompt, temperature, max_tokens)
        records.append(record)
        if progress_callback:
            progress_callback(record)
    return records, [k for k in model_keys if k not in available]


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
    parser.add_argument("--max-tokens", type=int, default=500)
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
