"""
dispatch.py -- send one prompt to all (or selected) models and save results.

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

# litellm is imported after load_dotenv() so its provider clients pick up
# the keys from .env immediately.
import litellm  # noqa: E402

litellm.suppress_debug_info = True


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
        if not os.getenv(entry["env_key"]):
            print(f"  [skip] '{key}' needs {entry['env_key']} in your .env file")
            continue
        available[key] = entry
    return available


def query_model(key, entry, prompt, system_prompt, temperature, max_tokens):
    """Call one model via LiteLLM and return a result dict. Never raises --
    failures are captured in the 'error' field so one bad model doesn't
    stop the whole run."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    record = {
        "model_key": key,
        "model_string": entry["model"],
        "region": entry["region"],
        "company": entry["company"],
        "prompt": prompt,
        "temperature": temperature,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "response": None,
        "error": None,
    }

    try:
        result = litellm.completion(
            model=entry["model"],
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        record["response"] = result.choices[0].message.content
    except Exception as exc:  # noqa: BLE001 -- we want to capture ANY provider error
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
