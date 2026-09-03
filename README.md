# Cross-Cultural LLM Comparison Tool

Sends the same prompt to multiple LLMs (grouped by region/company), and later
will fine-tune each one on your labeled "exploitative leadership" email data
and score them on a held-out set.

## What's built so far

- `config.py` -- the model registry (region, company, LiteLLM model string, API key needed)
- `dispatch.py` -- sends one prompt to all/selected models, saves results as JSONL (CLI use)
- `app.py` -- **web interface** (Streamlit) -- this is what you'd share with your professor
- `data/sample_labeled_emails.jsonl` -- example of the labeled-data format you'll use for fine-tuning later
- `.env.example` -- template for your API keys

## Not built yet (next steps)

- `finetune_mistral.py`, `finetune_together.py`, `finetune_vertex.py` -- one script per fine-tuning path
- `evaluate.py` -- scores fine-tuned models against your holdout set (accuracy, precision, recall, F1)
- After fine-tuning exists, `app.py` gets a second tab to browse/score those results too

## Running the web interface locally

```bash
pip install -r requirements.txt
cp .env.example .env   # then fill in your API keys
streamlit run app.py
```

This opens a browser tab at `http://localhost:8501`. Your professor picks
models in the sidebar, types a prompt, hits Run, and sees every model's
response side by side -- no terminal, no code.

## Making it live (so your professor just opens a link)

**Streamlit Community Cloud (free, recommended):**

1. Push this project to a GitHub repo (a **private** repo is fine and safer).
2. Go to https://streamlit.io/cloud, sign in with GitHub, click "New app",
   point it at your repo and `app.py`.
3. In the app's **Settings → Secrets**, paste your API keys in this format
   (same variable names as `.env.example`):
   ```
   MISTRAL_API_KEY = "your-key-here"
   TOGETHER_API_KEY = "your-key-here"
   GEMINI_API_KEY = "your-key-here"
   ```
   **Never commit your real `.env` file to GitHub** -- it's already excluded
   via `.gitignore` below. Secrets only go in the Streamlit Cloud panel.
4. Deploy. You'll get a public URL like `yourname-llm-tool.streamlit.app`
   to send your professor.

**Cost note:** every time someone clicks "Run" on the live page, it spends
your API credits. Keep an eye on usage on each provider's dashboard,
especially if you share the link widely.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Copy the env template and fill in the keys for the models you're using:
   ```bash
   cp .env.example .env
   # then edit .env with your actual API keys
   ```
   You don't need every key -- `dispatch.py` will skip any model whose key is missing
   and tell you which ones it skipped.

3. Get your API keys:
   - Mistral: https://console.mistral.ai
   - Gemini: https://aistudio.google.com/app/apikey
   - Together AI (covers Llama, Qwen, DeepSeek, GLM, Falcon-H1): https://api.together.ai/settings/api-keys

## Usage

Send a prompt to every configured model:
```bash
python dispatch.py --prompt "A manager tells their team: 'I need everyone here until this ships tonight, no exceptions.' Is this leadership behavior exploitative? Explain your reasoning."
```

Send to specific models only:
```bash
python dispatch.py --prompt "..." --models mistral,qwen,deepseek
```

Use a prompt from a file (handy once you have a folder of scenario prompts):
```bash
python dispatch.py --prompt-file scenarios/scenario1.txt
```

Fix the sampling temperature for methodological consistency (default is 0.3):
```bash
python dispatch.py --prompt "..." --temperature 0.0
```

## Output format

Each run appends to `results/run_<timestamp>.jsonl`, one JSON object per line:
```json
{
  "model_key": "mistral",
  "model_string": "mistral/mistral-large-latest",
  "region": "Europe (non-US)",
  "company": "Mistral AI",
  "prompt": "...",
  "temperature": 0.3,
  "timestamp": "2026-08-27T12:00:00+00:00",
  "response": "...",
  "error": null
}
```

Load results into pandas for analysis:
```python
import pandas as pd
df = pd.read_json("results/run_20260827_120000.jsonl", lines=True)
```

## A note on model IDs

Together AI and other providers periodically rename or retire model IDs.
Before a real data-collection run, double check the exact strings in
`config.py` against:
- https://docs.together.ai/docs/inference-models
- https://docs.mistral.ai/getting-started/models/
- https://ai.google.dev/gemini-api/docs/models
