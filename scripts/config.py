"""
Model registry for the cross-cultural LLM comparison tool.

Each entry maps a short "key" (what you type on the command line) to:
  - region:   your cultural cluster label, used in results/reports
  - company:  the model's main provider
  - model:    the LiteLLM model string (provider/model-name)
  - env_key:  which .env variable must be set for this model to run
  - trainable: whether this model has a fine-tuning path (see finetune_*.py)

LiteLLM model string format is "provider/model-name". If a provider
renames or retires a model, update the "model" value here -- nothing
else in the codebase needs to change.

IMPORTANT: exact model IDs (especially on Together AI) change as new
versions ship. Before a real run, check the current IDs at:
  https://docs.together.ai/docs/inference-models
  https://docs.mistral.ai/getting-started/models/
  https://ai.google.dev/gemini-api/docs/models
and update the strings below.
"""

MODEL_SEQUENCE = [
    "gemini",
    "qwen",
    "mistral",
    "claude",
    "gpt4o",
    "deepseek",
    "llama",
    "glm",
    "kimi",
    "apertus",
    "alia",
    "bielik",
]

MODELS = {
    # NOTE: gemini-3.6-flash is the current Google-required model string but is
    # not yet in LiteLLM's local registry, so LiteLLM would silently drop the
    # system message. dispatch.py works around this by prepending the system
    # prompt directly into the user message for any gemini/ model.
    "gemini": {
        "region": "Western/US",
        "company": "Google",
        "model": "gemini/gemini-3.6-flash",
        "env_key": "GEMINI_API_KEY",
        "trainable": True,
    },
    "qwen": {
        "region": "China",
        "company": "Alibaba (via Together AI)",
        "model": "together_ai/Qwen/Qwen3.5-9B",
        "env_key": "TOGETHER_API_KEY",
        "trainable": True,
    },
    "mistral": {
        "region": "Europe (non-US)",
        "company": "Mistral AI",
        "model": "mistral/open-mistral-7b",
        "env_key": "MISTRAL_API_KEY",
        "trainable": True,
    },
    "claude": {
        "region": "Western/US",
        "company": "Anthropic",
        "model": "anthropic/claude-haiku-4-5-20251001",
        "env_key": "ANTHROPIC_API_KEY",
        "trainable": False,
    },
    # Optional extras -- chat-only, no fine-tuning path (see thesis notes).
    # OpenAI project-scoped keys (sk-proj-...) require the project to have a
    # funded billing account. gpt-4o-mini is cheaper and more accessible on
    # new/free-trial projects; switch to "gpt-4o" once billing is confirmed.
    # Set OPENAI_PROJECT_ID in .env if your key belongs to a specific project.
    "gpt4o": {
        "region": "Western/US",
        "company": "OpenAI",
        "model": "gpt-4o-mini",   # cheaper; swap to "gpt-4o" once quota confirmed
        "env_key": "OPENAI_API_KEY",
        "trainable": False,
    },
    "deepseek": {
        "region": "China",
        "company": "DeepSeek (via Together AI)",
        "model": "together_ai/deepseek-ai/DeepSeek-V4.1-Flash",
        "env_key": "TOGETHER_API_KEY",
        "trainable": True,
    },
    "llama": {
        "region": "Western/US",
        "company": "Meta (via Together AI)",
        "model": "together_ai/meta-models/Muse-Spark-Open-816B-A42B",
        "env_key": "TOGETHER_API_KEY",
        "trainable": True,
    },
    "glm": {
        "region": "China",
        "company": "Zhipu / Z.ai (via Together AI)",
        "model": "together_ai/zai-org/GLM-5.3-Flash",
        "env_key": "TOGETHER_API_KEY",
        "trainable": True,
    },
    "kimi": {
        "region": "China",
        "company": "Moonshot AI (via Together AI)",
        "model": "together_ai/moonshotai/Kimi-K3",
        "env_key": "TOGETHER_API_KEY",
        "trainable": False,
    },
    "apertus": {
        "region": "Europe (non-US)",
        "company": "Swiss AI (EPFL / ETH Zurich)",
        "model": "publicai/swiss-ai/apertus-v1.5-70b",
        "env_key": "SWISS_AI_API_KEY",
        "env_key_aliases": ["APERTUS_API_KEY", "PUBLICAI_API_KEY"],
        "base_url_env_keys": ["SWISS_AI_BASE_URL", "APERTUS_BASE_URL", "PUBLICAI_BASE_URL"],
        "model_env_keys": ["SWISS_AI_MODEL", "APERTUS_MODEL"],
        "fallback_models": [
            "swiss-ai/apertus-70b-instruct",
            "swiss-ai/Apertus-70B-Instruct-2509",
            "swiss-ai/apertus-v1.5-8b",
            "swiss-ai/apertus-8b-instruct",
        ],
        "trainable": True,
    },
    # Note: utter-project/EuroLLM-22B-Instruct-2512 currently has no live serverless
    # provider on Hugging Face Inference Providers. ALIA-40B (Spain / EU) and
    # Bielik-11B (Poland / EU) are live sovereign European alternatives via HF_TOKEN.
    "alia": {
        "region": "Europe (non-US)",
        "company": "Barcelona Supercomputing Center (via Hugging Face)",
        "model": "huggingface/BSC-LT/ALIA-40b-instruct-2606",
        "env_key": "HF_TOKEN",
        "env_key_aliases": ["HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN"],
        "base_url_env_keys": ["ALIA_BASE_URL", "HF_BASE_URL"],
        "model_env_keys": ["ALIA_MODEL"],
        "fallback_models": [
            "BSC-LT/ALIA-40b-instruct",
        ],
        "trainable": True,
    },
    "bielik": {
        "region": "Europe (non-US)",
        "company": "SpeakLeash / Poland (via Hugging Face)",
        "model": "huggingface/speakleash/Bielik-11B-v3.0-Instruct",
        "env_key": "HF_TOKEN",
        "env_key_aliases": ["HUGGINGFACE_API_KEY", "HUGGINGFACEHUB_API_TOKEN"],
        "base_url_env_keys": ["BIELIK_BASE_URL", "HF_BASE_URL"],
        "model_env_keys": ["BIELIK_MODEL"],
        "trainable": True,
    },
}
