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

MODELS = {
    "mistral": {
        "region": "Europe (non-US)",
        "company": "Mistral AI",
        "model": "mistral/open-mistral-7b",
        "env_key": "MISTRAL_API_KEY",
        "trainable": True,
    },
    "gemini": {
        "region": "Western/US",
        "company": "Google",
        "model": "gemini/gemini-3.6-flash",
        "env_key": "GEMINI_API_KEY",
        "trainable": True,
    },
    "llama": {
        "region": "Western/US",
        "company": "Meta (via Together AI)",
        "model": "together_ai/meta-llama/Llama-3.3-70B-Instruct-Turbo",
        "env_key": "TOGETHER_API_KEY",
        "trainable": True,
    },
    "qwen": {
        "region": "China",
        "company": "Alibaba (via Together AI)",
        "model": "together_ai/Qwen/Qwen2.5-72B-Instruct-Turbo",
        "env_key": "TOGETHER_API_KEY",
        "trainable": True,
    },
    "deepseek": {
        "region": "China",
        "company": "DeepSeek (via Together AI)",
        "model": "together_ai/deepseek-ai/DeepSeek-V3",
        "env_key": "TOGETHER_API_KEY",
        "trainable": True,
    },
    "glm": {
        "region": "China",
        "company": "Zhipu / Z.ai (via Together AI)",
        "model": "together_ai/zai-org/GLM-4.5-Air",
        "env_key": "TOGETHER_API_KEY",
        "trainable": True,
    },
    "falcon": {
        "region": "Middle East",
        "company": "TII, UAE (via Together AI)",
        "model": "together_ai/tiiuae/falcon-11B",
        "env_key": "TOGETHER_API_KEY",
        "trainable": True,
    },
    # Optional extras -- chat-only, no fine-tuning path (see thesis notes).
    # Uncomment if you want them in your comparison anyway.
    # "gpt4o": {
    #     "region": "Western/US",
    #     "company": "OpenAI",
    #     "model": "gpt-4o",
    #     "env_key": "OPENAI_API_KEY",
    #     "trainable": False,
    # },
    # "claude": {
    #     "region": "Western/US",
    #     "company": "Anthropic",
    #     "model": "claude-sonnet-4-6",
    #     "env_key": "ANTHROPIC_API_KEY",
    #     "trainable": False,
    # },
}
