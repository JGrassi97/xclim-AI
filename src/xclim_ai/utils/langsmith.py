# xclim_tools.utils.langsmith
# =================================================

# LangSmith environment initialization utility.
# Sets environment variables needed for LangChain/LangSmith integration.

import os
from xclim_ai.utils.config import load_config

cfg = load_config()
langsmith_enabled = cfg["credentials"].get("langsmith", False)

if langsmith_enabled:
    langsmith_cfg = cfg.get("langsmith", {})
    langsmith_key = langsmith_cfg.get("langsmith_api_key", "")
    langchain_tracing = langsmith_cfg.get("lagchain_tracing_v2", "true")
    langchain_project = langsmith_cfg.get("lagchain_project", "xclim")
    langchain_endpoint = langsmith_cfg.get("lagchain_endpoint", "")


def initialize_langsmith() -> None:
    """
    Set LangSmith-related environment variables from config.
    This enables advanced LangChain tracing, logging, and debugging.
    """
    if not langsmith_enabled:
        return

    os.environ["LANGSMITH_API_KEY"] = langsmith_key
    os.environ["LANGCHAIN_TRACING_V2"] = langchain_tracing
    os.environ["LANGCHAIN_ENDPOINT"] = langchain_endpoint
    os.environ["LANGCHAIN_PROJECT"] = langchain_project