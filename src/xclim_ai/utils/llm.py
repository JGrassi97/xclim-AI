# xclim_tools.utils.llm
# =================================================

"""LLM and embedding initialization utilities.

Supported providers (select via ``config["credentials"]["provider"]``):
 - azure-openai
 - openai
 - ollama (local models served by the Ollama daemon)

Each provider exposes three factory functions after import:
 - ``initialize_llm(temperature=..., model=...)``  -> chat model for standard use
 - ``initialize_llm_rag(temperature=...)``         -> chat model dedicated to RAG (can be same as base)
 - ``initialize_embeddings()``                     -> embedding model client

Adding a new provider: replicate the pattern below and extend the final error message.
"""

from langchain_openai import (
    AzureChatOpenAI,
    ChatOpenAI,
    AzureOpenAIEmbeddings,
    OpenAIEmbeddings,
)
try:
    # Chat + Embeddings via local Ollama
    from langchain_ollama import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings
except Exception:  # pragma: no cover - optional dependency already in requirements but keep safe
    ChatOllama = None  # type: ignore
    OllamaEmbeddings = None  # type: ignore
from xclim_ai.utils.config import load_config

cfg = load_config()
provider = cfg["credentials"]["provider"]


###############################################################################
# Azure OpenAI setup
###############################################################################
if provider == "azure-openai":
    azure_key = cfg["azure-openai"]["azure_openai_api_key"]
    azure_endpoint = cfg["azure-openai"]["azure_openai_endpoint"]
    azure_version = cfg["azure-openai"]["azure_openai_api_version"]
    llm_model = cfg["azure-openai"]["azure_openai_llm_model"]
    rag_model = cfg["azure-openai"].get("azure_openai_llm_rag_model", llm_model)
    embedding_model = cfg["azure-openai"]["azure_openai_embedding_deployment_name"]

    def initialize_llm(temperature: float = 0.0, model: str = None) -> AzureChatOpenAI:
        """
        Initialize Azure OpenAI LLM client.
        """
        deployment = model or llm_model
        return AzureChatOpenAI(
            openai_api_version=azure_version,
            openai_api_key=azure_key,
            azure_endpoint=azure_endpoint,
            deployment_name=deployment,
            temperature=temperature,
        )

    def initialize_llm_rag(temperature: float = 0.0) -> AzureChatOpenAI:
        """
        Initialize Azure OpenAI LLM client for RAG usage.
        """
        return AzureChatOpenAI(
            openai_api_version=azure_version,
            openai_api_key=azure_key,
            azure_endpoint=azure_endpoint,
            deployment_name=rag_model,
            temperature=temperature,
        )

    def initialize_embeddings() -> AzureOpenAIEmbeddings:
        """
        Initialize Azure OpenAI embedding client.
        """
        return AzureOpenAIEmbeddings(
            openai_api_version=azure_version,
            api_key=azure_key,
            azure_endpoint=azure_endpoint,
            model=embedding_model,
        )


###############################################################################
# OpenAI setup
###############################################################################
elif provider == "openai":
    openai_key = cfg["openai"]["openai_api_key"]
    llm_model = cfg["openai"]["llm_model"]
    rag_model = cfg["openai"].get("llm_rag_model", llm_model)
    embedding_model = cfg["openai"]["embedding_model"]

    def initialize_llm(temperature: float = 0.0, model: str = None) -> ChatOpenAI:
        """
        Initialize OpenAI LLM client.
        """
        model_name = model or llm_model
        return ChatOpenAI(
            model_name=model_name,
            openai_api_key=openai_key
        )

    def initialize_llm_rag(temperature: float = 0.0) -> ChatOpenAI:
        """
        Initialize OpenAI LLM client for RAG usage.
        """
        return ChatOpenAI(
            model_name=rag_model,
            openai_api_key=openai_key
        )

    def initialize_embeddings() -> OpenAIEmbeddings:
        """
        Initialize OpenAI embedding client.
        """
        return OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_key,
        )


###############################################################################
# Ollama (local) setup
###############################################################################
elif provider == "ollama":
    if ChatOllama is None:
        raise ImportError(
            "ChatOllama not available. Ensure 'langchain-community' is installed."
        )

    ollama_cfg = cfg.get("ollama", {})
    base_url = ollama_cfg.get("base_url", "http://localhost:11434")
    llm_model = ollama_cfg.get("llm_model", "llama3.1:8b")
    rag_model = ollama_cfg.get("llm_rag_model", llm_model)
    embedding_model = ollama_cfg.get("embedding_model", "nomic-embed-text")

    def initialize_llm(temperature: float = 0.0, model: str | None = None):  # type: ignore
        """Initialize local Ollama chat model.

        Parameters
        ----------
        temperature : float
            Sampling temperature.
        model : str | None
            Override model name (defaults to config value).
        """
        model_name = model or llm_model
        return ChatOllama(model=model_name, base_url=base_url, temperature=temperature)

    def initialize_llm_rag(temperature: float = 0.0):  # type: ignore
        """Initialize Ollama chat model for RAG (can be same as base)."""
        return ChatOllama(model=rag_model, base_url=base_url, temperature=temperature)

    def initialize_embeddings():  # type: ignore
        """Initialize Ollama embedding model.

        Notes
        -----
        Requires the embedding model to be pulled locally, e.g.:
            ollama pull nomic-embed-text
        If unavailable, Ollama will attempt to pull it on first use.
        """
        if OllamaEmbeddings is None:
            raise ImportError(
                "OllamaEmbeddings not available. Ensure 'langchain-community' is installed."
            )
        return OllamaEmbeddings(model=embedding_model, base_url=base_url)

###############################################################################
# Gemini (Google) setup
###############################################################################
elif provider == "gemini":
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    except ImportError:
        raise ImportError("Install langchain-google-genai for Gemini support: pip install langchain-google-genai")
    gemini_cfg = cfg.get("gemini", {})
    gemini_api_key = gemini_cfg.get("gemini_api_key") or cfg.get("gemini_api_key")
    llm_model = gemini_cfg.get("llm_model", "models/gemini-1.5-pro-latest")
    embedding_model = gemini_cfg.get("embedding_model", "models/embedding-001")

    def initialize_llm(temperature: float = 0.0, model: str | None = None):
        model_name = model or llm_model
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=gemini_api_key,
            temperature=temperature,
        )

    def initialize_llm_rag(temperature: float = 0.0):
        return ChatGoogleGenerativeAI(
            model=llm_model,
            google_api_key=gemini_api_key,
            temperature=temperature,
        )

    def initialize_embeddings():
        return GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=gemini_api_key,
        )

###############################################################################
# Unknown provider
###############################################################################
else:
    raise ValueError(
        f"Unsupported provider: {provider}. Supported providers: azure-openai, openai, ollama, gemini"
    )