# xclim_tools.utils.llm
# =================================================

# Initialization of LLMs and embeddings for supported providers:
# - Azure OpenAI
# - OpenAI (standard)
# The selected backend is controlled via config["credentials"]["provider"]

from langchain_openai import (
    AzureChatOpenAI,
    ChatOpenAI,
    AzureOpenAIEmbeddings,
    OpenAIEmbeddings,
)
from xclim_ai.utils.config import load_config

cfg = load_config()
provider = cfg["credentials"]["provider"]


# Azure OpenAI setup
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


# OpenAI setup
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


# Unknown provider
else:
    raise ValueError(
        f"Unsupported provider: {provider}. "
        "Supported providers are 'azure-openai' and 'openai'."
    )