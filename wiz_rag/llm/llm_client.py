from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from wiz_rag.config import toml_config


# 获取 LLM 客户端
def get_llm(provider: str = None, model_name: str = None, api_key: str = None, **kwargs) -> BaseChatModel:
    if provider is None:
        provider = toml_config.llm.provider
    if model_name is None:
        model_name = toml_config.llm.model_name
    if api_key is None:
        api_key = toml_config.llm.api_key

    return init_chat_model(model_provider=provider, model=model_name, api_key=api_key, **kwargs)
