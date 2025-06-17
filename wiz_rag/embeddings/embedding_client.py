from langchain_huggingface import HuggingFaceEmbeddings

from wiz_rag.config import toml_config


# 获取 embeddings 模型
def get_embeddings_model():
    if toml_config.embedding.provider == "huggingface":
        return HuggingFaceEmbeddings(
            model_name=toml_config.embedding.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
    return None
