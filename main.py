import logging

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from wiz_rag.config import toml_config
from wiz_rag.embeddings.distance import cosine_similarity, euclidean_distance
from wiz_rag.embeddings.embedding_client import get_embeddings_model
from wiz_rag.llm.llm_client import get_llm
from wiz_rag.utils.logger import LogMixin


# 使用装饰器为类自动注入 logger
class MyClass(LogMixin):
    def foo(self):
        self.logger.info("Hello, World!")


if __name__ == "__main__":
    my_class = MyClass()
    my_class.foo()
    logging.info(toml_config.llm.provider)
    logging.info(toml_config.llm.model_name)
    logging.info(toml_config.llm.api_key)

    # chat_model: BaseChatModel = get_llm()
    # result: BaseMessage = chat_model.invoke("你好，你叫什么名字")
    # logging.info(result.content)

    embeddings_model = get_embeddings_model()
    embeddings = embeddings_model.embed_query("你好，你叫什么名字")
    logging.info(embeddings)

    # 余弦相似度
    logging.info(cosine_similarity(embeddings[0], embeddings[3]))

    # 欧几里得距离
    logging.info(euclidean_distance(embeddings[0], embeddings[3]))

    # 聚类
    texts = ['苹果', '菠萝', '西瓜', '斑马', '大象', '老鼠']
    embeddings = embeddings_model.embed_documents(texts)
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(embeddings)
    label = kmeans.labels_
    for i in range(len(texts)):
        logging.info(f"cls({texts[i]}) = {label[i]}")
