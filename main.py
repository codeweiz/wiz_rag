import logging

import chromadb
from attr.validators import max_len
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage

from wiz_rag.config import toml_config
from wiz_rag.embeddings.distance import cosine_similarity, euclidean_distance
from wiz_rag.embeddings.embedding_client import get_embeddings_model
from wiz_rag.embeddings.embedding_function import embedding_function
from wiz_rag.llm.llm_client import get_llm
from wiz_rag.utils.logger import LogMixin


# 使用装饰器为类自动注入 logger
class MyClass(LogMixin):
    def foo(self):
        self.logger.info("Hello, World!")


def test_log():
    my_class = MyClass()
    my_class.foo()
    logging.info(toml_config.llm.provider)
    logging.info(toml_config.llm.model_name)
    logging.info(toml_config.llm.api_key)


def test_chat_model():
    chat_model: BaseChatModel = get_llm()
    result: BaseMessage = chat_model.invoke("你好，你叫什么名字")
    logging.info(result.content)


def test_embedding():
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


def test_chroma():
    # 创建 Chroma 集合
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="rag_db", embedding_function=embedding_function,
                                                 metadata={"hnsw:space": "cosine"})
    documents = [
        "在向量搜索领域，我们拥有多种索引方法和向量处理技术，它们使我们能够在召回率、响应时间和内存使用之间做出权衡。",
        "虽然单独使用特定技术如倒排文件（IVF）、乘积量化（PQ）或分层导航小世界（HNSW）通常能够带来满意的结果，",
        "GraphRAG 本质上就是 RAG，只不过于一般的 RAG 相比，其检索路径上多了个知识图谱"]
    collection.add(documents=documents, ids=["id1", "id2", "id3"],
                   metadatas=[{"chapter": 3, "verse": 16}, {"chapter": 4, "verse": 5}, {"chapter": 12, "verse": 5}])
    count = collection.count()
    logging.info(count)

    get_collection = chroma_client.get_collection(name="rag_db", embedding_function=embedding_function, )
    id_result = get_collection.get(ids=['id2'], include=["documents", "embeddings", "metadatas"])
    logging.info(id_result)

    query = "索引技术有哪些？"
    result = get_collection.query(query_texts=query, n_results=2, include=["documents", "metadatas"])
    logging.info(result)

    result = get_collection.query(query_texts=query, n_results=2, include=["documents", "metadatas"],
                                  where={"verse": 5})
    logging.info(result)


def test_milvus():
    logging.info("Milvus")
    import numpy as np
    from pymilvus import (connections, utility, FieldSchema, CollectionSchema, DataType, Collection)

    connections.connect(host="127.0.0.1", port=19530)
    if utility.has_collection("rag_db"):
        utility.drop_collection("rag_db")

    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
        FieldSchema(name="documents", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=512),
        FieldSchema(name="verse", dtype=DataType.INT64)
    ]

    rag_db = Collection("rag_db", CollectionSchema(fields), consistency_level="Strong")

    documents = [
        "在向量搜索领域，我们拥有多种索引方法和向量处理技术，它们使我们能够在召回率、响应时间和内存使用之间做出权衡。",
        "虽然单独使用特定技术如倒排文件（IVF）、乘积量化（PQ）或分层导航小世界（HNSW）通常能够带来满意的结果，",
        "GraphRAG 本质上就是 RAG，只不过于一般的 RAG 相比，其检索路径上多了个知识图谱"]

    embeddings_model = get_embeddings_model()
    embeddings = embeddings_model.embed_documents(documents)
    data = [
        [str(i) for i in range(len(documents))],
        documents,
        np.array(embeddings),
        [16, 5, 5]
    ]
    rag_db.insert(data)
    rag_db.flush()
    logging.info(rag_db.num_entities)

    # 构建索引
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 128}
    }
    rag_db.create_index("embeddings", index_params)
    rag_db.load()

    # 搜索
    get_collection = Collection("rag_db")
    get_collection.load()
    query = "索引技术有哪些？"
    query_emb = embeddings_model.embed_query(query)
    search_params = {
        "metric_type": "L2"
    }
    results = get_collection.search(
        [query_emb],
        "embeddings",
        param=search_params,
        limit=2,
        output_fields=["documents", "verse"]
    )
    logging.info(results)


def test_milvus_lite():
    from pymilvus import MilvusClient, model

    client = MilvusClient()

    if client.has_collection(collection_name="demo_collection"):
        client.drop_collection(collection_name="demo_collection")
    client.create_collection(collection_name="demo_collection", dimension=768)

    embedding_fn = model.DefaultEmbeddingFunction()
    docs = [
        "Artificial intelligence was founded as an academic discipline in 1956.",
        "Alan Turing was the first person to conduct substantial research in AI.",
        "Born in Maida Vale, London, Turing was raised in southern England."
    ]
    vectors = embedding_fn.encode_documents(docs)
    print("Dim:", embedding_fn.dim, vectors[0].shape)

    data = [
        {"id": i, "vector": vectors[i], "text": docs[i], "subject": "history"}
        for i in range(len(vectors))
    ]
    print("Data has", len(data), "entities, each with fields: ", data[0].keys())
    print("Vector dim:", len(data[0]["vector"]))

    res = client.insert(collection_name="demo_collection", data=data)
    print(res)

    query_vectors = embedding_fn.encode_queries(["Who is Alan Turing?"])
    res = client.search(
        collection_name="demo_collection",
        data=query_vectors,
        limit=2,
        output_fields=["text", "subject"]
    )
    print("向量查询：", res)

    res = client.query(
        collection_name="demo_collection",
        filter="subject == 'history'",
        output_fields=["text", "subject"]
    )
    print("标量查询：", res)

    res = client.query(
        collection_name="demo_collection",
        ids=[0, 2],
        output_fields=["text", "text", "subject"]
    )
    print("ID 查询：", res)


def test_index():
    import numpy as np
    from scipy.cluster.vq import kmeans2

    # 1. 随机生成一个128维的查询向量
    query = np.random.normal(size=(128,))
    # 2. 随机生成1000个128维的向量作为“数据集”
    dataset = np.random.normal(size=(1000, 128))

    # --- Flat Index（暴力检索，遍历所有向量）---
    # 3. 计算query与每个向量的欧氏距离，返回距离最小的下标
    result1 = np.argmin(np.linalg.norm(query - dataset, axis=1))
    logging.info(result1)

    # --- IVF Index（倒排索引，常见加速方法）---
    num_part = 100  # 假设分成100个“桶”（聚类中心）
    # 4. 用kmeans聚类，把dataset分成100个聚类，得到每个向量的分配结果
    centroids, assignments = kmeans2(dataset, num_part, iter=1000)
    logging.info(centroids.shape)  # (100, 128)
    logging.info(assignments.shape)  # (1000,)

    # 5. 按聚类分桶，把每个向量分配到对应的桶
    index = [[] for _ in range(num_part)]
    for n, k in enumerate(assignments):
        index[k].append(n)

    # 6. 先找query最近的聚类中心
    result_id = np.argmin(np.linalg.norm(query - centroids, axis=1))
    # 7. 在该聚类中心的桶里找query最近的向量
    result2 = np.argmin(np.linalg.norm(query - dataset[index[result_id]], axis=1))
    logging.info(result2)

    # 8. 多桶检索：找query最近的100个聚类中心，把这些桶里的所有向量合起来
    result_ids = np.argsort(np.linalg.norm(query - centroids, axis=1))[:100]
    top3_index = []
    for c in result_ids:
        top3_index += index[c]
    # 9. 在这10个桶里的所有向量里找最近的
    result2 = np.argmin(np.linalg.norm(query - dataset[top3_index], axis=1))
    logging.info(top3_index[result2])


if __name__ == "__main__":
    logging.info("main")
