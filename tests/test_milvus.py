import logging

from wiz_rag.embeddings.embedding_client import get_embeddings_model
from wiz_rag.llm.llm_client import get_llm


def test_file_load():
    logging.info("test_rag start")

    from glob import glob
    text_lines = []

    for file_path in glob("data/milvus_docs/**/*.md"):
        with open(file_path, "r") as file:
            file_text = file.read()

        text_lines += file_text.split("# ")


def test_embeddings():
    embedding_model = get_embeddings_model()
    test_embedding = embedding_model.embed_query("This is a test")
    embedding_dim = len(test_embedding)
    print(embedding_dim)
    print(test_embedding[:10])


def test_collection():
    from pymilvus import MilvusClient

    milvus_client = MilvusClient(uri="./milvus_demo.db")
    collection_name = "my_rag_collection"

    if milvus_client.has_collection(collection_name=collection_name):
        milvus_client.drop_collection(collection_name=collection_name)
    milvus_client.create_collection(collection_name=collection_name, dimension=512, meric_type="IP",
                                    consistency_level="Strong")


def test_milvus_rag():
    # 导入数据
    from glob import glob
    text_lines = []

    for file_path in glob("../data/milvus_docs/en/faq/*.md"):
        try:
            with open(file_path, "r", encoding='utf-8') as file:
                file_text = file.read()

            # 改进分割逻辑
            chunks = file_text.split("# ")
            for i, chunk in enumerate(chunks):
                if chunk.strip():  # 过滤空内容
                    if i > 0:  # 给非第一个chunk重新加上标题标识
                        chunk = "# " + chunk
                    text_lines.append(chunk.strip())
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    # embedding 词向量
    embedding_model = get_embeddings_model()
    test_embedding = embedding_model.embed_query("This is a test")
    embedding_dim = len(test_embedding)

    # milvus 连接
    from pymilvus import MilvusClient

    milvus_client = MilvusClient(uri="./milvus_demo.db")
    collection_name = "my_rag_collection"

    if milvus_client.has_collection(collection_name=collection_name):
        milvus_client.drop_collection(collection_name=collection_name)

    # 修正拼写错误
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=embedding_dim,
        metric_type="IP",
        consistency_level="Strong"
    )

    # 向量化，插入数据
    from tqdm import tqdm
    data = []
    for i, line in enumerate(tqdm(text_lines, desc="Creating embeddings")):
        if line.strip():
            try:
                embedding = embedding_model.embed_query(line)
                data.append({"id": i, "vector": embedding, "text": line})
            except Exception as e:
                print(f"Error creating embedding for text {i}: {e}")

    if data:
        milvus_client.insert(collection_name=collection_name, data=data)

    # 向量检索查询
    query = "How is data stored in milvus?"
    query_embedding = embedding_model.embed_query(query)
    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[query_embedding],
        limit=3,
        search_params={"metric_type": "IP", "params": {}},
        output_fields=["text"],
    )
    import json
    retrieved_lines_with_distance = [(res["entity"]["text"], res["distance"]) for res in search_res[0]]
    print(json.dumps(retrieved_lines_with_distance, indent=4))

    # LLM 对检索出来的结果做总结
    chat_model = get_llm()
    context = "/n".join([line_with_distance[0] for line_with_distance in retrieved_lines_with_distance])
    SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {query}
    </question>
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT},
    ]
    response = chat_model.invoke(messages)
    print(response.content)
