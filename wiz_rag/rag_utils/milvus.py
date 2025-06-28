from pymilvus import MilvusClient


# 获取 milvus 连接
def get_milvus_client(uri: str):
    return MilvusClient(uri=uri)
