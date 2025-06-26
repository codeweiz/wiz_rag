import textwrap

from datasets import load_dataset
from pymilvus import MilvusClient, DataType
from tqdm import tqdm

from wiz_rag.embeddings.embedding_client import get_embeddings_model


def test():
    collection_name = "movie_search"
    dimension = 768
    batch_size = 1000

    # connect to milvus database
    client = MilvusClient("./milvus_demo.db")

    # remove collection if it already exists
    if client.has_collection(collection_name=collection_name):
        client.drop_collection(collection_name=collection_name)

    # create schema
    schema = MilvusClient.create_schema(
        auto_id=True,
        enable_dynamic_field=False,
    )

    # add fields to schema
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=64000)
    schema.add_field(field_name="type", datatype=DataType.VARCHAR, max_length=64000)
    schema.add_field(field_name="release_year", datatype=DataType.INT64)
    schema.add_field(field_name="rating", datatype=DataType.VARCHAR, max_length=64000)
    schema.add_field(field_name="description", datatype=DataType.VARCHAR, max_length=64000)
    schema.add_field(field_name="embedding", datatype=DataType.FLOAT_VECTOR, dim=dimension)

    # create collection with the schema
    client.create_collection(collection_name=collection_name, schema=schema)

    # create the index on the collection and load it
    # prepare index parameters
    index_params = client.prepare_index_params()

    # add an index on the embedding field
    index_params.add_index(
        field_name="embedding",
        metric_type="IP",
        index_type="AUTOINDEX",
        params={}
    )

    # create index
    client.create_index(collection_name=collection_name, index_params=index_params)

    # load collection
    client.load_collection(collection_name=collection_name, replica_number=1)

    # load dataset
    dataset = load_dataset("hugginglearners/netflix-shows", split="train")

    # get embedding model
    embedding_model = get_embeddings_model()

    # insert data
    batch = []
    for i in tqdm(range(0, len(dataset))):
        batch.append({
            "title": dataset[i]["title"] or "",
            "type": dataset[i]["type"] or "",
            "release_year": dataset[i]["release_year"] or -1,
            "rating": dataset[i]["rating"] or "",
            "description": dataset[i]["description"] or "",
        })

        if len(batch) % batch_size == 0 or i == len(dataset) - 1:
            embeddings = embedding_model.embed_documents([item["description"] for item in batch])

            for item, emb in zip(batch, embeddings):
                item["embedding"] = emb

            client.insert(collection_name=collection_name, data=batch)
            batch = []

    # start query
    query = "movie about a fluffly animal"
    expr = 'release_year < 2019 and rating like "PG%"'
    top_k = 5

    res = client.search(
        collection_name=collection_name,
        data=[embedding_model.embed_query(query)],
        filter=expr,
        limit=top_k,
        output_fields=["title", "type", "release_year", "rating", "description"],
        search_params={
            "metric_type": "IP",
            "params": {}
        }
    )

    for hit_group in res:
        print("Results:")
        for rank, hit in enumerate(hit_group, start=1):
            entity = hit["entity"]
            print(f"\tRank: {rank} Score: {hit['distance']:} Title: {entity.get('title', '')}")
            print(
                f"\t\tType: {entity.get('type', '')}"
                f"Release Year: {entity.get('release_year', '')}"
                f"Rating: {entity.get('rating', '')}"
            )
            description = entity.get("description", "")
            print(textwrap.fill(description, width=88))
            print()


def test_query():
    collection_name = "movie_search"

    # connect to milvus database
    client = MilvusClient("./milvus_demo.db")

    # start query
    query = "The Shawshank Redemption"
    expr = 'release_year < 2019'
    top_k = 5

    # get embedding model
    embedding_model = get_embeddings_model()

    res = client.search(
        collection_name=collection_name,
        data=[embedding_model.embed_query(query)],
        filter=expr,
        limit=top_k,
        output_fields=["title", "type", "release_year", "rating", "description"],
        search_params={
            "metric_type": "IP",
            "params": {}
        }
    )

    for hit_group in res:
        print("Results:")
        for rank, hit in enumerate(hit_group, start=1):
            entity = hit["entity"]
            print(
                f"Title: {entity.get('title', '')}\t"
                f"Rank: {rank} Score: {hit['distance']:}\t"
                f"Type: {entity.get('type', '')}\t"
                f"Release Year: {entity.get('release_year', '')}\t"
                f"Rating: {entity.get('rating', '')}\t"
            )
            description = entity.get("description", "")
            print(textwrap.fill(description, width=88))
            print()
