# 使用 Milvus 进行文本到图像搜索
# 文本到图像搜索是一种先进的技术，允许用户使用自然语言文本描述搜索图像。
# 它利用预训练的多模态模型将文本和图像转换为共享语义空间中的 Embeddings，从而实现基于相似度的比较。
from PIL import Image

from wiz_rag.clip_utils.clip import get_clip_model, encode_image, encode_text
from wiz_rag.rag_utils.milvus import get_milvus_client

milvus_db_uri = "./milvus_demo.db"
collection_name = "text_image_search"


# 将图片向量化，插入向量库中
def test_insert_image():
    # milvus 连接
    milvus_client = get_milvus_client(uri=milvus_db_uri)

    # 数据输入
    if milvus_client.has_collection(collection_name=collection_name):
        milvus_client.drop_collection(collection_name=collection_name)

    # create a new collection in quickstart mode
    milvus_client.create_collection(
        collection_name=collection_name,
        dimension=512,
        auto_id=True,
        enable_dynamic_field=True,
    )

    # insert images
    import os
    from glob import glob
    image_dir = "../data/images_folder/train"
    raw_data = []

    # load clip model
    model, preprocess = get_clip_model()

    for image_path in glob(os.path.join(image_dir, "**/*.JPEG")):
        image_embedding = encode_image(image_path, model, preprocess)
        image_dict = {"vector": image_embedding, "image_path": image_path}
        raw_data.append(image_dict)
    insert_result = milvus_client.insert(collection_name=collection_name, data=raw_data)
    print("Inserted", insert_result["insert_count"], "images to Milvus.")


# 执行文本搜索
def test_search():
    # milvus 连接
    milvus_client = get_milvus_client(uri=milvus_db_uri)

    # load clip model
    model, preprocess = get_clip_model()

    query_text = "a white dog"
    query_embedding = encode_text(query_text, model, preprocess)

    search_result = milvus_client.search(
        collection_name=collection_name,
        data=[query_embedding],
        limit=10,
        output_fields=["image_path"]
    )
    # print("Search result:", search_result)

    # visual result
    width = 150 * 5
    height = 150 * 2
    concatenated_image = Image.new("RGB", (width, height))

    result_images = []
    for result in search_result:
        for hit in result:
            filename = hit["entity"]["image_path"]
            img = Image.open(filename)
            img = img.resize((150, 150))
            result_images.append(img)

    for idx, img in enumerate(result_images):
        x = idx % 5
        y = idx // 5
        concatenated_image.paste(img, (x * 150, y * 150))

    print("Query text:", query_text)
    print("\nSearch results:")
    concatenated_image.show()
