from unstructured.partition.auto import partition


def test():
    # 自动解析 docx 文件为块
    elements = partition(filename="../data/unstructured/tender_document_001.docx")

    # 遍历每个块，识别块类型
    for element in elements:
        print(f"\n块类型: {type(element)}, 块内容: {element.text}")


def test_langchain():
    from langchain_unstructured import UnstructuredLoader
    file_path = [
        "../data/unstructured/tender_document_001.docx",
        # "../data/unstructured/test.docx",
    ]

    loader = UnstructuredLoader(file_path)
    docs = loader.load()
    print(docs[0])
