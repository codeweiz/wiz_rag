from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_milvus import Milvus

from wiz_rag.embeddings.embedding_client import get_embeddings_model
from wiz_rag.llm.llm_client import get_llm

# 提示词模板
PROMPT_TEMPLATE = """
Human: You are an AI assistant, and provides answers to questions by using fact based and statistical information when possible.
Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
<context>
{context}
</context>

<question>
{question}
</question>

The response should be specific and use statistics or numbers when possible.

Assistant:
"""

# RAG 提示词
rag_prompt = PromptTemplate(
    template=PROMPT_TEMPLATE, input_variables=["context", "question"]
)

# 词嵌入模型
embeddings = get_embeddings_model()

# 大语言模型
llm = get_llm()

# 向量数据库
vectorstore = Milvus(
    embedding_function=embeddings,
    connection_args={"uri": "./milvus.db"},
    auto_id=True,
    drop_old=True
)


# Document List to str
def format_doc(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)
