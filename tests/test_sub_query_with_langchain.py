# We use the Langchain WebBaseLoader to load documents from web sources and split them into chunks using the RecursiveCharacterTextSplitter.

import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from wiz_rag.rag_utils.sub_query import SubQueryRetriever
from wiz_rag.rag_utils.vanilla import vectorstore, format_doc, rag_prompt, llm


def test():
    # Create a WebBaseLoader instance to load documents from web sources
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )

    # Load documents from web sources using the loader
    documents = loader.load()

    # Initialize a RecursiveCharacterTextSplitter for splitting text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

    # Split the documents into chunks using the text_splitter
    docs = text_splitter.split_documents(documents)

    # Build the chain
    # We load the docs into milvus vectorstore, and build a milvus retriever
    vectorstore.add_documents(docs)
    retriever = vectorstore.as_retriever()

    # Define the vanilla RAG chain.
    vanilla_rag_chain = (
            {"context": retriever | format_doc, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
    )

    # Define the sub query chain.
    sub_query_retriever = SubQueryRetriever.from_vectorstore(vectorstore)
    sub_query_chain = (
            {"context": sub_query_retriever | format_doc, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
    )

    # Test the chain
    query = "What is the difference between Short-Term Memory and Long-Term Memory?"

    vanilla_res = vanilla_rag_chain.invoke(query)
    print("vanilla_res:", vanilla_res)
    print("\n")

    sub_query_res = sub_query_chain.invoke(query)
    print("sub_query_res:", sub_query_res)
