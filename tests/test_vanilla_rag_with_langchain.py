# This guide demonstrates how to build a Retrieval-Augmented Generation (RAG) system using LangChain and Milvus.
# The RAG system combines a retrieval system with a generative model to generate new text based on a given prompt.
# The system first retrieves relevant documents from a corpus using Milvus, and then uses a generative model to generate new text based on the retrieved documents.
# LangChain is a framework for developing applications powered by large language models (LLMs).
# Milvus is the world's most advanced open-source vector database, built to power embedding similarity search and AI applications.
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    # Let's take a look at the first document
    print(docs[1])

    # As we can see, the document is already split into chunks. And the content of the data is about the AI agent.

    # Initialize a Milvus vector store with the documents
    vectorstore.add_documents(docs)

    # vector similarity search
    query = "What is self-reflection of an AI Agent?"
    similar_docs = vectorstore.similarity_search(query, k=3)
    print(similar_docs)

    # convert vector store to retriever
    retriever = vectorstore.as_retriever()

    # Define the RAG chain for AI response generation
    rag_chain = (
            {"context": retriever | format_doc, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
    )

    res = rag_chain.invoke(query)
    print("\n")
    print(res)
