from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

import os
def load_pdf(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents

extracted_data = load_pdf("data")
print(len(extracted_data))
print(extracted_data[0].page_content[:500])

def filter_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []

    for doc in docs:
        src = doc.metadata.get("source")  # metadata key is usually lowercase

        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )

    return minimal_docs

def text_splitter(minimal_docs):
    text_splitter=RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=26,
        length_function=len
    ) 
    texts= text_splitter.split_documents(minimal_docs)
    return texts

