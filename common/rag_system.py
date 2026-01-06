import os
from typing import List, Dict, Union
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from common.config import EMBEDDING_MODEL

def get_embedding_model_name() -> str:
    return EMBEDDING_MODEL if EMBEDDING_MODEL else "mahonzhan/all-MiniLM-L6-v2"

def initialize_chromadb(documents: List[Document], model_name: str, chunk_size: int, chunk_overlap: int, persist_directory: str) -> Chroma:
    from common.document_processor import split_documents
    texts = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedding_model_name = get_embedding_model_name()
    embeddings = OllamaEmbeddings(model=embedding_model_name)
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    if texts:
        vectorstore.add_documents(texts)
    
    return vectorstore

def load_chromadb(model_name: str, persist_directory: str) -> Chroma:
    if not os.path.exists(persist_directory):
        return None
    
    embedding_model_name = get_embedding_model_name()
    embeddings = OllamaEmbeddings(model=embedding_model_name)
    return Chroma(persist_directory=persist_directory, embedding_function=embeddings)

def create_retrieval_chain(vectorstore: Chroma, model_name: str, search_kwargs: Dict, prompt_template: str):
    llm = OllamaLLM(model=model_name)
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
        }
    )
    return qa_chain

def get_chromadb_documents(vectorstore: Chroma) -> List[Union[str, Dict]]:
    docs = vectorstore._collection.get()
    documentos = []
    documentos_processados = set()
    
    for metadata in docs["metadatas"]:
        source = metadata.get("source", "")
        if source and source not in documentos_processados:
            if metadata.get("type") == "url":
                documentos.append(source)
            else:
                documentos.append({
                    "name": source,
                    "type": metadata.get("type", "unknown"),
                    "text": ""
                })
            documentos_processados.add(source)
    
    return documentos
