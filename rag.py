import chromadb
from chromadb.config import Settings
from typing import List, Dict
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from config import EMBEDDING_MODEL

class RAGSystem:
    
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = None):
        if persist_directory is None:
            persist_directory = "./chroma_db"
        
        os.makedirs(persist_directory, exist_ok=True)
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.vectorstore = None
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def index_documents(self, documents: Dict[str, str], chunk_size: int, chunk_overlap: int, clear_existing: bool = False):
        if clear_existing:
            self.clear_collection()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        loaded_documents = []
        
        for source, text in documents.items():
            doc = Document(
                page_content=text,
                metadata={"url": source}
            )
            loaded_documents.append(doc)
        
        chunked_documents = text_splitter.split_documents(loaded_documents)
        
        ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        
        try:
            self.vectorstore = Chroma.from_documents(
                documents=chunked_documents,
                embedding=ollama_embeddings,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                migrations=False
            )
            print(f"✓ Indexados {len(chunked_documents)} chunks de {len(documents)} documentos")
        except Exception as e:
            error_msg = str(e)
            if "already exists" in error_msg or "different settings" in error_msg:
                print(f"  ⚠️  Instância existente detectada. Limpando e recriando...")
                self.clear_collection()
                self.vectorstore = Chroma.from_documents(
                    documents=chunked_documents,
                    embedding=ollama_embeddings,
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name,
                    migrations=False
                )
                print(f"✓ Indexados {len(chunked_documents)} chunks de {len(documents)} documentos (após limpar)")
            else:
                print(f"Erro ao indexar documentos: {e}")
                raise
    
    def retrieve_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        if self.vectorstore is None:
            ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=ollama_embeddings,
                collection_name=self.collection_name
            )
        
        results = self.vectorstore.similarity_search_with_score(
            query,
            k=n_results
        )
        
        retrieved_docs = []
        for doc, score in results:
            retrieved_docs.append({
                'document': doc.page_content,
                'metadata': doc.metadata,
                'distance': score
            })
        
        return retrieved_docs
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        if not retrieved_docs:
            return "Nenhum documento relevante encontrado."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            # Pega a URL ou caminho do arquivo
            source = doc['metadata'].get('url', 'Fonte desconhecida')
            # Se for um caminho de arquivo, mostra apenas o nome do arquivo
            if os.path.exists(source) or source.endswith('.txt'):
                source = os.path.basename(source)
            content = doc['document']
            context_parts.append(f"[Documento {i} - Fonte: {source}]\n{content}\n")
        
        return "\n".join(context_parts)
    
    def clear_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        
        self.vectorstore = None
        
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        print("✓ Coleção limpa com sucesso")
    
    def get_collection_info(self) -> Dict:
        if self.vectorstore is None:
            ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
            self.vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=ollama_embeddings,
                collection_name=self.collection_name
            )
        
        count = self.vectorstore._collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count
        }
