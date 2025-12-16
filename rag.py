import chromadb
from chromadb.config import Settings
from typing import List, Dict
import hashlib
import os

class RAGSystem:
    
    def __init__(self, collection_name: str = "rag_documents", persist_directory: str = None):
        if persist_directory is None:
            persist_directory = "./chroma_db"
        
        os.makedirs(persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        self.collection_name = collection_name
    
    def _create_chunks(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        # Validação: overlap deve ser menor que chunk_size
        if chunk_overlap >= chunk_size:
            raise ValueError(f"chunk_overlap ({chunk_overlap}) deve ser menor que chunk_size ({chunk_size})")
        
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - chunk_overlap
            
            # Evita loop infinito se overlap for muito grande
            if start >= len(text) - chunk_overlap:
                break
        
        if start < len(text):
            chunks.append(text[start:])
        
        return chunks
    
    def _generate_id(self, url: str, chunk_index: int) -> str:
        content = f"{url}_{chunk_index}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def index_documents(self, documents: Dict[str, str], chunk_size: int, chunk_overlap: int, clear_existing: bool = False):
        if clear_existing:
            self.clear_collection()
        
        # Garante que a coleção existe antes de adicionar documentos
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        except Exception as e:
            print(f"Erro ao garantir coleção: {e}")
            raise
        
        all_chunks = []
        all_ids = []
        all_metadatas = []
        
        for source, text in documents.items():
            chunks = self._create_chunks(text, chunk_size, chunk_overlap)
            
            for idx, chunk in enumerate(chunks):
                chunk_id = self._generate_id(source, idx)
                all_chunks.append(chunk)
                all_ids.append(chunk_id)
                # Usa 'url' como chave mesmo para arquivos locais (mantém compatibilidade)
                all_metadatas.append({
                    "url": source,
                    "chunk_index": idx,
                    "total_chunks": len(chunks)
                })
        
        if all_chunks:
            # Adiciona documentos em batches para evitar erro de batch size
            batch_size = 4000  # Limite seguro para ChromaDB
            total_batches = (len(all_chunks) + batch_size - 1) // batch_size
            
            try:
                for i in range(0, len(all_chunks), batch_size):
                    batch_chunks = all_chunks[i:i + batch_size]
                    batch_ids = all_ids[i:i + batch_size]
                    batch_metadatas = all_metadatas[i:i + batch_size]
                    
                    batch_num = (i // batch_size) + 1
                    if total_batches > 1:
                        print(f"  Adicionando batch {batch_num}/{total_batches} ({len(batch_chunks)} chunks)...")
                    
                self.collection.add(
                        documents=batch_chunks,
                        ids=batch_ids,
                        metadatas=batch_metadatas
                )
                
                print(f"✓ Indexados {len(all_chunks)} chunks de {len(documents)} documentos")
            except Exception as e:
                print(f"Erro ao adicionar documentos: {e}")
                # Tenta recriar a coleção e tentar novamente com batches menores
                try:
                    self.collection = self.client.get_or_create_collection(
                        name=self.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    
                    # Tenta com batch ainda menor
                    smaller_batch_size = 2000
                    for i in range(0, len(all_chunks), smaller_batch_size):
                        batch_chunks = all_chunks[i:i + smaller_batch_size]
                        batch_ids = all_ids[i:i + smaller_batch_size]
                        batch_metadatas = all_metadatas[i:i + smaller_batch_size]
                        
                    self.collection.add(
                            documents=batch_chunks,
                            ids=batch_ids,
                            metadatas=batch_metadatas
                    )
                    
                    print(f"✓ Indexados {len(all_chunks)} chunks de {len(documents)} documentos (após recriar coleção)")
                except Exception as e2:
                    print(f"Erro ao adicionar documentos após recriar coleção: {e2}")
                    raise
    
    def retrieve_documents(self, query: str, n_results: int = 3) -> List[Dict]:
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        retrieved_docs = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for i in range(len(results['documents'][0])):
                doc = {
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                    'distance': results['distances'][0][i] if results['distances'] else None
                }
                retrieved_docs.append(doc)
        
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
            # Tenta deletar a coleção se existir
            try:
                self.client.delete_collection(name=self.collection_name)
            except Exception:
                # Se não existir, não é problema
                pass
            
            # Recria a coleção garantindo que ela existe
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Verifica se a coleção foi criada corretamente
            if self.collection.count() > 0:
                # Se ainda tem documentos, limpa todos
                all_ids = self.collection.get()['ids']
                if all_ids:
                    self.collection.delete(ids=all_ids)
            
            print("✓ Coleção limpa com sucesso")
        except Exception as e:
            print(f"Erro ao limpar coleção: {e}")
            # Tenta recriar a coleção mesmo em caso de erro
            try:
                self.collection = self.client.get_or_create_collection(
                    name=self.collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
            except Exception as e2:
                print(f"Erro ao recriar coleção: {e2}")
                raise
    
    def get_collection_info(self) -> Dict:
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "total_chunks": count
        }
