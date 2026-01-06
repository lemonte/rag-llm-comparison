MODELOS_EXECUTAR = ["mistral"]
CHUNK_SIZES = [1000 ]
CHUNK_OVERLAPS = [ 200]
K_DOCUMENTS_LIST = [3]

MODELO_REFERENCIA = "gpt-4o-mini"
CHUNK_SIZE_REFERENCIA = 1000
CHUNK_OVERLAP_REFERENCIA = 100
K_DOCUMENTS_REFERENCIA = 3

import time
import csv
import json
import numpy as np
from datetime import datetime
from itertools import product
from openai import OpenAI
from langchain_ollama import OllamaEmbeddings
from common.config import (
    DOCUMENT_URLS,
    LOCAL_FILES,
    PERGUNTAS_TESTE_2,
    PROMPT_TEMPLATE,
    OPENAI_API_KEY,
    OPENAI_VECTOR_STORE_ID,
    OPENAI_FILE_SEARCH_MODEL,
    EMBEDDING_MODEL,
    get_chromadb_folder
)
from common.document_processor import fetch_and_process_urls
from common.rag_system import initialize_chromadb, load_chromadb, create_retrieval_chain

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

_embedding_model_cache = None

def get_embedding_model():
    global _embedding_model_cache
    if _embedding_model_cache is None:
        model_name = EMBEDDING_MODEL if EMBEDDING_MODEL else "mahonzhan/all-MiniLM-L6-v2"
        _embedding_model_cache = OllamaEmbeddings(model=model_name)
    return _embedding_model_cache

def _extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text is not None:
        return str(output_text).strip()
    
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks).strip()

def _serialize_result(result) -> dict:
    if hasattr(result, "__dict__"):
        return {k: str(v) for k, v in result.__dict__.items()}
    if hasattr(result, "model_dump"):
        return result.model_dump()
    return {"content": str(result)}

def execute_openai_file_search_query(question: str) -> tuple:
    start_time = time.time()
    
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY não está configurada.")
    
    response = openai_client.responses.create(
        model=OPENAI_FILE_SEARCH_MODEL,
        input=question,
        tools=[{
            "type": "file_search",
            "vector_store_ids": [OPENAI_VECTOR_STORE_ID],
            "max_num_results": K_DOCUMENTS_REFERENCIA
        }],
        include=["file_search_call.results"]
    )
    
    latency = time.time() - start_time
    answer = _extract_response_text(response)
    
    results = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "file_search_call":
            continue
        raw_results = getattr(item, "results", None) or []
        for result in raw_results:
            results.append(_serialize_result(result))
    
    return answer, results, latency

def _cosine_similarity(a: list, b: list) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)

def embed_texts(texts: list) -> list:
    embeddings_model = get_embedding_model()
    return embeddings_model.embed_documents(texts)

def main():
    print("=" * 80)
    print("Etapa 3 - Comparação com OpenAI File Search")
    print("=" * 80)
    
    if not openai_client:
        print("Erro: OPENAI_API_KEY não está configurada.")
        return
    
    documents_list = [url for url in DOCUMENT_URLS] + [file for file in LOCAL_FILES]
    
    if not documents_list:
        print("Erro: Nenhum documento encontrado para processar")
        return
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"avaliacao_rag_{timestamp}.csv"
    compare_csv_filename = f"comparacao_openai_mistral_{timestamp}.csv"
    
    fieldnames = [
        "numero_questao",
        "pergunta",
        "resposta",
        "latency_segundos",
        "num_docs_recuperados",
        "avg_proximidade",
        "proximidades",
    ]
    
    compare_fieldnames = [
        "numero_questao",
        "pergunta",
        "modelo_mistral",
        "modelo_openai",
        "vector_store_id",
        "embedding_model",
        "resposta_mistral",
        "resposta_gpt",
        "similaridade_cosseno",
        "latencia_mistral_segundos",
        "latencia_openai_segundos",
        "openai_num_resultados",
        "openai_resultados",
    ]
    
    csv_file = open(csv_filename, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()
    
    compare_csv_file = open(compare_csv_filename, "w", newline="", encoding="utf-8")
    compare_writer = csv.DictWriter(compare_csv_file, fieldnames=compare_fieldnames)
    compare_writer.writeheader()
    compare_csv_file.flush()
    
    combinations = list(product(MODELOS_EXECUTAR, CHUNK_SIZES, CHUNK_OVERLAPS, K_DOCUMENTS_LIST))
    
    print(f"Total de combinações Ollama: {len(combinations)}")
    print(f"Configuração de referência OpenAI: {MODELO_REFERENCIA}, Chunk: {CHUNK_SIZE_REFERENCIA}, Overlap: {CHUNK_OVERLAP_REFERENCIA}, K: {K_DOCUMENTS_REFERENCIA}")
    print()
    
    for idx, (model, chunk_size, chunk_overlap, k_docs) in enumerate(combinations, 1):
        print(f"[{idx}/{len(combinations)}] Modelo: {model}, Chunk: {chunk_size}, Overlap: {chunk_overlap}, K: {k_docs}")
        print("-" * 80)
        
        if chunk_overlap >= chunk_size:
            print(f"Pulando: overlap ({chunk_overlap}) >= chunk_size ({chunk_size})")
            continue
        
        persist_dir = get_chromadb_folder(model, chunk_size, k_docs, chunk_overlap)
        
        vectorstore = load_chromadb(model, persist_dir)
        
        if not vectorstore:
            print("Inicializando ChromaDB...")
            documents, num_docs_processed = fetch_and_process_urls(documents_list)
            
            if not documents:
                print("Erro: Nenhum documento processado")
                continue
            
            vectorstore = initialize_chromadb(documents, model, chunk_size, chunk_overlap, persist_dir)
            print(f"Documentos indexados: {num_docs_processed}")
        else:
            print("ChromaDB carregado do disco")
        
        search_kwargs = {"k": k_docs}
        qa_chain = create_retrieval_chain(vectorstore, model, search_kwargs, PROMPT_TEMPLATE)
        
        for pergunta_idx, item_pergunta in enumerate(PERGUNTAS_TESTE_2, 1):
            numero_questao = item_pergunta["numero_questao"]
            pergunta = item_pergunta["pergunta"]
            print(f"  [{pergunta_idx}/{len(PERGUNTAS_TESTE_2)}] Q{numero_questao}: {pergunta[:60]}...")
            
            start_time = time.time()
            response = qa_chain.invoke({"query": pergunta})
            latency = time.time() - start_time
            
            answer = response["result"]
            source_docs = response["source_documents"]
            
            num_docs = len(source_docs)
            
            proximidades = []
            for doc in source_docs:
                if hasattr(doc, "metadata") and "distance" in doc.metadata:
                    similaridade = 1 - doc.metadata["distance"]
                    proximidades.append(similaridade)
                else:
                    proximidades.append(None)
            
            avg_proximidade = sum([p for p in proximidades if p is not None]) / len([p for p in proximidades if p is not None]) if proximidades and any(p is not None for p in proximidades) else 0.0
            
            proximidades_str = ";".join([f"{p:.4f}" if p is not None else "N/A" for p in proximidades]) if proximidades else ""
            
            result = {
                "numero_questao": numero_questao,
                "pergunta": pergunta,
                "resposta": answer,
                "latency_segundos": round(latency, 4),
                "num_docs_recuperados": num_docs,
                "avg_proximidade": round(avg_proximidade, 4),
                "proximidades": proximidades_str,
            }
            writer.writerow(result)
            csv_file.flush()
            
            print("  → Executando OpenAI File Search...")
            gpt_answer, file_search_results, openai_latency = execute_openai_file_search_query(pergunta)
            
            similarity = 0.0
            if answer and gpt_answer:
                embeddings = embed_texts([answer, gpt_answer])
                similarity = _cosine_similarity(embeddings[0], embeddings[1])
            
            compare_result = {
                "numero_questao": numero_questao,
                "pergunta": pergunta,
                "modelo_mistral": model,
                "modelo_openai": OPENAI_FILE_SEARCH_MODEL,
                "vector_store_id": OPENAI_VECTOR_STORE_ID,
                "embedding_model": EMBEDDING_MODEL,
                "resposta_mistral": answer,
                "resposta_gpt": gpt_answer,
                "similaridade_cosseno": round(similarity, 6),
                "latencia_mistral_segundos": round(latency, 4),
                "latencia_openai_segundos": round(openai_latency, 4),
                "openai_num_resultados": len(file_search_results),
                "openai_resultados": json.dumps(file_search_results, ensure_ascii=False) if file_search_results else "",
            }
            compare_writer.writerow(compare_result)
            compare_csv_file.flush()
        
        print()
    
    csv_file.close()
    compare_csv_file.close()
    
    print("=" * 80)
    print("Etapa 3 concluída!")
    print(f"Arquivos salvos: {csv_filename}, {compare_csv_filename}")
    print("=" * 80)

if __name__ == "__main__":
    main()
