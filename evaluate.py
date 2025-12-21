import time
import csv
import json
from datetime import datetime
from typing import List
import numpy as np
from openai import OpenAI
from config import (
    PERGUNTAS_TESTE,
    OLLAMA_MODEL_SINGLE as OLLAMA_MODEL,
    PROMPT_TEMPLATE,
    NUM_RETRIEVED_DOCS_SINGLE as NUM_RETRIEVED_DOCS,
    CHUNK_SIZE_SINGLE as CHUNK_SIZE,
    CHUNK_OVERLAP_SINGLE as CHUNK_OVERLAP,
    OPENAI_API_KEY,
    OPENAI_VECTOR_STORE_ID,
    OPENAI_FILE_SEARCH_MODEL,
    OPENAI_EMBEDDING_MODEL,
    REFERENCE_URLS,
    LOCAL_FILES,
)
from scraper import scrape_multiple_urls, read_multiple_local_files
from rag import RAGSystem
import ollama

openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


def get_persist_directory(chunk_size: int, chunk_overlap: int) -> str:
    return f"chroma_db_{chunk_size}_{chunk_overlap}"


def execute_rag_query(rag_system: RAGSystem, question: str) -> tuple:
    """
    Executa uma query RAG e retorna resposta, documentos recuperados e latência.
    """
    start_time = time.time()
    
    # Recupera documentos
    retrieved_docs = rag_system.retrieve_documents(
        question,
        n_results=NUM_RETRIEVED_DOCS
    )
    
    if not retrieved_docs:
        return "", retrieved_docs, time.time() - start_time
    
    # Formata contexto
    context = rag_system.format_context(retrieved_docs)
    
    # Gera prompt
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=question
    )
    
    # Gera resposta
    try:
        response = ollama.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            stream=False,
            options={"temperature": 0}
        )
        answer = response.get('response', '').strip()
    except Exception as e:
        print(f"Erro ao gerar resposta: {e}")
        answer = ""
    
    latency = time.time() - start_time
    
    return answer, retrieved_docs, latency


def ensure_indexed(rag_system: RAGSystem, chunk_size: int, chunk_overlap: int) -> None:
    collection_info = rag_system.get_collection_info()
    if collection_info["total_chunks"] > 0:
        return

    all_documents = {}
    if REFERENCE_URLS:
        url_documents = scrape_multiple_urls(REFERENCE_URLS, delay=1.0)
        all_documents.update(url_documents)
    if LOCAL_FILES:
        local_documents = read_multiple_local_files(LOCAL_FILES)
        all_documents.update(local_documents)
    if not all_documents:
        raise ValueError("Nenhum documento encontrado para indexar")

    rag_system.index_documents(
        all_documents,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        clear_existing=True
    )


def _extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text is not None:
        return str(output_text).strip()
    try:
        chunks = []
        for item in getattr(response, "output", []) or []:
            for content in getattr(item, "content", []) or []:
                text = getattr(content, "text", None)
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()
    except Exception:
        return ""


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
            "max_num_results": NUM_RETRIEVED_DOCS
        }],
        include=["file_search_call.results"]
    )
    latency = time.time() - start_time
    answer = _extract_response_text(response)
    results = []
    try:
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "file_search_call":
                continue
            results = getattr(item, "results", None) or []
    except Exception:
        results = []
    return answer, _normalize_file_search_results(results), latency


def _normalize_file_search_results(results) -> list:
    normalized = []
    for r in results or []:
        if isinstance(r, dict):
            normalized.append(r)
            continue
        model_dump = getattr(r, "model_dump", None)
        if callable(model_dump):
            normalized.append(model_dump())
            continue
        to_dict = getattr(r, "to_dict", None)
        if callable(to_dict):
            normalized.append(to_dict())
            continue
        data = getattr(r, "__dict__", None)
        if isinstance(data, dict) and data:
            normalized.append(data)
            continue
        normalized.append({"value": str(r)})
    return normalized


def _cosine_similarity(a: list, b: list) -> float:
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        return 0.0
    return float(np.dot(va, vb) / denom)


def embed_texts(texts: List[str]) -> List[List[float]]:
    if not openai_client:
        raise RuntimeError("OPENAI_API_KEY não está configurada.")
    response = openai_client.embeddings.create(
        model=OPENAI_EMBEDDING_MODEL,
        input=texts
    )
    return [item.embedding for item in response.data]


def main():
    """
    Executa o fluxo de avaliação simplificado:
    1. Para cada pergunta em PERGUNTAS_TESTE:
       - Executa RAG e mede latência
    2. Gera tabela CSV com resultados
    """
    print("=" * 80)
    print("Sistema de Avaliação RAG - Métricas de Tempo")
    print("=" * 80)
    print(f"Total de perguntas: {len(PERGUNTAS_TESTE)}")
    print(f"Modelo: {OLLAMA_MODEL}")
    print()
    
    # Inicializa sistema RAG
    persist_dir = get_persist_directory(CHUNK_SIZE, CHUNK_OVERLAP)
    rag_system = RAGSystem(persist_directory=persist_dir)
    
    try:
        ensure_indexed(rag_system, CHUNK_SIZE, CHUNK_OVERLAP)
    except Exception as e:
        print(f"ERRO ao indexar documentos: {e}")
        return

    collection_info = rag_system.get_collection_info()
    print(f"Documentos indexados: {collection_info['total_chunks']} chunks")
    print()

    if not openai_client:
        print("ERRO: OPENAI_API_KEY não está configurada.")
        print("Crie um arquivo .env com OPENAI_API_KEY=... e rode novamente.")
        return
    
    # Lista para armazenar resultados
    results = []
    compare_results = []
    total_start_time = time.time()
    
    # Processa cada pergunta
    for i, pergunta in enumerate(PERGUNTAS_TESTE, 1):
        print(f"[{i}/{len(PERGUNTAS_TESTE)}] Processando: {pergunta[:60]}...")
        
        # Executa RAG
        print("  → Executando RAG...")
        answer, docs, latency = execute_rag_query(rag_system, pergunta)
        
        if not answer:
            print("  ⚠️  Nenhuma resposta gerada")
            answer = ""
        
        print(f"  ✓ Resposta gerada (latência: {latency:.2f}s)")

        print("  → Executando OpenAI File Search...")
        try:
            gpt_answer, file_search_results, openai_latency = execute_openai_file_search_query(pergunta)
        except Exception as e:
            print(f"  ⚠️  Erro OpenAI: {e}")
            gpt_answer = ""
            file_search_results = []
            openai_latency = 0.0

        similarity = 0.0
        if answer and gpt_answer:
            try:
                embeddings = embed_texts([answer, gpt_answer])
                similarity = _cosine_similarity(embeddings[0], embeddings[1])
            except Exception as e:
                print(f"  ⚠️  Erro ao gerar embeddings/similaridade: {e}")
                similarity = 0.0
        
        # Calcula informações sobre documentos recuperados
        num_docs = len(docs)
        
        # Calcula proximidades (similaridades)
        proximidades = []
        if docs:
            for doc in docs:
                if doc.get('distance') is not None:
                    similaridade = 1 - doc['distance']
                    proximidades.append(similaridade)
                else:
                    proximidades.append(None)
        
        # Calcula média de proximidades
        avg_proximidade = sum([p for p in proximidades if p is not None]) / len([p for p in proximidades if p is not None]) if proximidades and any(p is not None for p in proximidades) else 0.0
        
        # Formata proximidades como string (lista)
        proximidades_str = ';'.join([f"{p:.4f}" if p is not None else "N/A" for p in proximidades]) if proximidades else ""
        
        # Armazena resultados
        result = {
            'pergunta': pergunta,
            'resposta': answer,
            'latency_segundos': round(latency, 4),
            'num_docs_recuperados': num_docs,
            'avg_proximidade': round(avg_proximidade, 4),
            'proximidades': proximidades_str,
        }
        results.append(result)

        compare_results.append({
            "pergunta": pergunta,
            "modelo_mistral": OLLAMA_MODEL,
            "modelo_openai": OPENAI_FILE_SEARCH_MODEL,
            "vector_store_id": OPENAI_VECTOR_STORE_ID,
            "embedding_model": OPENAI_EMBEDDING_MODEL,
            "resposta_mistral": answer,
            "resposta_gpt": gpt_answer,
            "similaridade_cosseno": round(similarity, 6),
            "latencia_mistral_segundos": round(latency, 4),
            "latencia_openai_segundos": round(openai_latency, 4),
            "openai_num_resultados": len(file_search_results),
            "openai_resultados": json.dumps(file_search_results, ensure_ascii=False) if file_search_results else "",
        })
        
        print()
    
    # Calcula throughput
    total_time = time.time() - total_start_time
    total_queries = len(PERGUNTAS_TESTE)
    throughput = total_queries / total_time if total_time > 0 else 0
    
    print("=" * 80)
    print("Avaliação concluída!")
    print(f"Tempo total: {total_time:.2f}s")
    print(f"Throughput: {throughput:.4f} queries/segundo")
    print()
    
    # Gera arquivo CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"avaliacao_rag_{timestamp}.csv"
    compare_csv_filename = f"comparacao_openai_mistral_{timestamp}.csv"
    
    # Define colunas do CSV
    fieldnames = [
        'pergunta',
        'resposta',
        'latency_segundos',
        'num_docs_recuperados',
        'avg_proximidade',
        'proximidades',
    ]
    
    with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
        
        # Adiciona linha com throughput
        throughput_row = {
            'pergunta': 'THROUGHPUT',
            'resposta': '',
            'latency_segundos': round(throughput, 4),
            'num_docs_recuperados': '',
            'avg_proximidade': '',
            'proximidades': '',
        }
        writer.writerow(throughput_row)
    
    print(f"✓ Resultados salvos em: {csv_filename}")
    print()

    compare_fieldnames = [
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
    with open(compare_csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=compare_fieldnames)
        writer.writeheader()
        writer.writerows(compare_results)
    print(f"✓ Comparação salva em: {compare_csv_filename}")
    print()
    
    # Estatísticas resumidas
    latencies = [r['latency_segundos'] for r in results]
    
    print("=" * 80)
    print("ESTATÍSTICAS RESUMIDAS")
    print("=" * 80)
    print(f"Latência média: {sum(latencies) / len(latencies):.2f}s")
    print(f"Latência mínima: {min(latencies):.2f}s")
    print(f"Latência máxima: {max(latencies):.2f}s")
    print(f"Throughput: {throughput:.4f} queries/segundo")
    print("=" * 80)


if __name__ == "__main__":
    main()
