import time
import csv
import os
from datetime import datetime
from config import (
    PERGUNTAS_TESTE,
    OLLAMA_MODEL,
    PROMPT_TEMPLATE,
    NUM_RETRIEVED_DOCS,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)
from rag import RAGSystem
import ollama


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
    
    # Verifica se há documentos indexados
    collection_info = rag_system.get_collection_info()
    if collection_info["total_chunks"] == 0:
        print("ERRO: Nenhum documento indexado encontrado!")
        print("Execute o app.py primeiro para indexar os documentos.")
        return
    
    print(f"Documentos indexados: {collection_info['total_chunks']} chunks")
    print()
    
    # Lista para armazenar resultados
    results = []
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


