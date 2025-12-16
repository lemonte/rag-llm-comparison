"""
Algoritmo de Avaliação Completa
Executa todas as combinações de parâmetros e salva resultados em planilha
"""
import time
import csv
import os
from datetime import datetime
from itertools import product
from config import (
    OLLAMA_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    NUM_RETRIEVED_DOCS,
    PERGUNTAS_TESTE,
    PROMPT_TEMPLATE,
    REFERENCE_URLS,
    LOCAL_FILES
)
from scraper import scrape_multiple_urls, read_multiple_local_files
from rag import RAGSystem
from evaluation import (
    calculate_llm_judge_score,
    llm_as_judge_evaluate_response
)
import ollama


def get_persist_directory(chunk_size: int, chunk_overlap: int) -> str:
    """Retorna o diretório de persistência baseado nos parâmetros"""
    return f"chroma_db_{chunk_size}_{chunk_overlap}"


def ensure_indexed(rag_system: RAGSystem, chunk_size: int, chunk_overlap: int):
    """Garante que os documentos estão indexados"""
    collection_info = rag_system.get_collection_info()
    
    if collection_info["total_chunks"] == 0:
        print(f"  → Indexando documentos (chunk_size={chunk_size}, overlap={chunk_overlap})...")
        
        all_documents = {}
        
        # Processa URLs
        if REFERENCE_URLS:
            url_documents = scrape_multiple_urls(REFERENCE_URLS, delay=1.0)
            all_documents.update(url_documents)
        
        # Processa arquivos locais
        if LOCAL_FILES:
            local_documents = read_multiple_local_files(LOCAL_FILES)
            all_documents.update(local_documents)
        
        if all_documents:
            rag_system.index_documents(
                all_documents,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                clear_existing=True
            )
            print(f"  ✓ {len(all_documents)} documento(s) indexado(s)")
        else:
            raise ValueError("Nenhum documento encontrado para indexar")
    else:
        print(f"  ✓ Documentos já indexados ({collection_info['total_chunks']} chunks)")


def execute_query(
    rag_system: RAGSystem,
    question: str,
    model: str,
    num_docs: int,
    prompt_template: str
) -> dict:
    """Executa uma query RAG e retorna todas as métricas"""
    start_time = time.time()
    
    # Recupera documentos
    retrieved_docs = rag_system.retrieve_documents(
        question,
        n_results=num_docs
    )
    
    if not retrieved_docs:
        return {
            "resposta": "",
            "tempo_segundos": time.time() - start_time,
            "num_docs_recuperados": 0,
            "avaliacao_documentos": None,
            "nota_final": None,
            "feedback": None,
            "referencias_utilizadas": "",
            "grau_proximidade": "",
            "contexto": ""
        }
    
    # Formata contexto
    context = rag_system.format_context(retrieved_docs)
    
    # Gera prompt
    prompt = prompt_template.format(
        context=context,
        question=question
    )
    
    # Gera resposta
    try:
        response = ollama.generate(
            model=model,
            prompt=prompt,
            stream=False,
            options={"temperature": 0}
        )
        answer = response.get('response', '').strip()
    except Exception as e:
        print(f"    ⚠️  Erro ao gerar resposta: {e}")
        answer = ""
    
    query_time = time.time() - start_time
    
    # Coleta informações dos documentos
    referencias = []
    proximidades = []
    for doc in retrieved_docs:
        url = doc['metadata'].get('url', 'Desconhecida')
        referencias.append(url)
        if doc.get('distance') is not None:
            similaridade = 1 - doc['distance']
            proximidades.append(f"{similaridade:.4f}")
        else:
            proximidades.append("N/A")
    
    # Avalia documentos com LLM-as-a-Judge
    avaliacao_docs = None
    try:
        avaliacao_docs = calculate_llm_judge_score(
            question=question,
            retrieved_docs=retrieved_docs,
            model="gpt-4o-mini"
        )
    except Exception as e:
        print(f"    ⚠️  Erro ao avaliar documentos: {e}")
    
    # Avalia resposta final com LLM-as-a-Judge
    nota_final = None
    feedback = None
    try:
        result = llm_as_judge_evaluate_response(
            question=question,
            context=context,
            answer=answer,
            model="gpt-4o-mini",
            return_feedback=True
        )
        if isinstance(result, tuple):
            nota_final, feedback = result
        else:
            nota_final = result
    except Exception as e:
        print(f"    ⚠️  Erro ao avaliar resposta: {e}")
    
    return {
        "resposta": answer,
        "tempo_segundos": round(query_time, 4),
        "num_docs_recuperados": len(retrieved_docs),
        "avaliacao_documentos": round(avaliacao_docs, 2) if avaliacao_docs is not None else None,
        "nota_final": round(nota_final, 2) if nota_final is not None else None,
        "feedback": feedback if feedback else "",
        "referencias_utilizadas": "; ".join(referencias),
        "grau_proximidade": "; ".join(proximidades),
        "contexto": context[:500] + "..." if len(context) > 500 else context
    }


def main():
    """Executa avaliação completa com todas as combinações"""
    print("=" * 80)
    print("Avaliação Completa - Todas as Combinações de Parâmetros")
    print("=" * 80)
    
    # Garante que são listas
    models = OLLAMA_MODEL if isinstance(OLLAMA_MODEL, list) else [OLLAMA_MODEL]
    chunk_sizes = CHUNK_SIZE if isinstance(CHUNK_SIZE, list) else [CHUNK_SIZE]
    chunk_overlaps = CHUNK_OVERLAP if isinstance(CHUNK_OVERLAP, list) else [CHUNK_OVERLAP]
    num_docs_list = NUM_RETRIEVED_DOCS if isinstance(NUM_RETRIEVED_DOCS, list) else [NUM_RETRIEVED_DOCS]
    perguntas = PERGUNTAS_TESTE if isinstance(PERGUNTAS_TESTE, list) else [PERGUNTAS_TESTE]
    
    print(f"Modelos: {models}")
    print(f"Chunk Sizes: {chunk_sizes}")
    print(f"Chunk Overlaps: {chunk_overlaps}")
    print(f"Números de Documentos: {num_docs_list}")
    print(f"Total de Perguntas: {len(perguntas)}")
    print()
    
    # Filtra overlaps inválidos (overlap >= chunk_size será pulado)
    valid_combinations = 0
    for chunk_size, chunk_overlap in product(chunk_sizes, chunk_overlaps):
        if chunk_overlap < chunk_size:
            valid_combinations += 1
    
    total_combinations = len(models) * valid_combinations * len(num_docs_list) * len(perguntas)
    total_possible = len(models) * len(chunk_sizes) * len(chunk_overlaps) * len(num_docs_list) * len(perguntas)
    
    print(f"Total de combinações possíveis: {total_possible}")
    print(f"Total de combinações válidas (overlap < chunk_size): {total_combinations}")
    print(f"⚠️  Combinações com overlap >= chunk_size serão puladas automaticamente")
    print("=" * 80)
    print()
    
    # Prepara arquivo CSV para salvamento incremental
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"avaliacao_completa_{timestamp}.csv"
    
    fieldnames = [
        "pergunta",
        "modelo",
        "chunk_size",
        "chunk_overlap",
        "num_docs",
        "tempo_segundos",
        "num_docs_recuperados",
        "avaliacao_documentos",
        "nota_final",
        "feedback",
        "referencias_utilizadas",
        "grau_proximidade",
        "resposta",
        "contexto"
    ]
    
    # Cria arquivo CSV e escreve header
    csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()  # Garante que o header seja escrito imediatamente
    
    # Contador para feedback
    total_saved = 0
    combination_count = 0
    
    # Itera sobre todas as combinações
    total_combinations_list = list(product(models, chunk_sizes, chunk_overlaps, num_docs_list))
    
    for model, chunk_size, chunk_overlap, num_docs in total_combinations_list:
        combination_count += 1
        
        # Validação: overlap deve ser menor que chunk_size
        if chunk_overlap >= chunk_size:
            print(f"\n[{combination_count}/{len(total_combinations_list)}] "
                  f"Modelo: {model}, Chunk: {chunk_size}, Overlap: {chunk_overlap}, Docs: {num_docs}")
            print("-" * 80)
            print(f"  ⚠️  Pulando: overlap ({chunk_overlap}) >= chunk_size ({chunk_size}) - combinação inválida")
            # Adiciona resultado vazio para manter consistência
            for pergunta in perguntas:
                result = {
                    "pergunta": pergunta,
                    "modelo": model,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "num_docs": num_docs,
                    "resposta": "",
                    "tempo_segundos": 0,
                    "num_docs_recuperados": 0,
                    "avaliacao_documentos": None,
                    "nota_final": None,
                    "feedback": f"Erro: overlap ({chunk_overlap}) >= chunk_size ({chunk_size})",
                    "referencias_utilizadas": "",
                    "grau_proximidade": "",
                    "contexto": ""
                }
                # Salva imediatamente
                writer.writerow(result)
                csv_file.flush()
                total_saved += 1
                if total_saved % 50 == 0:
                    print(f"  💾 {total_saved} resultados salvos...")
            continue
        
        print(f"\n[{combination_count}/{len(total_combinations_list)}] "
              f"Modelo: {model}, Chunk: {chunk_size}, Overlap: {chunk_overlap}, Docs: {num_docs}")
        print("-" * 80)
        
        # Cria diretório de persistência para esta combinação
        persist_dir = get_persist_directory(chunk_size, chunk_overlap)
        rag_system = RAGSystem(persist_directory=persist_dir)
        
        # Garante que está indexado
        try:
            ensure_indexed(rag_system, chunk_size, chunk_overlap)
        except Exception as e:
            print(f"  ⚠️  Erro ao indexar: {e}")
            # Adiciona resultados com erro para todas as perguntas desta combinação
            for pergunta in perguntas:
                result = {
                    "pergunta": pergunta,
                    "modelo": model,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "num_docs": num_docs,
                    "resposta": "",
                    "tempo_segundos": 0,
                    "num_docs_recuperados": 0,
                    "avaliacao_documentos": None,
                    "nota_final": None,
                    "feedback": f"Erro ao indexar: {str(e)}",
                    "referencias_utilizadas": "",
                    "grau_proximidade": "",
                    "contexto": ""
                }
                # Salva imediatamente
                writer.writerow(result)
                csv_file.flush()
                total_saved += 1
                if total_saved % 50 == 0:
                    print(f"  💾 {total_saved} resultados salvos...")
            continue
        
        # Executa todas as perguntas para esta combinação
        for pergunta_idx, pergunta in enumerate(perguntas, 1):
            print(f"  [{pergunta_idx}/{len(perguntas)}] Pergunta: {pergunta[:60]}...")
            
            try:
                result = execute_query(
                    rag_system=rag_system,
                    question=pergunta,
                    model=model,
                    num_docs=num_docs,
                    prompt_template=PROMPT_TEMPLATE
                )
                
                # Adiciona informações da combinação
                result.update({
                    "pergunta": pergunta,
                    "modelo": model,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "num_docs": num_docs
                })
                
                # Salva imediatamente após cada pergunta
                writer.writerow(result)
                csv_file.flush()
                total_saved += 1
                
                print(f"    ✓ Concluído (Tempo: {result['tempo_segundos']:.2f}s, "
                      f"Nota Final: {result['nota_final'] if result['nota_final'] else 'N/A'}) - Salvo no CSV")
                
                # Feedback a cada 50 resultados
                if total_saved % 50 == 0:
                    print(f"  💾 {total_saved} resultados salvos no arquivo...")
                
            except Exception as e:
                print(f"    ⚠️  Erro: {e}")
                # Adiciona resultado com erro
                result = {
                    "pergunta": pergunta,
                    "modelo": model,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "num_docs": num_docs,
                    "resposta": "",
                    "tempo_segundos": 0,
                    "num_docs_recuperados": 0,
                    "avaliacao_documentos": None,
                    "nota_final": None,
                    "feedback": f"Erro: {str(e)}",
                    "referencias_utilizadas": "",
                    "grau_proximidade": "",
                    "contexto": ""
                }
                
                # Salva imediatamente mesmo em caso de erro
                writer.writerow(result)
                csv_file.flush()
                total_saved += 1
                
                if total_saved % 50 == 0:
                    print(f"  💾 {total_saved} resultados salvos no arquivo...")
    
    # Fecha arquivo (todos os resultados já foram salvos individualmente)
    csv_file.close()
    
    print()
    print("=" * 80)
    print("Avaliação Concluída!")
    print(f"Total de resultados salvos: {total_saved}")
    print(f"Arquivo salvo: {csv_filename}")
    print("=" * 80)


if __name__ == "__main__":
    main()


