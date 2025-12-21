import time
import csv
from datetime import datetime
from itertools import product
from typing import Dict, List
import numpy as np
from langchain_community.embeddings import OllamaEmbeddings
from config import (
    OLLAMA_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    NUM_RETRIEVED_DOCS,
    PERGUNTAS_TESTE,
    PROMPT_TEMPLATE,
    REFERENCE_URLS,
    LOCAL_FILES,
    OPENAI_API_KEY,
    OPENAI_VECTOR_STORE_ID,
    EMBEDDING_MODEL
)
from scraper import scrape_multiple_urls, read_multiple_local_files
from rag import RAGSystem
from evaluation import (
    calculate_llm_judge_score,
    llm_as_judge_evaluate_response
)
import ollama
from openai import OpenAI


def get_persist_directory(chunk_size: int, chunk_overlap: int) -> str:
    """Retorna diretório de persistência."""
    return f"chroma_db_{chunk_size}_{chunk_overlap}"


def ensure_indexed(rag_system: RAGSystem, chunk_size: int, chunk_overlap: int):
    """Garante que documentos estão indexados."""
    collection_info = rag_system.get_collection_info()
    
    if collection_info["total_chunks"] == 0:
        print(f"  → Indexando documentos (chunk_size={chunk_size}, overlap={chunk_overlap})...")
        
        all_documents = {}
        
        if REFERENCE_URLS:
            url_documents = scrape_multiple_urls(REFERENCE_URLS, delay=1.0)
            all_documents.update(url_documents)
        
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
    """Executa query RAG e retorna métricas."""
    start_time = time.time()
    
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
    
    context = rag_system.format_context(retrieved_docs)
    
    if model == "openai-gpt-4.1":
        try:
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY não configurada")
            
            client = OpenAI(api_key=OPENAI_API_KEY)
            
            response = client.responses.create(
                model="gpt-4.1",
                input=question,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [OPENAI_VECTOR_STORE_ID],
                    "max_num_results": num_docs
                }],
                include=["file_search_call.results"]
            )
            
            if hasattr(response, 'output') and response.output:
                answer = response.output.strip()
            elif hasattr(response, 'text') and response.text:
                answer = response.text.strip()
            elif hasattr(response, 'choices') and len(response.choices) > 0:
                answer = response.choices[0].message.content.strip() if hasattr(response.choices[0].message, 'content') else str(response)
            else:
                answer = str(response)
        except Exception as e:
            print(f"    ⚠️  Erro ao gerar resposta com OpenAI: {e}")
            answer = ""
    else:
        prompt = prompt_template.format(
            context=context,
            question=question
        )
        
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
    
    avaliacao_docs = None
    try:
        avaliacao_docs = calculate_llm_judge_score(
            question=question,
            retrieved_docs=retrieved_docs,
            model="gpt-4o-mini"
        )
    except Exception as e:
        print(f"    ⚠️  Erro ao avaliar documentos: {e}")
    
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


def calculate_text_similarity(text1: str, text2: str, ollama_embeddings: OllamaEmbeddings) -> float:
    """Calcula similaridade de cosseno entre dois textos usando OllamaEmbeddings."""
    if not text1 or not text2:
        return 0.0
    
    embedding1 = ollama_embeddings.embed_query(text1)
    embedding2 = ollama_embeddings.embed_query(text2)
    
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    return float(similarity)


def analyze_similarity(csv_filename: str):
    """Analisa similaridade entre respostas do Mistral e GPT-4.1."""
    print(f"  → Carregando resultados de {csv_filename}...")
    
    results_by_question: Dict[str, Dict] = {}
    
    with open(csv_filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pergunta = row.get('pergunta', '')
            modelo = row.get('modelo', '')
            resposta = row.get('resposta', '')
            tempo = row.get('tempo_segundos', '0')
            numero_questao = row.get('numero_questao', '')
            
            if not pergunta:
                continue
            
            if pergunta not in results_by_question:
                results_by_question[pergunta] = {
                    'numero_questao': numero_questao,
                    'pergunta': pergunta,
                    'mistral': None,
                    'gpt': None
                }
            
            if modelo == 'mistral':
                results_by_question[pergunta]['mistral'] = {
                    'resposta': resposta,
                    'tempo_segundos': tempo
                }
            elif modelo == 'openai-gpt-4.1':
                results_by_question[pergunta]['gpt'] = {
                    'resposta': resposta,
                    'tempo_segundos': tempo
                }
    
    print(f"  → Encontradas {len(results_by_question)} perguntas únicas")
    print(f"  → Inicializando Ollama Embeddings ({EMBEDDING_MODEL})...")
    
    ollama_embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    similarity_csv_filename = f"similaridade_respostas_{timestamp}.csv"
    
    fieldnames = [
        "numero_questao",
        "pergunta",
        "resposta_mistral",
        "tempo_mistral_segundos",
        "resposta_gpt",
        "tempo_gpt_segundos",
        "similaridade"
    ]
    
    similarity_file = open(similarity_csv_filename, 'w', newline='', encoding='utf-8')
    similarity_writer = csv.DictWriter(similarity_file, fieldnames=fieldnames)
    similarity_writer.writeheader()
    similarity_file.flush()
    
    total_compared = 0
    for pergunta, data in results_by_question.items():
        mistral_data = data.get('mistral')
        gpt_data = data.get('gpt')
        
        if not mistral_data or not gpt_data:
            continue
        
        resposta_mistral = mistral_data.get('resposta', '')
        resposta_gpt = gpt_data.get('resposta', '')
        
        if not resposta_mistral or not resposta_gpt:
            continue
        
        print(f"  → Calculando similaridade para questão {data['numero_questao']}...")
        similarity = calculate_text_similarity(resposta_mistral, resposta_gpt, ollama_embeddings)
        
        result = {
            "numero_questao": data['numero_questao'],
            "pergunta": pergunta,
            "resposta_mistral": resposta_mistral,
            "tempo_mistral_segundos": mistral_data.get('tempo_segundos', '0'),
            "resposta_gpt": resposta_gpt,
            "tempo_gpt_segundos": gpt_data.get('tempo_segundos', '0'),
            "similaridade": round(similarity, 4)
        }
        
        similarity_writer.writerow(result)
        similarity_file.flush()
        total_compared += 1
        
        if total_compared % 10 == 0:
            print(f"  💾 {total_compared} comparações salvas...")
    
    similarity_file.close()
    
    print()
    print("=" * 80)
    print("Análise de Similaridade Concluída!")
    print(f"Total de comparações: {total_compared}")
    print(f"Arquivo salvo: {similarity_csv_filename}")
    print("=" * 80)


def main():
    """Executa avaliação completa com todas combinações."""
    print("=" * 80)
    print("Avaliação Completa - Todas as Combinações de Parâmetros")
    print("=" * 80)
    
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
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"avaliacao_completa_{timestamp}.csv"
    
    fieldnames = [
        "numero_questao",
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
    
    csv_file = open(csv_filename, 'w', newline='', encoding='utf-8')
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()
    
    total_saved = 0
    combination_count = 0
    
    total_combinations_list = list(product(models, chunk_sizes, chunk_overlaps, num_docs_list))
    
    for model, chunk_size, chunk_overlap, num_docs in total_combinations_list:
        combination_count += 1
        
        if chunk_overlap >= chunk_size:
            print(f"\n[{combination_count}/{len(total_combinations_list)}] "
                  f"Modelo: {model}, Chunk: {chunk_size}, Overlap: {chunk_overlap}, Docs: {num_docs}")
            print("-" * 80)
            print(f"  ⚠️  Pulando: overlap ({chunk_overlap}) >= chunk_size ({chunk_size}) - combinação inválida")
            for pergunta_item in perguntas:
                if isinstance(pergunta_item, dict):
                    numero_questao = pergunta_item.get("numero_questao", "")
                    pergunta_texto = pergunta_item.get("pergunta", "")
                else:
                    numero_questao = ""
                    pergunta_texto = pergunta_item
                
                result = {
                    "numero_questao": numero_questao,
                    "pergunta": pergunta_texto,
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
                writer.writerow(result)
                csv_file.flush()
                total_saved += 1
                if total_saved % 50 == 0:
                    print(f"  💾 {total_saved} resultados salvos...")
            continue
        
        print(f"\n[{combination_count}/{len(total_combinations_list)}] "
              f"Modelo: {model}, Chunk: {chunk_size}, Overlap: {chunk_overlap}, Docs: {num_docs}")
        print("-" * 80)
        
        persist_dir = get_persist_directory(chunk_size, chunk_overlap)
        rag_system = RAGSystem(persist_directory=persist_dir)
        
        try:
            ensure_indexed(rag_system, chunk_size, chunk_overlap)
        except Exception as e:
            print(f"  ⚠️  Erro ao indexar: {e}")
            for pergunta_item in perguntas:
                if isinstance(pergunta_item, dict):
                    numero_questao = pergunta_item.get("numero_questao", "")
                    pergunta_texto = pergunta_item.get("pergunta", "")
                else:
                    numero_questao = ""
                    pergunta_texto = pergunta_item
                
                result = {
                    "numero_questao": numero_questao,
                    "pergunta": pergunta_texto,
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
                writer.writerow(result)
                csv_file.flush()
                total_saved += 1
                if total_saved % 50 == 0:
                    print(f"  💾 {total_saved} resultados salvos...")
            continue
        
        for pergunta_idx, pergunta_item in enumerate(perguntas, 1):
            if isinstance(pergunta_item, dict):
                numero_questao = pergunta_item.get("numero_questao", "")
                pergunta_texto = pergunta_item.get("pergunta", "")
            else:
                numero_questao = ""
                pergunta_texto = pergunta_item
            
            print(f"  [{pergunta_idx}/{len(perguntas)}] Questão {numero_questao}: {pergunta_texto[:60]}...")
            
            try:
                result = execute_query(
                    rag_system=rag_system,
                    question=pergunta_texto,
                    model=model,
                    num_docs=num_docs,
                    prompt_template=PROMPT_TEMPLATE
                )
                
                result.update({
                    "numero_questao": numero_questao,
                    "pergunta": pergunta_texto,
                    "modelo": model,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "num_docs": num_docs
                })
                
                writer.writerow(result)
                csv_file.flush()
                total_saved += 1
                
                print(f"    ✓ Concluído (Tempo: {result['tempo_segundos']:.2f}s, "
                      f"Nota Final: {result['nota_final'] if result['nota_final'] else 'N/A'}) - Salvo no CSV")
                
                if total_saved % 50 == 0:
                    print(f"  💾 {total_saved} resultados salvos no arquivo...")
                
            except Exception as e:
                print(f"    ⚠️  Erro: {e}")
                result = {
                    "numero_questao": numero_questao,
                    "pergunta": pergunta_texto,
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
                
                writer.writerow(result)
                csv_file.flush()
                total_saved += 1
                
                if total_saved % 50 == 0:
                    print(f"  💾 {total_saved} resultados salvos no arquivo...")
    
    csv_file.close()
    
    print()
    print("=" * 80)
    print("Avaliação Concluída!")
    print(f"Total de resultados salvos: {total_saved}")
    print(f"Arquivo salvo: {csv_filename}")
    print("=" * 80)
    
    print()
    print("=" * 80)
    print("Calculando similaridade entre respostas do Mistral e GPT-4.1...")
    print("=" * 80)
    
    analyze_similarity(csv_filename)


if __name__ == "__main__":
    main()



