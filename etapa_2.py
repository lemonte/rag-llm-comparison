MODELOS_EXECUTAR = ["mistral", "neural-chat"]
CHUNK_SIZES = [1000, 1200]
CHUNK_OVERLAPS = [100, 200]
K_DOCUMENTS_LIST = [3, 4]

import time
import csv
from datetime import datetime
from itertools import product
from common.config import DOCUMENT_URLS, LOCAL_FILES, PERGUNTAS_TESTE_2, PROMPT_TEMPLATE, get_chromadb_folder
from common.document_processor import fetch_and_process_urls
from common.rag_system import initialize_chromadb, load_chromadb, create_retrieval_chain
from common.evaluation import calculate_llm_judge_score, llm_as_judge_evaluate_response

def main():
    print("=" * 80)
    print("Etapa 2 - Avaliação Completa com LLM-as-a-Judge")
    print("=" * 80)
    
    documents_list = [url for url in DOCUMENT_URLS] + [file for file in LOCAL_FILES]
    
    if not documents_list:
        print("Erro: Nenhum documento encontrado para processar")
        return
    
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
    
    csv_file = open(csv_filename, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    csv_file.flush()
    
    combinations = list(product(MODELOS_EXECUTAR, CHUNK_SIZES, CHUNK_OVERLAPS, K_DOCUMENTS_LIST))
    
    print(f"Total de combinações: {len(combinations)}")
    print()
    
    total_saved = 0
    
    for idx, (model, chunk_size, chunk_overlap, num_docs) in enumerate(combinations, 1):
        print(f"[{idx}/{len(combinations)}] Modelo: {model}, Chunk: {chunk_size}, Overlap: {chunk_overlap}, Docs: {num_docs}")
        print("-" * 80)
        
        if chunk_overlap >= chunk_size:
            print(f"Pulando: overlap ({chunk_overlap}) >= chunk_size ({chunk_size})")
            for item_pergunta in PERGUNTAS_TESTE_2:
                result = {
                    "numero_questao": item_pergunta["numero_questao"],
                    "pergunta": item_pergunta["pergunta"],
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
            continue
        
        persist_dir = get_chromadb_folder(model, chunk_size, num_docs, chunk_overlap)
        
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
        
        search_kwargs = {"k": num_docs}
        qa_chain = create_retrieval_chain(vectorstore, model, search_kwargs, PROMPT_TEMPLATE)
        
        for pergunta_idx, item_pergunta in enumerate(PERGUNTAS_TESTE_2, 1):
            numero_questao = item_pergunta["numero_questao"]
            pergunta = item_pergunta["pergunta"]
            print(f"  [{pergunta_idx}/{len(PERGUNTAS_TESTE_2)}] Q{numero_questao}: {pergunta[:60]}...")
            
            start_time = time.time()
            response = qa_chain.invoke({"query": pergunta})
            query_time = time.time() - start_time
            
            answer = response["result"]
            source_docs = response["source_documents"]
            
            referencias = []
            proximidades = []
            retrieved_docs_for_eval = []
            
            for doc in source_docs:
                source = doc.metadata.get("source", "Desconhecida")
                referencias.append(source)
                retrieved_docs_for_eval.append({
                    "document": doc.page_content,
                    "metadata": doc.metadata
                })
                proximidades.append("N/A")
            
            context = "\n".join([doc.page_content[:200] for doc in source_docs])
            
            avaliacao_docs = calculate_llm_judge_score(
                question=pergunta,
                retrieved_docs=retrieved_docs_for_eval,
                model="gpt-4o-mini"
            )
            
            result_eval = llm_as_judge_evaluate_response(
                question=pergunta,
                context=context,
                answer=answer,
                model="gpt-4o-mini",
                return_feedback=True
            )
            
            if isinstance(result_eval, tuple):
                nota_final, feedback = result_eval
            else:
                nota_final = result_eval
                feedback = ""
            
            result = {
                "numero_questao": numero_questao,
                "pergunta": pergunta,
                "modelo": model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "num_docs": num_docs,
                "tempo_segundos": round(query_time, 4),
                "num_docs_recuperados": len(source_docs),
                "avaliacao_documentos": avaliacao_docs,
                "nota_final": nota_final,
                "feedback": feedback,
                "referencias_utilizadas": "; ".join(referencias),
                "grau_proximidade": "; ".join(proximidades),
                "resposta": answer,
                "contexto": context[:500] + "..." if len(context) > 500 else context
            }
            
            writer.writerow(result)
            csv_file.flush()
            total_saved += 1
            
            if total_saved % 50 == 0:
                print(f"  💾 {total_saved} resultados salvos...")
        
        print()
    
    csv_file.close()
    
    print("=" * 80)
    print("Etapa 2 concluída!")
    print(f"Total de resultados salvos: {total_saved}")
    print(f"Arquivo salvo: {csv_filename}")
    print("=" * 80)

if __name__ == "__main__":
    main()
