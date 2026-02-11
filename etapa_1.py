MODELOS_EXECUTAR = ["llama3.2","llama3.1:405b", "llama3:70b","mistral", "deepseek-r1", "phi3", "neural-chat", "solar", "moondream"]
CHUNK_SIZES = [1000, 1200]
CHUNK_OVERLAPS = [100,200]
K_DOCUMENTS_LIST = [2,3]

import time
from itertools import product
from common.config import DOCUMENT_URLS, LOCAL_FILES, PERGUNTAS_TESTE_1, PROMPT_TEMPLATE_1, get_chromadb_folder, get_metrics_file
from common.document_processor import fetch_and_process_urls
from common.rag_system import initialize_chromadb, load_chromadb, create_retrieval_chain
from common.metrics import salvar_metricas_inicializacao, salvar_metrica_pergunta, salvar_metrica_processamento

def main():
    print("=" * 80)
    print("Etapa 1 - Execução Sequencial com Geração de Planilhas")
    print("=" * 80)
    
    documents_list = [url for url in DOCUMENT_URLS] + [file for file in LOCAL_FILES]
    
    if not documents_list:
        print("Erro: Nenhum documento encontrado para processar")
        return
    
    combinations = list(product(MODELOS_EXECUTAR, CHUNK_SIZES, CHUNK_OVERLAPS, K_DOCUMENTS_LIST))
    
    print(f"Total de combinações: {len(combinations)}")
    print()
    
    for idx, (model, chunk_size, chunk_overlap, k_docs) in enumerate(combinations, 1):
        print(f"[{idx}/{len(combinations)}] Modelo: {model}, Chunk: {chunk_size}, Overlap: {chunk_overlap}, K: {k_docs}")
        print("-" * 80)
        
        if chunk_overlap >= chunk_size:
            print(f"Pulando: overlap ({chunk_overlap}) >= chunk_size ({chunk_size})")
            continue
        
        persist_dir = get_chromadb_folder(model, chunk_size, k_docs, chunk_overlap)
        metrics_file = get_metrics_file(model, chunk_size, k_docs, chunk_overlap)
        
        inicio_total = time.time()
        
        vectorstore = load_chromadb(model, persist_dir)
        
        if not vectorstore:
            print("Inicializando ChromaDB...")
            inicio_proc = time.time()
            documents, num_docs = fetch_and_process_urls(documents_list)
            
            if not documents:
                print("Erro: Nenhum documento processado")
                continue
            
            vectorstore = initialize_chromadb(documents, model, chunk_size, chunk_overlap, persist_dir)
            tempo_proc = time.time() - inicio_proc
            
            salvar_metrica_processamento(
                tempo_proc,
                "inicializacao_documentos",
                num_docs,
                model,
                chunk_size,
                chunk_overlap,
                metrics_file
            )
            print(f"Documentos indexados: {num_docs}")
        else:
            print("ChromaDB carregado do disco")
        
        tempo_inicializacao = time.time() - inicio_total
        salvar_metricas_inicializacao(tempo_inicializacao, metrics_file)
        
        print(f"Executando {len(PERGUNTAS_TESTE_1)} perguntas...")
        
        search_kwargs = {"k": k_docs}
        qa_chain = create_retrieval_chain(vectorstore, model, search_kwargs, PROMPT_TEMPLATE_1)
        
        for pergunta in PERGUNTAS_TESTE_1:
            inicio = time.time()
            response = qa_chain.invoke({"query": pergunta})
            tempo = time.time() - inicio
            
            fontes = [doc.metadata["source_id"] for doc in response["source_documents"]]
            
            salvar_metrica_pergunta(
                pergunta,
                tempo,
                response["result"],
                fontes,
                model,
                k_docs,
                chunk_size,
                chunk_overlap,
                metrics_file
            )
        
        print(f"Concluído! Planilha salva em: {metrics_file}")
        print()
    
    print("=" * 80)
    print("Etapa 1 concluída!")
    print("=" * 80)

if __name__ == "__main__":
    main()
