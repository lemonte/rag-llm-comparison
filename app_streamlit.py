import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor
from common.config import PROMPT_TEMPLATE, get_chromadb_folder, get_documents_file
from common.document_processor import process_file, process_url, normalizar_documento, fetch_and_process_urls
from common.rag_system import initialize_chromadb, load_chromadb, create_retrieval_chain, get_chromadb_documents

MODEL_OPTIONS = ["neural-chat"]
CHUNK_SIZE = 1000
K_DOCUMENTS = 3
CHUNK_OVERLAP = 100

DOCUMENTS_FILE = get_documents_file()

def salvar_documentos(documentos):
    docs_normalizados = [
        doc for doc in (normalizar_documento(d) for d in documentos)
        if doc is not None
    ]
    with open(DOCUMENTS_FILE, "w") as f:
        json.dump(docs_normalizados, f)

def carregar_documentos():
    if os.path.exists(DOCUMENTS_FILE):
        with open(DOCUMENTS_FILE, "r") as f:
            documentos = json.load(f)
            return documentos
    return []

def processar_documento_async(doc_info, selected_model, chunk_size, chunk_overlap, k_documents):
    persist_dir = get_chromadb_folder(selected_model, chunk_size, k_documents, chunk_overlap)
    vectorstore = load_chromadb(selected_model, persist_dir)
    if not vectorstore:
        return
    
    if isinstance(doc_info, str):
        text = process_url(doc_info)
        metadata = {
            "source": doc_info,
            "source_id": doc_info.split("/")[-1].replace(".html", ""),
            "type": "url"
        }
    else:
        text = doc_info["text"]
        metadata = {
            "source": doc_info["name"],
            "source_id": doc_info["name"],
            "type": doc_info["type"]
        }
    
    if text:
        doc = Document(
            page_content=text,
            metadata=metadata
        )
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        texts = text_splitter.split_documents([doc])
        
        vectorstore.add_documents(texts)
        
        doc_name = doc_info if isinstance(doc_info, str) else doc_info["name"]
        documentos_processando = st.session_state.get("documentos_processando", set())
        if doc_name in documentos_processando:
            documentos_processando.remove(doc_name)
            st.session_state.documentos_processando = documentos_processando

def iniciar_processamento_async(doc_info, selected_model, chunk_size, chunk_overlap, k_documents):
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ProcessadorDoc")
    
    future = executor.submit(
        processar_documento_async,
        doc_info,
        selected_model,
        chunk_size,
        chunk_overlap,
        k_documents
    )
    
    def done_callback(future):
        try:
            future.result()
        except Exception as e:
            print(f"Erro no callback do processamento: {str(e)}")
        finally:
            executor.shutdown(wait=False)
    
    future.add_done_callback(done_callback)

def streamlit_app():
    st.title("🤖 Assistente de Documentação com RAG")
    
    selected_model = st.sidebar.selectbox("Modelo:", MODEL_OPTIONS, index=0)
    chunk_size = st.sidebar.slider("Tamanho do Chunk:", min_value=100, max_value=2000, value=CHUNK_SIZE, step=100)
    chunk_overlap = st.sidebar.slider("Sobreposição do Chunk:", min_value=0, max_value=500, value=CHUNK_OVERLAP, step=10)
    k_documents = st.sidebar.slider("Número de documentos para recuperar:", min_value=1, max_value=10, value=K_DOCUMENTS)
    
    st.sidebar.header("📚 Gerenciar Documentos")
    
    if "documentos_processando" not in st.session_state:
        st.session_state.documentos_processando = set()
    
    if "vectorstore" in st.session_state:
        current_docs = get_chromadb_documents(st.session_state.vectorstore)
        if current_docs:
            st.session_state.document_urls = current_docs
            salvar_documentos(current_docs)
    elif "document_urls" not in st.session_state:
        st.session_state.document_urls = carregar_documentos()
    
    st.sidebar.markdown("### Documentos Carregados:")
    for i, doc in enumerate(st.session_state.document_urls):
        col1, col2 = st.sidebar.columns([4, 1])
        with col1:
            doc_name = doc if isinstance(doc, str) else doc["name"]
            status = "🔄" if doc_name in st.session_state.documentos_processando else "✅"
            if isinstance(doc, str):
                st.text(f"{i+1}. 🌐 {doc} {status}")
            else:
                st.text(f"{i+1}. 📄 {doc['name']} {status}")
        with col2:
            if st.button("🗑️", key=f"delete_{i}"):
                st.session_state.document_urls.pop(i)
                salvar_documentos(st.session_state.document_urls)
                st.rerun()
    
    with st.sidebar.expander("➕ Adicionar Nova URL"):
        new_url = st.text_input("URL do documento:")
        if st.button("Adicionar URL"):
            if new_url and new_url not in st.session_state.document_urls:
                st.session_state.document_urls.append(new_url)
                st.session_state.documentos_processando.add(new_url)
                iniciar_processamento_async(new_url, selected_model, chunk_size, chunk_overlap, k_documents)
                salvar_documentos(st.session_state.document_urls)
                st.success("URL adicionada com sucesso! Processando...")
                st.rerun()
            elif new_url in st.session_state.document_urls:
                st.warning("Esta URL já foi adicionada!")
    
    with st.sidebar.expander("📁 Upload de Arquivo"):
        uploaded_file = st.file_uploader("Escolha um arquivo", type=["pdf", "md", "txt"])
        if uploaded_file is not None:
            file_name = uploaded_file.name
            existing_files = [doc.get("name") if isinstance(doc, dict) else None for doc in st.session_state.document_urls]
            
            if file_name not in existing_files:
                text = process_file(uploaded_file)
                if text:
                    doc_info = {
                        "name": file_name,
                        "text": text,
                        "type": os.path.splitext(file_name)[1][1:]
                    }
                    st.session_state.document_urls.append(doc_info)
                    st.session_state.documentos_processando.add(file_name)
                    iniciar_processamento_async(doc_info, selected_model, chunk_size, chunk_overlap, k_documents)
                    salvar_documentos(st.session_state.document_urls)
                    st.success(f"Arquivo {file_name} enviado com sucesso! Processando...")
                    st.rerun()
            else:
                st.warning("Este arquivo já foi carregado!")
    
    if st.sidebar.button("🔄 Reprocessar Base de Conhecimento"):
        with st.spinner("Reprocessando todos os documentos..."):
            inicio_proc = time.time()
            documents, num_docs = fetch_and_process_urls(st.session_state.document_urls)
            if documents:
                persist_dir = get_chromadb_folder(selected_model, chunk_size, k_documents, chunk_overlap)
                vectorstore = initialize_chromadb(documents, selected_model, chunk_size, chunk_overlap, persist_dir)
                if vectorstore:
                    tempo_proc = time.time() - inicio_proc
                    st.session_state.vectorstore = vectorstore
                    st.success(f"Base de conhecimento atualizada com sucesso em {tempo_proc:.2f} segundos! {num_docs} documentos processados.")
                else:
                    st.error("Erro ao criar o ChromaDB")
            else:
                st.warning("Nenhum documento para processar.")
    
    if "vectorstore" not in st.session_state:
        with st.spinner("Carregando base de conhecimento..."):
            persist_dir = get_chromadb_folder(selected_model, chunk_size, k_documents, chunk_overlap)
            vectorstore = load_chromadb(selected_model, persist_dir)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.success("Base de conhecimento carregada com sucesso!")
            else:
                documents, num_docs = fetch_and_process_urls(st.session_state.document_urls)
                if documents:
                    vectorstore = initialize_chromadb(documents, selected_model, chunk_size, chunk_overlap, persist_dir)
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.success(f"Base de conhecimento inicializada com sucesso! {num_docs} documentos processados.")
                    else:
                        st.error("Erro ao criar o ChromaDB")
                else:
                    st.error("Erro ao inicializar a base de conhecimento.")
                    return
    
    current_params = (selected_model, chunk_size, k_documents, chunk_overlap)
    if "current_params" not in st.session_state or st.session_state.current_params != current_params:
        with st.spinner("Atualizando parâmetros..."):
            persist_dir = get_chromadb_folder(selected_model, chunk_size, k_documents, chunk_overlap)
            vectorstore = load_chromadb(selected_model, persist_dir)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.current_params = current_params
                st.success("Parâmetros atualizados com sucesso!")
            else:
                st.error("Erro ao atualizar parâmetros.")
                return
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Digite sua pergunta"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                vectorstore = st.session_state.vectorstore
                search_kwargs = {"k": k_documents}
                qa_chain = create_retrieval_chain(vectorstore, selected_model, search_kwargs, PROMPT_TEMPLATE)
                
                response = qa_chain.invoke({"query": prompt})
                answer = response["result"]
                source_docs = response["source_documents"]
                
                st.markdown(answer)
                
                if source_docs:
                    with st.expander("Fontes consultadas"):
                        for i, doc in enumerate(source_docs, 1):
                            st.markdown(f"**Fonte {i}:** [{doc.metadata['source_id']}]({doc.metadata['source']})")
                            st.markdown(f"Trecho: {doc.page_content[:200]}...")
        
        st.session_state.messages.append({"role": "assistant", "content": answer})
    
    if st.sidebar.button("Limpar Histórico"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    streamlit_app()
