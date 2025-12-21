import streamlit as st
import ollama
import time
from config import (
    OLLAMA_MODEL_SINGLE as OLLAMA_MODEL,
    CHUNK_SIZE_SINGLE as CHUNK_SIZE,
    CHUNK_OVERLAP_SINGLE as CHUNK_OVERLAP,
    NUM_RETRIEVED_DOCS_SINGLE as NUM_RETRIEVED_DOCS,
    PROMPT_TEMPLATE,
    REFERENCE_URLS,
    LOCAL_FILES,
    OPENAI_API_KEY,
    OPENAI_VECTOR_STORE_ID
)
from openai import OpenAI
from scraper import scrape_multiple_urls, read_multiple_local_files
from rag import RAGSystem
from evaluation import llm_as_judge_evaluate_response, calculate_llm_judge_score

st.set_page_config(
    page_title="Sistema RAG",
    page_icon="🔍",
    layout="centered"
)

def get_persist_directory(chunk_size: int, chunk_overlap: int) -> str:
    return f"chroma_db_{chunk_size}_{chunk_overlap}"

persist_dir = get_persist_directory(CHUNK_SIZE, CHUNK_OVERLAP)

@st.cache_resource
def get_rag_system(persist_directory):
    return RAGSystem(persist_directory=persist_directory)

rag_system = get_rag_system(persist_dir)

def needs_indexing():
    has_urls = REFERENCE_URLS and len(REFERENCE_URLS) > 0
    has_local_files = LOCAL_FILES and len(LOCAL_FILES) > 0
    
    if not has_urls and not has_local_files:
        return False, "Nenhuma URL ou arquivo local configurado em config.py"
    
    collection_info = rag_system.get_collection_info()
    if collection_info["total_chunks"] > 0:
        return False, None
    return True, "Nenhum dado indexado encontrado"

if 'indexed' not in st.session_state:
    needs_index, message = needs_indexing()
    
    if needs_index:
        has_urls = REFERENCE_URLS and len(REFERENCE_URLS) > 0
        has_local_files = LOCAL_FILES and len(LOCAL_FILES) > 0
        
        if not has_urls and not has_local_files:
            st.session_state.indexed = False
            st.warning("⚠️ Configure as URLs ou arquivos locais em config.py")
        else:
            index_placeholder = st.empty()
            with index_placeholder.container():
                with st.spinner(f"Indexando documentos... {message if message else ''}"):
                    all_documents = {}
                    
                    # Processa URLs
                    if has_urls:
                        url_documents = scrape_multiple_urls(REFERENCE_URLS, delay=1.0)
                        all_documents.update(url_documents)
                    
                    # Processa arquivos locais
                    if has_local_files:
                        local_documents = read_multiple_local_files(LOCAL_FILES)
                        all_documents.update(local_documents)
                    
                    if all_documents:
                        rag_system.index_documents(
                            all_documents,
                            chunk_size=CHUNK_SIZE,
                            chunk_overlap=CHUNK_OVERLAP,
                            clear_existing=True
                        )
                        st.session_state.indexed = True
                        index_placeholder.success(f"✓ {len(all_documents)} documento(s) indexado(s) com sucesso!")
                    else:
                        st.error("Erro ao extrair conteúdo das URLs ou arquivos locais")
                        st.session_state.indexed = False
    else:
        st.session_state.indexed = True

st.title("🔍 Sistema RAG")
st.markdown("Faça perguntas sobre a documentação indexada")

with st.sidebar:
    st.header("ℹ️ Informações")
    
    collection_info = rag_system.get_collection_info()
    st.metric("Chunks Indexados", collection_info["total_chunks"])
    
    st.divider()
    
    st.subheader("Configuração")
    st.text(f"Modelo: {OLLAMA_MODEL}")
    st.text(f"Chunk Size: {CHUNK_SIZE}")
    st.text(f"Overlap: {CHUNK_OVERLAP}")
    st.text(f"URLs: {len(REFERENCE_URLS) if REFERENCE_URLS else 0}")
    st.text(f"Arquivos Locais: {len(LOCAL_FILES) if LOCAL_FILES else 0}")
    
    st.divider()
    
    if REFERENCE_URLS:
        st.subheader("URLs Indexadas")
        for url in REFERENCE_URLS:
            st.text(f"• {url[:50]}..." if len(url) > 50 else f"• {url}")
    
    if LOCAL_FILES:
        st.subheader("Arquivos Locais Indexados")
        for file in LOCAL_FILES:
            st.text(f"• {file}")

collection_info = rag_system.get_collection_info()
if collection_info["total_chunks"] == 0:
    st.warning("⚠️ Nenhum documento indexado. Verifique as URLs ou arquivos locais em config.py e recarregue a página.")
else:
    query = st.text_input(
        "Sua pergunta",
        placeholder="Ex: Como funciona o sistema?",
        help="Digite sua pergunta sobre a documentação indexada"
    )
    
    if st.button("🔍 Buscar", type="primary") or query:
        if query:
            with st.spinner("Processando consulta..."):
                try:
                    retrieved_docs = rag_system.retrieve_documents(
                        query,
                        n_results=NUM_RETRIEVED_DOCS
                    )
                    
                    if retrieved_docs:
                        context = rag_system.format_context(retrieved_docs)
                        prompt = PROMPT_TEMPLATE.format(
                            context=context,
                            question=query
                        )
                        
                        try:
                            # Mede o tempo de geração
                            start_time = time.time()
                            
                            if OLLAMA_MODEL == "openai-gpt-4.1":
                                if not OPENAI_API_KEY:
                                    st.error("OPENAI_API_KEY não configurada")
                                    st.stop()
                                
                                client = OpenAI(api_key=OPENAI_API_KEY)
                                
                                response = client.responses.create(
                                    model="gpt-4.1",
                                    input=query,
                                    tools=[{
                                        "type": "file_search",
                                        "vector_store_ids": [OPENAI_VECTOR_STORE_ID],
                                        "max_num_results": NUM_RETRIEVED_DOCS
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
                            else:
                                response = ollama.generate(
                                    model=OLLAMA_MODEL,
                                    prompt=prompt,
                                    stream=False,
                                    options={
                                        "temperature": 0
                                    }
                                )
                                answer = response.get('response', '')
                            
                            generation_time = time.time() - start_time
                            
                            # Layout: Resposta com tempo ao lado
                            col1, col2 = st.columns([3, 1])
                            with col1:
                            st.subheader("📝 Resposta")
                            with col2:
                                st.metric("⏱️ Tempo", f"{generation_time:.2f}s")
                            
                            st.write(answer)
                            
                            # Avalia documentos usando LLM-as-a-Judge
                            llm_judge_score = None
                            try:
                                with st.spinner("Avaliando documentos com LLM-as-a-Judge..."):
                                    llm_judge_score = calculate_llm_judge_score(
                                        question=query,
                                        retrieved_docs=retrieved_docs,
                                        model="gpt-4o-mini"
                                    )
                            except Exception as e:
                                st.warning(f"Não foi possível avaliar os documentos: {e}")
                                st.info("Certifique-se de que a variável de ambiente OPENAI_API_KEY está configurada no arquivo .env para usar LLM-as-a-Judge.")
                            
                            # Avalia a resposta final usando LLM-as-a-Judge
                            response_score = None
                            feedback = None
                            try:
                                with st.spinner("Avaliando qualidade da resposta (LLM-as-a-Judge)..."):
                                    result = llm_as_judge_evaluate_response(
                                        question=query,
                                        context=context,
                                        answer=answer,
                                        model="gpt-4o-mini",
                                        return_feedback=True
                                    )
                                    if isinstance(result, tuple):
                                        response_score, feedback = result
                                    else:
                                        response_score = result
                                        feedback = None
                            except Exception as e:
                                st.warning(f"Não foi possível avaliar a resposta: {e}")
                                st.info("Certifique-se de que a variável de ambiente OPENAI_API_KEY está configurada no arquivo .env para usar LLM-as-a-Judge.")
                            
                            # Exibe métricas de avaliação
                            st.markdown("---")
                            
                            # LLM-as-a-Judge (Documentos)
                            st.markdown("**LLM-as-a-Judge (Documentos):**")
                            if llm_judge_score is not None:
                                st.write(f"Nota: {llm_judge_score:.2f} / 5.00")
                                st.caption("Utiliza LLMs (GPT) para avaliar semanticamente a relevância dos documentos recuperados")
                            else:
                                st.write("Nota: Não disponível")
                                st.caption("Configure OPENAI_API_KEY no arquivo .env para usar LLM-as-a-Judge")
                            
                            # Nota Final (LLM-as-a-Judge)
                            st.markdown("**Nota Final (LLM-as-a-Judge):**")
                            if response_score is not None:
                                st.write(f"Nota: {response_score:.2f} / 5.00")
                                st.caption("Utiliza LLM para avaliar semanticamente a qualidade da resposta final, considerando o contexto fornecido, a pergunta original e a resposta gerada")
                                
                                # Exibe feedback se disponível
                                if feedback:
                                    with st.expander("📝 Feedback da Avaliação", expanded=True):
                                        st.write(feedback)
                            else:
                                st.write("Nota: Não disponível")
                                st.caption("Configure OPENAI_API_KEY no arquivo .env para usar LLM-as-a-Judge")
                            
                            with st.expander("📄 Documentos Recuperados", expanded=False):
                                for i, doc in enumerate(retrieved_docs, 1):
                                    st.markdown(f"**Documento {i}**")
                                    st.markdown(f"*Fonte: {doc['metadata'].get('url', 'URL desconhecida')}*")
                                    if doc['distance'] is not None:
                                        st.markdown(f"*Similaridade: {1 - doc['distance']:.4f}*")
                                    st.markdown(doc['document'][:500] + "..." if len(doc['document']) > 500 else doc['document'])
                                    st.divider()
                            
                        except Exception as e:
                            st.error(f"Erro ao gerar resposta com Ollama: {e}")
                            st.info("Certifique-se de que:")
                            st.info("1. Ollama está rodando localmente")
                            st.info(f"2. O modelo '{OLLAMA_MODEL}' está instalado")
                            st.info("3. Você pode testar com: ollama list")
                    else:
                        st.warning("Nenhum documento relevante foi encontrado para sua consulta.")
                        
                except Exception as e:
                    st.error(f"Erro ao processar consulta: {e}")

st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        Sistema RAG | Ollama + ChromaDB
    </div>
    """,
    unsafe_allow_html=True
)
