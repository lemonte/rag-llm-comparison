import streamlit as st
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from bs4 import BeautifulSoup
import requests
import os
import pickle
import subprocess
import pandas as pd
import time
from datetime import datetime
from PyPDF2 import PdfReader
import markdown
import io
import json
import threading
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configurações do sistema
DOCUMENT_URLS = [
    "https://docs.pybricks.com/en/latest/hubs/primehub.html",
    'https://docs.pybricks.com/en/stable/pupdevices/dcmotor.html',
]


# model_options = [ "llama3.2","llama3.1:405b", "llama3:70b","mistral", "deepseek-r1", "phi3", "neural-chat", "solar", "moondream",  "deepseek-r1:671b"]

MODEL_OPTIONS = ["neural-chat"]

# chunk_size
CHUNK_SIZE = 1000 # 1200

# k_documents
K_DOCUMENTS = 3 # 4

# chunk_overlap
CHUNK_OVERLAP = 100 # 200



CHROMADB_FOLDER = f"chromadb_storage_{MODEL_OPTIONS[0]}_{CHUNK_SIZE}_{K_DOCUMENTS}_{CHUNK_OVERLAP}"
METRICS_FILE = f"metricas_rag_{MODEL_OPTIONS[0]}_{CHUNK_SIZE}_{K_DOCUMENTS}_{CHUNK_OVERLAP}.xlsx"
ALLOWED_EXTENSIONS = ['.pdf', '.md', '.txt']
DOCUMENTS_FILE = "documentos_processados.json"


# Lista de perguntas para teste
PERGUNTAS_TESTE = [
    # Prime Hub e suas funcionalidades
    "Quais são os parâmetros aceitos na inicialização do Prime Hub?",
    "Como acender a luz de status do Prime Hub em uma cor específica?",
    "Quais funcionalidades estão disponíveis no Prime Hub e como utilizá-las?",
    "Como utilizar o Bluetooth do Prime Hub para enviar e receber dados?",
    "Quais sensores podem ser conectados ao Prime Hub?",
    "Como configurar o botão central do Prime Hub para executar uma função personalizada?",
    "Como ler os valores do IMU (Unidade de Medida Inercial) do Prime Hub?",
    "O que a função imu.heading() retorna e como utilizá-la corretamente?",
    "Como configurar a exibição de números no display do Prime Hub?",
    "Como exibir uma animação na matriz de LEDs do Prime Hub?",
    "Quais são as opções disponíveis para o controle da luz de status no Prime Hub?",

    # Motores e Sensores
    "Como controlar um motor DC sem sensor de rotação?",
    "Qual a diferença entre brake() e stop() ao parar um motor?",
    "Como usar um sensor de distância no Pybricks?",
    "Quais sensores podem medir a inclinação do Prime Hub?",
    "Como medir a força aplicada com um sensor de força no Pybricks?",

    # Comunicação e Integração
    "Como transmitir dados entre dois Prime Hubs usando Bluetooth?",
    "O que é ble.signal_strength() e como usá-lo?",
    "Como armazenar e recuperar dados na memória persistente do Prime Hub?",

    # Energia e Sistema
    "Como verificar a voltagem da bateria do Prime Hub?",
    "Como saber se o Prime Hub está sendo carregado?",
    "Quais são os motivos que podem levar o Prime Hub a reiniciar?",

    # Controle e Movimentação de Robôs
    "Como programar um robô para andar em linha reta usando o Pybricks?",
    "Como fazer um robô girar em torno do próprio eixo no Pybricks?"
]


def salvar_metricas_inicializacao(tempo_inicializacao):
    """Salva as métricas de inicialização do sistema"""
    df = pd.DataFrame({
        'data_hora': [datetime.now()],
        'tempo_inicializacao': [tempo_inicializacao]
    })
    
    if os.path.exists(METRICS_FILE):
        try:
            with pd.ExcelWriter(METRICS_FILE, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                try:
                    df_existente = pd.read_excel(METRICS_FILE, sheet_name='inicializacao')
                    df_final = pd.concat([df_existente, df], ignore_index=True)
                except:
                    df_final = df
                
                df_final.to_excel(writer, sheet_name='inicializacao', index=False)
        except Exception as e:
            print(f"Erro ao atualizar arquivo existente: {str(e)}")
            df.to_excel(METRICS_FILE, sheet_name='inicializacao', index=False)
    else:
        df.to_excel(METRICS_FILE, sheet_name='inicializacao', index=False)
    
    # print(f"Métricas de inicialização salvas com sucesso: tempo={tempo_inicializacao:.2f}s")

def salvar_metrica_pergunta(pergunta, tempo_resposta, resposta, fontes_usadas, model_name, k_documents, chunk_size, chunk_overlap):
    """Salva as métricas de uma pergunta individual"""
    novo_registro = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pergunta': pergunta,
        'resposta': resposta,
        'tempo_resposta': tempo_resposta,
        'fontes_usadas': ', '.join(fontes_usadas),
        'modelo': model_name,
        'k_documents': k_documents,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap
    }
    df_novo = pd.DataFrame([novo_registro])
    
    if os.path.exists(METRICS_FILE):
        try:
            with pd.ExcelWriter(METRICS_FILE, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                try:
                    df_existente = pd.read_excel(METRICS_FILE, sheet_name='perguntas')
                    df_final = pd.concat([df_existente, df_novo], ignore_index=True)
                except:
                    df_final = df_novo
                
                df_final.to_excel(writer, sheet_name='perguntas', index=False)
        except Exception as e:
            print(f"Erro ao atualizar arquivo existente: {str(e)}")
            df_novo.to_excel(METRICS_FILE, sheet_name='perguntas', index=False)
    else:
        df_novo.to_excel(METRICS_FILE, sheet_name='perguntas', index=False)
    
    # print(f"Métricas de pergunta salvas com sucesso: {novo_registro}")

def salvar_metrica_processamento(tempo_proc, tipo_operacao, num_documentos, modelo=None, chunk_size=None, chunk_overlap=None):
    """Salva métricas de processamento na planilha"""
    novo_registro = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'tipo_operacao': tipo_operacao,
        'tempo_processamento': tempo_proc,
        'num_documentos': num_documentos,
        'modelo': modelo,
        'chunk_size': chunk_size,
        'chunk_overlap': chunk_overlap
    }
    df_novo = pd.DataFrame([novo_registro])
    
    if os.path.exists(METRICS_FILE):
        try:
            with pd.ExcelWriter(METRICS_FILE, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
                try:
                    df_existente = pd.read_excel(METRICS_FILE, sheet_name='processamento')
                    df_final = pd.concat([df_existente, df_novo], ignore_index=True)
                except:
                    df_final = df_novo
                
                df_final.to_excel(writer, sheet_name='processamento', index=False)
        except Exception as e:
            print(f"Erro ao atualizar arquivo existente: {str(e)}")
            df_novo.to_excel(METRICS_FILE, sheet_name='processamento', index=False)
    else:
        df_novo.to_excel(METRICS_FILE, sheet_name='processamento', index=False)
    
    # print(f"Métricas de processamento salvas com sucesso: {novo_registro}")

def executar_teste_automatico(qa_chain, model_name, k_documents, chunk_size, chunk_overlap):
    """Executa o teste automático com a lista de perguntas"""
    resultados = []
    for pergunta in PERGUNTAS_TESTE:
        inicio = time.time()
        response = qa_chain.invoke({"query": pergunta})
        tempo = time.time() - inicio
        
        fontes = [doc.metadata['source_id'] for doc in response["source_documents"]]
        salvar_metrica_pergunta(pergunta, tempo, response["result"], fontes, 
                              model_name, k_documents, chunk_size, chunk_overlap)
        
        resultados.append({
            'pergunta': pergunta,
            'tempo': tempo,
            'resposta': response["result"],
            'fontes': fontes
        })
    return resultados

def normalizar_documento(item):
    """Normaliza o formato do documento para salvar no JSON"""
    if isinstance(item, str):
        if item.startswith('http'):
            return {
                'name': item,
                'type': 'url',
                'text': ''  # texto será carregado durante o processamento
            }
        else:
            return {
                'name': item,
                'type': 'file',
                'text': ''  # texto será carregado durante o processamento
            }
    elif isinstance(item, dict):
        return {
            'name': item.get('name', ''),
            'type': item.get('type', 'unknown'),
            'text': ''  # texto será carregado durante o processamento
        }
    return None

def salvar_documentos(documentos):
    """Salva a lista de documentos em um arquivo JSON"""
    try:
        # Normaliza todos os documentos antes de salvar
        docs_normalizados = [
            doc for doc in (normalizar_documento(d) for d in documentos)
            if doc is not None
        ]
        with open(DOCUMENTS_FILE, 'w') as f:
            json.dump(docs_normalizados, f)
            print(f"Documentos salvos no JSON: {docs_normalizados}")
    except Exception as e:
        print(f"Erro ao salvar documentos: {str(e)}")

def carregar_documentos():
    """Carrega a lista de documentos do arquivo JSON"""
    try:
        if os.path.exists(DOCUMENTS_FILE):
            with open(DOCUMENTS_FILE, 'r') as f:
                documentos = json.load(f)
                print(f"Documentos carregados do JSON: {documentos}")
                return documentos
    except Exception as e:
        print(f"Erro ao carregar documentos: {str(e)}")
    return []

def get_chromadb_documents(vectorstore):
    """Recupera a lista de documentos do ChromaDB"""
    try:
        docs = vectorstore._collection.get()
        documentos = []
        documentos_processados = set()  
        
        for metadata in docs['metadatas']:
            source = metadata.get('source', '')
            if source and source not in documentos_processados:
                if metadata.get('type') == 'url':
                    documentos.append(source)
                else:
                    documentos.append({
                        'name': source,
                        'type': metadata.get('type', 'unknown'),
                        'text': ''  
                    })
                documentos_processados.add(source)
        
        return documentos
    except Exception as e:
        st.error(f"Erro ao recuperar documentos do ChromaDB: {e}")
        return []

# def initialize_chromadb(documents, model_name, chunk_size, chunk_overlap):
#     """Inicializa o ChromaDB com os documentos fornecidos"""
#     texts = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
#     embeddings = OllamaEmbeddings(model=model_name)
#     vectorstore = Chroma(persist_directory=CHROMADB_FOLDER, embedding_function=embeddings)
#     if texts:
#         vectorstore.add_documents(texts)
#     return vectorstore

def load_chromadb(model_name):
    """Carrega uma instância existente do ChromaDB"""
    if os.path.exists(CHROMADB_FOLDER):
        embeddings = OllamaEmbeddings(model=model_name)
        return Chroma(persist_directory=CHROMADB_FOLDER, embedding_function=embeddings)
    return None

def processar_documento_async(doc_info, selected_model, chunk_size, chunk_overlap):
    """Processa um novo documento de forma assíncrona"""
    try:
        inicio_proc = time.time()
        
        # Carrega o vectorstore existente
        vectorstore = load_chromadb(selected_model)
        if not vectorstore:
            print("Erro: Não foi possível carregar o ChromaDB")
            return
        
        if isinstance(doc_info, str):
            text = process_url(doc_info)
            metadata = {
                "source": doc_info,
                "source_id": doc_info.split("/")[-1].replace(".html", ""),
                "type": "url"
            }
        else:
            text = doc_info['text']
            metadata = {
                "source": doc_info['name'],
                "source_id": doc_info['name'],
                "type": doc_info['type']
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
            
            # Adiciona os novos documentos ao vectorstore
            vectorstore.add_documents(texts)
            
            tempo_proc = time.time() - inicio_proc
            salvar_metrica_processamento(
                tempo_proc, 
                "processamento_novo_documento", 
                1,
                selected_model, 
                chunk_size, 
                chunk_overlap
            )
            
            # Atualiza o status de processamento usando uma cópia local do estado
            doc_name = doc_info if isinstance(doc_info, str) else doc_info['name']
            documentos_processando = st.session_state.get('documentos_processando', set())
            if doc_name in documentos_processando:
                documentos_processando.remove(doc_name)
                st.session_state.documentos_processando = documentos_processando
            
            print(f"Documento processado com sucesso: {doc_name}")
            
    except Exception as e:
        print(f"Erro no processamento assíncrono: {str(e)}")
        # Remove o documento da lista em caso de erro
        doc_name = doc_info if isinstance(doc_info, str) else doc_info['name']
        documentos_processando = st.session_state.get('documentos_processando', set())
        if doc_name in documentos_processando:
            documentos_processando.remove(doc_name)
            st.session_state.documentos_processando = documentos_processando

def iniciar_processamento_async(doc_info, selected_model, chunk_size, chunk_overlap):
    """Inicia o processamento assíncrono em um contexto seguro"""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    
    # Cria um executor dedicado para esta tarefa
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ProcessadorDoc")
    
    try:
        # Submete a tarefa para execução assíncrona
        future = executor.submit(
            processar_documento_async,
            doc_info,
            selected_model,
            chunk_size,
            chunk_overlap
        )
        
        # Registra callbacks para sucesso/erro
        def done_callback(future):
            try:
                future.result()  # Verifica se houve exceção
                print("Processamento assíncrono concluído com sucesso")
            except Exception as e:
                print(f"Erro no callback do processamento: {str(e)}")
            finally:
                executor.shutdown(wait=False)
        
        future.add_done_callback(done_callback)
        
    except Exception as e:
        print(f"Erro ao iniciar processamento assíncrono: {str(e)}")
        executor.shutdown(wait=False)

def process_pdf(file):
    """Processa arquivo PDF e retorna o texto extraído"""
    try:
        pdf_reader = PdfReader(io.BytesIO(file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Erro ao processar PDF: {e}")
        return None

def process_markdown(file):
    """Processa arquivo Markdown e retorna o texto extraído"""
    try:
        content = file.read().decode('utf-8')
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        return soup.get_text()
    except Exception as e:
        st.error(f"Erro ao processar Markdown: {e}")
        return None

def process_text(file):
    """Processa arquivo de texto simples"""
    try:
        return file.read().decode('utf-8')
    except Exception as e:
        st.error(f"Erro ao processar arquivo de texto: {e}")
        return None

def process_url(url):
    """Processa URL e retorna o texto extraído"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        return soup.get_text()
    except Exception as e:
        st.error(f"Erro ao processar URL {url}: {e}")
        return None

def process_file(file):
    """Processa arquivo baseado em sua extensão"""
    file_extension = os.path.splitext(file.name)[1].lower()
    
    if file_extension == '.pdf':
        return process_pdf(file)
    elif file_extension == '.md':
        return process_markdown(file)
    elif file_extension == '.txt':
        return process_text(file)
    else:
        st.error(f"Formato de arquivo não suportado: {file_extension}")
        return None

def carregar_texto_arquivo(nome_arquivo):
    """Carrega o texto de um arquivo local"""
    try:
        with open(nome_arquivo, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Erro ao ler arquivo {nome_arquivo}: {str(e)}")
        return None

def fetch_and_process_urls(urls):
    """Processa URLs e documentos locais"""
    documents = []
    num_docs = 0
    print(f"\nIniciando processamento de {len(urls)} documentos")
    
    for item in urls:
        try:
            # Normaliza o item se necessário
            if isinstance(item, str):
                doc_info = normalizar_documento(item)
            else:
                doc_info = item
            
            if not doc_info or not doc_info.get('name'):
                print(f"Item ignorado (formato inválido): {item}")
                continue
            
            nome = doc_info['name']
            tipo = doc_info['type']
            
            # Processa URLs
            if tipo == 'url' or nome.startswith('http'):
                print(f"Processando URL: {nome}")
                text = process_url(nome)
                if text:
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": nome,
                            "source_id": nome.split("/")[-1].replace(".html", ""),
                            "type": "url"
                        }
                    )
                    documents.append(doc)
                    num_docs += 1
                    print(f"URL processada com sucesso: {nome}")
                else:
                    print(f"URL ignorada (sem texto): {nome}")
            
            # Processa arquivos locais
            else:
                print(f"Processando arquivo local: {nome}")
                text = carregar_texto_arquivo(nome)
                if text:
                    doc = Document(
                        page_content=text,
                        metadata={
                            "source": nome,
                            "source_id": os.path.basename(nome),
                            "type": tipo
                        }
                    )
                    documents.append(doc)
                    num_docs += 1
                    print(f"Arquivo local processado com sucesso: {nome}")
                else:
                    print(f"Arquivo local ignorado (sem texto): {nome}")
            
        except Exception as e:
            print(f"Erro ao processar item {item}: {str(e)}")
    
    print(f"\nResumo do processamento:")
    print(f"Total de documentos processados: {num_docs}")
    print(f"Documentos gerados: {len(documents)}")
    return documents, num_docs

def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Divide os documentos em chunks menores"""
    print(f"Dividindo {len(documents)} documentos em chunks")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    
    texts = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc.page_content)
        texts.extend([
            Document(
                page_content=text,
                metadata=doc.metadata
            ) for text in split_texts
        ])
    
    print(f"Total de chunks gerados: {len(texts)}")
    return texts

def initialize_chromadb(documents, model_name, chunk_size, chunk_overlap):
    """Inicializa o ChromaDB com os documentos fornecidos"""
    print("Inicializando ChromaDB...")
    texts = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embeddings = OllamaEmbeddings(model=model_name)
    vectorstore = Chroma(persist_directory=CHROMADB_FOLDER, embedding_function=embeddings)
    if texts:
        print(f"Adicionando {len(texts)} chunks ao ChromaDB")
        vectorstore.add_documents(texts)
        print("Documentos adicionados com sucesso")
    return vectorstore

def create_retrieval_chain(vectorstore, model_name, search_kwargs):
    llm = OllamaLLM(model=model_name)
    
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
    
    template = """
    Use o contexto abaixo para responder à pergunta. 
    Se a informação estiver presente no contexto, inclua a citação usando [source_id].
    Se não souber a resposta, diga que não sabe.
    Responda no mesmo idioma da pergunta.
    
    Contexto: {context}
    
    Pergunta: {question}
    """
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={
            "prompt": PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
        }
    )
    return qa_chain


def loadModels():
    model_options = MODEL_OPTIONS
    # try:
    #     response = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
    #     output = response.stdout.strip()
    #     models = output.splitlines() 
    #     print(response.json())
    # except Exception as e:
    #     print(e)
    return model_options


# Aplicação Streamlit
def streamlit_app():
    st.title("🤖 Assistente de Documentação com RAG")
    
    salvar_metricas_inicializacao(time.time())
    model_options = loadModels()
    selected_model = st.sidebar.selectbox("Modelo:", model_options, index=0)
    
    chunk_size = st.sidebar.slider("Tamanho do Chunk:", min_value=100, max_value=2000, value=CHUNK_SIZE, step=100)
    chunk_overlap = st.sidebar.slider("Sobreposição do Chunk:", min_value=0, max_value=500, value=CHUNK_OVERLAP, step=10)
    
    k_documents = st.sidebar.slider("Número de documentos para recuperar:", min_value=1, max_value=10, value=K_DOCUMENTS)
    
    st.sidebar.header("📚 Gerenciar Documentos")
    
    if 'documentos_processando' not in st.session_state:
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
            doc_name = doc if isinstance(doc, str) else doc['name']
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
                st.session_state.inicio_proc = time.time()
                
                iniciar_processamento_async(new_url, selected_model, chunk_size, chunk_overlap)
                
                salvar_documentos(st.session_state.document_urls)
                st.success("URL adicionada com sucesso! Processando...")
                st.rerun()
            elif new_url in st.session_state.document_urls:
                st.warning("Esta URL já foi adicionada!")
    
    with st.sidebar.expander("📁 Upload de Arquivo"):
        uploaded_file = st.file_uploader("Escolha um arquivo", type=['pdf', 'md', 'txt'])
        if uploaded_file is not None:
            file_name = uploaded_file.name
            existing_files = [doc.get('name') if isinstance(doc, dict) else None 
                            for doc in st.session_state.document_urls]
            
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
                    st.session_state.inicio_proc = time.time()
                    
                    iniciar_processamento_async(doc_info, selected_model, chunk_size, chunk_overlap)
                    
                    salvar_documentos(st.session_state.document_urls)
                    st.success(f"Arquivo {file_name} enviado com sucesso! Processando...")
                    st.rerun()
            else:
                st.warning("Este arquivo já foi carregado!")
    
    if st.sidebar.button("🔄 Reprocessar Base de Conhecimento"):
        with st.spinner("Reprocessando todos os documentos..."):
            inicio_proc = time.time()
            try:
                print("\n=== Iniciando reprocessamento da base de conhecimento ===")
                print(f"Documentos a processar: {st.session_state.document_urls}")
                
                documents, num_docs = fetch_and_process_urls(st.session_state.document_urls)
                if documents:
                    vectorstore = initialize_chromadb(documents, selected_model, chunk_size, chunk_overlap)
                    if vectorstore:
                        tempo_proc = time.time() - inicio_proc
                        print(f"Tempo total de processamento: {tempo_proc:.2f} segundos")
                        
                        salvar_metrica_processamento(
                            tempo_proc,
                            "reprocessamento_documentos",
                            num_docs,
                            selected_model,
                            chunk_size,
                            chunk_overlap
                        )
                        
                        st.session_state.vectorstore = vectorstore
                        st.success(f"Base de conhecimento atualizada com sucesso em {tempo_proc:.2f} segundos! {num_docs} documentos processados.")
                        print("=== Reprocessamento concluído com sucesso ===\n")
                    else:
                        st.error("Erro ao criar o ChromaDB")
                else:
                    st.warning("Nenhum documento para processar.")
            except Exception as e:
                print(f"Erro durante o reprocessamento: {str(e)}")
                st.error(f"Erro ao reprocessar documentos: {str(e)}")
    
    if "vectorstore" not in st.session_state:
        with st.spinner("Carregando base de conhecimento..."):
            inicio = time.time()
            vectorstore = load_chromadb(selected_model)
            if vectorstore:
                tempo_total = time.time() - inicio
                st.session_state.vectorstore = vectorstore
                st.success(f"Base de conhecimento carregada com sucesso em {tempo_total:.2f} segundos!")
            else:
                inicio_proc = time.time()
                documents, num_docs = fetch_and_process_urls(st.session_state.document_urls)
                if documents:
                    vectorstore = initialize_chromadb(documents, selected_model, chunk_size, chunk_overlap)
                    if vectorstore:
                        tempo_proc = time.time() - inicio_proc
                        salvar_metrica_processamento(
                            tempo_proc,
                            "inicializacao_documentos",
                            num_docs,
                            selected_model,
                            chunk_size,
                            chunk_overlap
                        )
                        
                        st.session_state.vectorstore = vectorstore
                        st.success(f"Base de conhecimento inicializada com sucesso em {tempo_proc:.2f} segundos! {num_docs} documentos processados.")
                        qa_chain = create_retrieval_chain(st.session_state.vectorstore, selected_model, search_kwargs={"k": K_DOCUMENTS})
                        resultados = executar_teste_automatico(qa_chain, selected_model, K_DOCUMENTS, chunk_size, chunk_overlap)
                        st.write(resultados)
                        
                    else:
                        st.error("Erro ao criar o ChromaDB")
                else:
                    st.error("Erro ao inicializar a base de conhecimento.")
                    return
    
    if "current_model" not in st.session_state or st.session_state.current_model != selected_model:
        with st.spinner("Atualizando modelo..."):
            vectorstore = load_chromadb(selected_model)
            if vectorstore:
                st.session_state.vectorstore = vectorstore
                st.session_state.current_model = selected_model
                st.success("Modelo atualizado com sucesso!")
            else:
                st.error("Erro ao atualizar o modelo.")
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
                search_kwargs = {"k": k_documents}
                qa_chain = create_retrieval_chain(st.session_state.vectorstore, selected_model, search_kwargs)
                
                inicio = time.time()
                response = qa_chain.invoke({"query": prompt})
                tempo = time.time() - inicio
                fontes = [doc.metadata['source_id'] for doc in response["source_documents"]]
                salvar_metrica_pergunta(prompt, tempo, response["result"], fontes,
                                      selected_model, k_documents, chunk_size, chunk_overlap)
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

    if st.sidebar.button("Executar Teste Automático"):
        qa_chain = create_retrieval_chain(st.session_state.vectorstore, selected_model, search_kwargs={"k": k_documents})
        resultados = executar_teste_automatico(qa_chain, selected_model, k_documents, chunk_size, chunk_overlap)
        st.write(resultados)

if __name__ == "__main__":
    streamlit_app()
