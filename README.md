# Sistema RAG com Streamlit, Ollama e ChromaDB

Sistema de Retrieval Augmented Generation (RAG) que permite fazer consultas sobre documentação através de web scraping, armazenamento em base vetorial (ChromaDB) e geração de respostas usando Ollama local.

## Características

- **Web Scraping**: Extração automática de conteúdo de URLs fornecidas
- **Base Vetorial**: Armazenamento em ChromaDB com embeddings automáticos (all-MiniLM-L6-v2)
- **RAG**: Respostas baseadas apenas nos documentos recuperados
- **Interface Web**: Interface amigável com Streamlit
- **Configurável**: Parâmetros ajustáveis via código e interface

## Pré-requisitos

- Python 3.8 ou superior
- Ollama instalado e rodando localmente
- Modelo LLM baixado no Ollama (ex: llama2, mistral, etc.)

## Setup e Instalação

1. **Criar ambiente virtual:**
   ```bash
   python -m venv .venv
   ```

2. **Ativar ambiente virtual:**
   - Linux/Mac:
     ```bash
     source .venv/bin/activate
     ```
   - Windows:
     ```bash
     .venv\Scripts\activate
     ```

3. **Instalar dependências:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar o projeto:**
   - Edite `config.py` para ajustar:
     - Nome do modelo Ollama
     - Tamanho de chunks
     - Sobreposição entre chunks
     - Número de documentos recuperados
     - Template do prompt
     - Lista de URLs para scraping

5. **Executar aplicação:**
   ```bash
   streamlit run app.py
   ```

## Uso

1. Adicione as URLs da documentação que deseja indexar
2. Clique em "Processar Documentos" para fazer scraping e indexação
3. Faça consultas sobre a documentação indexada
4. O sistema retornará respostas baseadas apenas nos documentos recuperados

## Estrutura do Projeto

- `app.py`: Interface Streamlit principal
- `config.py`: Variáveis de configuração
- `scraper.py`: Funções de web scraping
- `rag.py`: Gerenciamento de ChromaDB e RAG
- `requirements.txt`: Dependências do projeto

## Notas

- O ChromaDB usa o modelo all-MiniLM-L6-v2 por padrão para embeddings
- Os dados são armazenados localmente na pasta `chroma_db/`
- Certifique-se de que o Ollama está rodando antes de fazer consultas







