# Guia de Execução - Sistema de Avaliação RAG

Este guia explica como configurar e executar o sistema completo de avaliação.

## 📋 Pré-requisitos

1. **Python 3.8 ou superior**
2. **Ollama instalado e rodando**
   - Baixe em: https://ollama.ai
   - Verifique se está rodando: `ollama list`

## 🔧 Passo 1: Configurar Ambiente Python

```bash
# Criar ambiente virtual
python -m venv .venv

# Ativar ambiente virtual
# Linux/Mac:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt
```

## 🤖 Passo 2: Configurar Ollama e Baixar Modelos

### 2.1 Baixar modelo de embeddings
```bash
ollama pull mahonzhan/all-MiniLM-L6-v2
```

### 2.2 Baixar modelo Mistral (para geração de respostas)
```bash
ollama pull mistral
```

### 2.3 Verificar modelos instalados
```bash
ollama list
```

Você deve ver pelo menos:
- `mahonzhan/all-MiniLM-L6-v2`
- `mistral`

## 🔑 Passo 3: Configurar API Keys (Opcional mas Recomendado)

### 3.1 Criar arquivo .env
```bash
cp .env-example .env
```

### 3.2 Editar .env e adicionar sua OpenAI API Key
```bash
OPENAI_API_KEY=sk-proj-sua-chave-aqui
```

**Nota:** A OpenAI API Key é necessária para:
- LLM-as-a-Judge (avaliação de qualidade das respostas)
- Modelo GPT-4.1 (se configurado)

Se não configurar, o sistema funcionará, mas sem essas funcionalidades.

## 📝 Passo 4: Verificar Configurações

Edite `config.py` se necessário para ajustar:
- **OLLAMA_MODEL**: Modelos a serem testados (padrão: `["mistral", "openai-gpt-4.1"]`)
- **CHUNK_SIZE**: Tamanhos de chunk (padrão: `[1200]`)
- **CHUNK_OVERLAP**: Overlap entre chunks (padrão: `[200]`)
- **NUM_RETRIEVED_DOCS**: Número de documentos recuperados (padrão: `[3]`)
- **REFERENCE_URLS**: URLs para fazer scraping
- **LOCAL_FILES**: Arquivos locais para indexar

## 🚀 Passo 5: Executar Avaliação Completa

### Executar avaliação completa (recomendado)
```bash
python comprehensive_evaluation.py
```

Este script irá:
1. ✅ Indexar documentos (se necessário)
2. ✅ Executar todas as perguntas com todos os modelos configurados
3. ✅ Gerar arquivo CSV com resultados: `avaliacao_completa_YYYYMMDD_HHMMSS.csv`
4. ✅ Calcular similaridade entre respostas do Mistral e GPT-4.1
5. ✅ Gerar arquivo CSV de similaridade: `similaridade_respostas_YYYYMMDD_HHMMSS.csv`

### Executar interface web (opcional)
```bash
streamlit run app.py
```

Acesse: http://localhost:8501

## 📊 Resultados Gerados

### 1. Arquivo de Avaliação Completa
`avaliacao_completa_YYYYMMDD_HHMMSS.csv`

Contém:
- Número da questão
- Pergunta
- Modelo usado
- Parâmetros (chunk_size, chunk_overlap, num_docs)
- Resposta gerada
- Tempo de resposta
- Avaliações (LLM-as-a-Judge)
- Documentos recuperados
- Contexto usado

### 2. Arquivo de Similaridade
`similaridade_respostas_YYYYMMDD_HHMMSS.csv`

Contém:
- Número da questão
- Pergunta
- Resposta do Mistral
- Tempo do Mistral
- Resposta do GPT-4.1
- Tempo do GPT-4.1
- Similaridade (0-1) entre as respostas

## ⚠️ Troubleshooting

### Erro: "Model not found"
- Verifique se o modelo foi baixado: `ollama list`
- Baixe o modelo: `ollama pull mahonzhan/all-MiniLM-L6-v2`

### Erro: "Ollama connection refused"
- Verifique se o Ollama está rodando: `ollama list`
- Inicie o Ollama se necessário

### Erro: "OPENAI_API_KEY não configurada"
- Crie arquivo `.env` com sua chave
- Ou remova `"openai-gpt-4.1"` da lista de modelos em `config.py`

### Erro ao indexar documentos
- Verifique se as URLs em `REFERENCE_URLS` estão acessíveis
- Verifique se os arquivos em `LOCAL_FILES` existem

## 📁 Estrutura de Arquivos Importantes

```
novo-teste/
├── comprehensive_evaluation.py  # Script principal de avaliação
├── config.py                   # Configurações
├── rag.py                      # Sistema RAG
├── scraper.py                  # Web scraping
├── evaluation.py               # Funções de avaliação
├── app.py                      # Interface Streamlit
├── requirements.txt            # Dependências Python
├── .env                        # API Keys (criar a partir de .env-example)
└── chroma_db_*/                # Bancos de dados vetoriais (gerados automaticamente)
```

## 🎯 Resumo Rápido

```bash
# 1. Ambiente
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows
pip install -r requirements.txt

# 2. Ollama
ollama pull mahonzhan/all-MiniLM-L6-v2
ollama pull mistral

# 3. Configurar .env (opcional)
cp .env-example .env
# Editar .env e adicionar OPENAI_API_KEY

# 4. Executar
python comprehensive_evaluation.py
```

## 📞 Suporte

Se encontrar problemas:
1. Verifique se todos os pré-requisitos estão instalados
2. Verifique se os modelos do Ollama foram baixados
3. Verifique os logs de erro no terminal
4. Certifique-se de que o arquivo `.env` está configurado corretamente (se usar OpenAI)
