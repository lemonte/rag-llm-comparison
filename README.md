# Guia de Execução

## Pré-requisitos
- Python instalado (recomendado Python 3.10+)
- Dependências listadas em `requirements.txt`

## Ambiente virtual (opcional)
Crie e ative um ambiente virtual para isolar as dependências.

macOS/Linux:
```bash
python -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

## Instalar dependências
Com o ambiente virtual ativo (se optar por usar):
```bash
pip install -r requirements.txt
```

## Executar a aplicação
Execute o `main2.py` com o Streamlit:
```bash
streamlit run main2.py
```
Se o comando `streamlit` não estiver no PATH, use:
```bash
python -m streamlit run main2.py
```

Ao iniciar, o Streamlit exibirá uma URL local (por exemplo, `http://localhost:8501`). Abra-a no navegador para usar a aplicação.
