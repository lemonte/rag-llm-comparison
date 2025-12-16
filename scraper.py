import requests
from bs4 import BeautifulSoup
from typing import Optional, Dict
import time
import os

def scrape_url(url: str, timeout: int = 30) -> Optional[str]:
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'lxml')
        
        for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
            script.decompose()
        
        text = soup.get_text()
        
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        return text
        
    except requests.exceptions.RequestException as e:
        print(f"Erro ao fazer requisição para {url}: {e}")
        return None
    except Exception as e:
        print(f"Erro ao processar {url}: {e}")
        return None

def scrape_multiple_urls(urls: list, delay: float = 1.0) -> dict:
    results = {}
    
    for i, url in enumerate(urls):
        print(f"Processando {i+1}/{len(urls)}: {url}")
        text = scrape_url(url)
        
        if text:
            results[url] = text
            print(f"✓ Conteúdo extraído: {len(text)} caracteres")
        else:
            print(f"✗ Falha ao extrair conteúdo de {url}")
        
        if i < len(urls) - 1:
            time.sleep(delay)
    
    return results


def read_local_file(file_path: str) -> Optional[str]:
    """
    Lê um arquivo local e retorna seu conteúdo como string.
    """
    try:
        if not os.path.exists(file_path):
            print(f"Arquivo não encontrado: {file_path}")
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return content
    except Exception as e:
        print(f"Erro ao ler arquivo {file_path}: {e}")
        return None


def read_multiple_local_files(file_paths: list) -> Dict[str, str]:
    """
    Lê múltiplos arquivos locais e retorna um dicionário com o caminho como chave.
    """
    results = {}
    
    for i, file_path in enumerate(file_paths):
        print(f"Processando arquivo {i+1}/{len(file_paths)}: {file_path}")
        content = read_local_file(file_path)
        
        if content:
            # Usa o caminho absoluto como chave
            abs_path = os.path.abspath(file_path)
            results[abs_path] = content
            print(f"✓ Conteúdo lido: {len(content)} caracteres")
        else:
            print(f"✗ Falha ao ler arquivo {file_path}")
    
    return results
