import requests
from bs4 import BeautifulSoup
from typing import Dict, List
import time
import os


class WebScraper:
    
    def __init__(self, delay: float = 1.0):
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def scrape_url(self, url: str) -> str:
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            cleaned_text = '\n'.join(lines)
            
            return cleaned_text
            
        except requests.RequestException as e:
            raise Exception(f"Erro ao fazer scraping de {url}: {e}")
    
    def save_to_file(self, content: str, filename: str) -> None:
        filepath = os.path.join(os.getcwd(), filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✓ Conteúdo salvo em: {filepath}")


def scrape_multiple_urls(urls: List[str], delay: float = 1.0) -> Dict[str, str]:
    scraper = WebScraper(delay=delay)
    documents = {}
    
    for url in urls:
        print(f"Fazendo scraping de: {url}")
        try:
            content = scraper.scrape_url(url)
            documents[url] = content
            print(f"✓ Concluído: {url}")
            
            if delay > 0:
                time.sleep(delay)
                
        except Exception as e:
            print(f"✗ Erro ao processar {url}: {e}")
    
    return documents


def read_multiple_local_files(filenames: List[str]) -> Dict[str, str]:
    documents = {}
    
    for filename in filenames:
        filepath = os.path.join(os.getcwd(), filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️ Arquivo não encontrado: {filepath}")
            continue
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            documents[filepath] = content
            print(f"✓ Arquivo lido: {filename}")
        except Exception as e:
            print(f"✗ Erro ao ler {filename}: {e}")
    
    return documents


def scrape_and_save_urls(urls: List[str], output_dir: str = None, delay: float = 1.0) -> None:
    if output_dir is None:
        output_dir = os.getcwd()
    
    os.makedirs(output_dir, exist_ok=True)
    
    scraper = WebScraper(delay=delay)
    
    for url in urls:
        print(f"Fazendo scraping de: {url}")
        try:
            content = scraper.scrape_url(url)
            
            filename = url.split('/')[-1] or url.split('/')[-2]
            filename = filename.replace('.html', '.txt')
            
            if not filename.endswith('.txt'):
                filename += '.txt'
            
            filepath = os.path.join(output_dir, filename)
            scraper.save_to_file(content, filepath)
            
            if delay > 0:
                time.sleep(delay)
                
        except Exception as e:
            print(f"✗ Erro ao processar {url}: {e}")


if __name__ == "__main__":
    from config import REFERENCE_URLS
    
    if not REFERENCE_URLS:
        print("Nenhuma URL configurada em config.py")
        exit(1)
    
    print(f"Fazendo scraping de {len(REFERENCE_URLS)} URL(s)...")
    print("-" * 60)
    
    scrape_and_save_urls(REFERENCE_URLS, delay=1.0)
    
    print("-" * 60)
    print("Scraping concluído!")
