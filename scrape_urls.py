import os
from urllib.parse import urlparse
from config import REFERENCE_URLS
from scraper import scrape_multiple_urls


def generate_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    path_parts = [part for part in parsed.path.split('/') if part]
    
    if not path_parts:
        filename = parsed.netloc.replace('.', '_')
    else:
        filename = path_parts[-1]
    
    if not filename.endswith('.txt'):
        filename += '.txt'
    
    return filename


def save_documents_to_files(documents: dict) -> None:
    for url, content in documents.items():
        filename = generate_filename_from_url(url)
        filepath = os.path.join(os.getcwd(), filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✓ Arquivo salvo: {filepath}")


def main():
    if not REFERENCE_URLS:
        print("Nenhuma URL configurada em config.py")
        return
    
    print(f"Fazendo scraping de {len(REFERENCE_URLS)} URL(s)...")
    print("-" * 60)
    
    url_documents = scrape_multiple_urls(REFERENCE_URLS, delay=1.0)
    
    if url_documents:
        print("-" * 60)
        print("Salvando documentos em arquivos locais...")
        save_documents_to_files(url_documents)
        print("-" * 60)
        print(f"✓ Scraping concluído! {len(url_documents)} arquivo(s) salvo(s).")
    else:
        print("⚠️  Nenhum documento foi obtido do scraping.")


if __name__ == "__main__":
    main()
