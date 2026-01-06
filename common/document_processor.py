import os
import io
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import markdown
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Optional, Union

def process_pdf(file) -> Optional[str]:
    pdf_reader = PdfReader(io.BytesIO(file.read()))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def process_markdown(file) -> Optional[str]:
    content = file.read().decode("utf-8")
    html = markdown.markdown(content)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text()

def process_text(file) -> Optional[str]:
    return file.read().decode("utf-8")

def process_url(url: str) -> Optional[str]:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    return soup.get_text()

def process_file(file) -> Optional[str]:
    file_extension = os.path.splitext(file.name)[1].lower()
    
    if file_extension == ".pdf":
        return process_pdf(file)
    if file_extension == ".md":
        return process_markdown(file)
    if file_extension == ".txt":
        return process_text(file)
    
    return None

def carregar_texto_arquivo(nome_arquivo: str) -> Optional[str]:
    with open(nome_arquivo, "r", encoding="utf-8") as f:
        return f.read()

def normalizar_documento(item: Union[str, Dict]) -> Optional[Dict]:
    if isinstance(item, str):
        if item.startswith("http"):
            return {
                "name": item,
                "type": "url",
                "text": ""
            }
        return {
            "name": item,
            "type": "file",
            "text": ""
        }
    
    if isinstance(item, dict):
        return {
            "name": item.get("name", ""),
            "type": item.get("type", "unknown"),
            "text": ""
        }
    
    return None

def fetch_and_process_urls(urls: List[Union[str, Dict]]) -> tuple[List[Document], int]:
    documents = []
    num_docs = 0
    
    for item in urls:
        if isinstance(item, str):
            doc_info = normalizar_documento(item)
        else:
            doc_info = item
        
        if not doc_info or not doc_info.get("name"):
            continue
        
        nome = doc_info["name"]
        tipo = doc_info["type"]
        
        if tipo == "url" or nome.startswith("http"):
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
        else:
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
    
    return documents, num_docs

def split_documents(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
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
    
    return texts
