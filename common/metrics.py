import os
import pandas as pd
from datetime import datetime
from typing import List

def salvar_metricas_inicializacao(tempo_inicializacao: float, metrics_file: str):
    df = pd.DataFrame({
        "data_hora": [datetime.now()],
        "tempo_inicializacao": [tempo_inicializacao]
    })
    
    if os.path.exists(metrics_file):
        with pd.ExcelWriter(metrics_file, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            try:
                df_existente = pd.read_excel(metrics_file, sheet_name="inicializacao")
                df_final = pd.concat([df_existente, df], ignore_index=True)
            except:
                df_final = df
            
            df_final.to_excel(writer, sheet_name="inicializacao", index=False)
    else:
        df.to_excel(metrics_file, sheet_name="inicializacao", index=False)

def salvar_metrica_pergunta(
    pergunta: str,
    tempo_resposta: float,
    resposta: str,
    fontes_usadas: List[str],
    model_name: str,
    k_documents: int,
    chunk_size: int,
    chunk_overlap: int,
    metrics_file: str
):
    novo_registro = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "pergunta": pergunta,
        "resposta": resposta,
        "tempo_resposta": tempo_resposta,
        "fontes_usadas": ", ".join(fontes_usadas),
        "modelo": model_name,
        "k_documents": k_documents,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }
    df_novo = pd.DataFrame([novo_registro])
    
    if os.path.exists(metrics_file):
        with pd.ExcelWriter(metrics_file, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            try:
                df_existente = pd.read_excel(metrics_file, sheet_name="perguntas")
                df_final = pd.concat([df_existente, df_novo], ignore_index=True)
            except:
                df_final = df_novo
            
            df_final.to_excel(writer, sheet_name="perguntas", index=False)
    else:
        df_novo.to_excel(metrics_file, sheet_name="perguntas", index=False)

def salvar_metrica_processamento(
    tempo_proc: float,
    tipo_operacao: str,
    num_documentos: int,
    modelo: str,
    chunk_size: int,
    chunk_overlap: int,
    metrics_file: str
):
    novo_registro = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tipo_operacao": tipo_operacao,
        "tempo_processamento": tempo_proc,
        "num_documentos": num_documentos,
        "modelo": modelo,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }
    df_novo = pd.DataFrame([novo_registro])
    
    if os.path.exists(metrics_file):
        with pd.ExcelWriter(metrics_file, mode="a", engine="openpyxl", if_sheet_exists="overlay") as writer:
            try:
                df_existente = pd.read_excel(metrics_file, sheet_name="processamento")
                df_final = pd.concat([df_existente, df_novo], ignore_index=True)
            except:
                df_final = df_novo
            
            df_final.to_excel(writer, sheet_name="processamento", index=False)
    else:
        df_novo.to_excel(metrics_file, sheet_name="processamento", index=False)
