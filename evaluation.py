import math
from typing import List, Dict, Optional, Union, Tuple
from openai import OpenAI
from config import OPENAI_API_KEY, PROMPT_TEMPLATE

# Configura o cliente OpenAI
client = None
if OPENAI_API_KEY:
    client = OpenAI(api_key=OPENAI_API_KEY)


def calculate_ndcg(relevances: List[float], k: Optional[int] = None) -> float:
    """
    Calcula o NDCG (Normalized Discounted Cumulative Gain) de 0 a 5.
    
    Args:
        relevances: Lista de relevâncias (graus de relevância) dos documentos
        k: Número de documentos a considerar (None = todos)
    
    Returns:
        NDCG normalizado de 0 a 5
    """
    if not relevances:
        return 0.0
    
    if k is not None:
        relevances = relevances[:k]
    
    # Calcula DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for i, rel in enumerate(relevances, 1):
        dcg += rel / math.log2(i + 1)  # log2(i+1) para evitar log(1) = 0
    
    # Calcula IDCG (Ideal DCG) - ordenando as relevâncias em ordem decrescente
    ideal_relevances = sorted(relevances, reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_relevances, 1):
        idcg += rel / math.log2(i + 1)
    
    # Normaliza para 0-5
    if idcg == 0:
        return 0.0
    
    ndcg = (dcg / idcg) * 5.0
    return round(ndcg, 2)


def calculate_ndcg_from_similarities(retrieved_docs: List[Dict]) -> float:
    """
    Calcula NDCG baseado nas similaridades dos documentos recuperados.
    
    Args:
        retrieved_docs: Lista de documentos recuperados com 'distance'
    
    Returns:
        NDCG de 0 a 5
    """
    if not retrieved_docs:
        return 0.0
    
    # Converte distâncias em relevâncias (similaridades)
    relevances = []
    for doc in retrieved_docs:
        if doc.get('distance') is not None:
            # Converte distância (0-1) em relevância (0-5)
            # Quanto menor a distância, maior a similaridade
            similarity = 1 - doc['distance']
            # Normaliza para escala 0-5
            relevance = similarity * 5.0
            relevances.append(relevance)
        else:
            relevances.append(0.0)
    
    return calculate_ndcg(relevances)


def calculate_llm_judge_score(
    question: str,
    retrieved_docs: List[Dict],
    model: str = "gpt-4o-mini"
) -> float:
    """
    Utiliza LLM-as-a-Judge para avaliar semanticamente a qualidade dos documentos recuperados.
    Avalia cada documento usando GPT e retorna uma métrica agregada.
    
    Args:
        question: Pergunta do usuário
        retrieved_docs: Lista de documentos recuperados
        model: Modelo do OpenAI a ser usado
    
    Returns:
        Nota agregada de 0 a 5 baseada na avaliação semântica dos documentos
    """
    if not retrieved_docs:
        return 0.0
    
    # Usa LLM-as-a-Judge para avaliar cada documento
    relevances = llm_as_judge_evaluate_sync(
        question=question,
        retrieved_docs=retrieved_docs,
        model=model
    )
    
    if not relevances:
        return 0.0
    
    # Calcula a média das relevâncias (já estão na escala 0-5)
    avg_relevance = sum(relevances) / len(relevances)
    
    # Garante que está no range 0-5
    avg_relevance = max(0.0, min(5.0, avg_relevance))
    
    return round(avg_relevance, 2)


async def llm_as_judge_evaluate(
    question: str,
    retrieved_docs: List[Dict],
    model: str = "gpt-4o-mini"
) -> List[float]:
    """
    Utiliza LLM-as-a-Judge para avaliar semanticamente a relevância dos documentos.
    
    Args:
        question: Pergunta do usuário
        retrieved_docs: Lista de documentos recuperados
        model: Modelo do OpenAI a ser usado
    
    Returns:
        Lista de relevâncias (0-5) para cada documento
    """
    if not retrieved_docs:
        return []
    
    if not client:
        # Se não houver API key, retorna relevâncias baseadas em similaridade
        return [calculate_ndcg_from_similarities([doc]) * 5.0 for doc in retrieved_docs]
    
    relevances = []
    
    # Avalia cada documento individualmente
    for i, doc in enumerate(retrieved_docs, 1):
        document_text = doc.get('document', '')[:1000]  # Limita tamanho para economia
        
        prompt = f"""Você é um avaliador especializado em avaliar a relevância semântica de documentos para uma pergunta específica.

Pergunta do usuário: {question}

Documento a avaliar:
{document_text}

Avalie a relevância semântica deste documento para responder à pergunta acima. 
Considere:
- Quão bem o documento responde ou se relaciona com a pergunta
- A qualidade e completude da informação
- A precisão e utilidade do conteúdo

Retorne APENAS um número de 0 a 5, onde:
- 0 = Nenhuma relevância
- 1 = Relevância muito baixa
- 2 = Relevância baixa
- 3 = Relevância moderada
- 4 = Relevância alta
- 5 = Relevância muito alta/perfeita

Número (0-5):"""
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Você é um avaliador objetivo de relevância semântica. Retorne apenas números."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            # Extrai o número da resposta
            answer = response.choices[0].message.content.strip()
            # Tenta extrair apenas o número
            try:
                relevance = float(answer.split()[0])
                # Garante que está no range 0-5
                relevance = max(0.0, min(5.0, relevance))
            except (ValueError, IndexError):
                # Se não conseguir extrair, usa similaridade como fallback
                if doc.get('distance') is not None:
                    similarity = 1 - doc['distance']
                    relevance = similarity * 5.0
                else:
                    relevance = 0.0
            
            relevances.append(relevance)
            
        except Exception as e:
            # Em caso de erro, usa similaridade como fallback
            print(f"Erro ao avaliar com LLM: {e}")
            if doc.get('distance') is not None:
                similarity = 1 - doc['distance']
                relevance = similarity * 5.0
            else:
                relevance = 0.0
            relevances.append(relevance)
    
    return relevances


def llm_as_judge_evaluate_sync(
    question: str,
    retrieved_docs: List[Dict],
    model: str = "gpt-4o-mini"
) -> List[float]:
    """
    Versão síncrona do LLM-as-a-Judge para uso no Streamlit.
    """
    if not retrieved_docs:
        return []
    
    if not client:
        # Se não houver API key, retorna relevâncias baseadas em similaridade
        relevances = []
        for doc in retrieved_docs:
            if doc.get('distance') is not None:
                similarity = 1 - doc['distance']
                relevance = similarity * 5.0
            else:
                relevance = 0.0
            relevances.append(relevance)
        return relevances
    
    relevances = []
    
    # Avalia cada documento individualmente
    for i, doc in enumerate(retrieved_docs, 1):
        document_text = doc.get('document', '')[:1000]  # Limita tamanho para economia
        
        prompt = f"""Você é um avaliador especializado em avaliar a relevância semântica de documentos para uma pergunta específica.

Pergunta do usuário: {question}

Documento a avaliar:
{document_text}

Avalie a relevância semântica deste documento para responder à pergunta acima. 
Considere:
- Quão bem o documento responde ou se relaciona com a pergunta
- A qualidade e completude da informação
- A precisão e utilidade do conteúdo

Retorne APENAS um número de 0 a 5, onde:
- 0 = Nenhuma relevância
- 1 = Relevância muito baixa
- 2 = Relevância baixa
- 3 = Relevância moderada
- 4 = Relevância alta
- 5 = Relevância muito alta/perfeita

Número (0-5):"""
        
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Você é um avaliador objetivo de relevância semântica. Retorne apenas números."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=10
            )
            
            # Extrai o número da resposta
            answer = response.choices[0].message.content.strip()
            # Tenta extrair apenas o número
            try:
                relevance = float(answer.split()[0])
                # Garante que está no range 0-5
                relevance = max(0.0, min(5.0, relevance))
            except (ValueError, IndexError):
                # Se não conseguir extrair, usa similaridade como fallback
                if doc.get('distance') is not None:
                    similarity = 1 - doc['distance']
                    relevance = similarity * 5.0
                else:
                    relevance = 0.0
            
            relevances.append(relevance)
            
        except Exception as e:
            # Em caso de erro, usa similaridade como fallback
            print(f"Erro ao avaliar com LLM: {e}")
            if doc.get('distance') is not None:
                similarity = 1 - doc['distance']
                relevance = similarity * 5.0
            else:
                relevance = 0.0
            relevances.append(relevance)
    
    return relevances


def calculate_ndcg_with_llm_judge(
    question: str,
    retrieved_docs: List[Dict],
    model: str = "gpt-4o-mini"
) -> float:
    """
    Calcula NDCG usando LLM-as-a-Judge para avaliar relevâncias semânticas.
    
    Args:
        question: Pergunta do usuário
        retrieved_docs: Lista de documentos recuperados
        model: Modelo do OpenAI a ser usado
    
    Returns:
        NDCG de 0 a 5 baseado em avaliação semântica do LLM
    """
    relevances = llm_as_judge_evaluate_sync(question, retrieved_docs, model)
    return calculate_ndcg(relevances)


def llm_as_judge_evaluate_response(
    question: str,
    context: str,
    answer: str,
    model: str = "gpt-4o-mini",
    return_feedback: bool = False
) -> Union[float, Tuple[float, str]]:
    """
    Utiliza LLM-as-a-Judge para avaliar a qualidade da resposta final.
    Avalia a resposta considerando o contexto fornecido e a pergunta original.
    
    Args:
        question: Pergunta original do usuário
        context: Contexto (documentos recuperados formatados)
        answer: Resposta gerada pelo sistema RAG
        model: Modelo do OpenAI a ser usado
        return_feedback: Se True, retorna (nota, feedback). Se False, retorna apenas nota.
    
    Returns:
        Se return_feedback=False: Nota de 0 a 5
        Se return_feedback=True: Tupla (nota: float, feedback: str)
    """
    if not answer or not question:
        return 0
    
    if not client:
        raise ValueError("Sem client")
    
    # Limita o tamanho do contexto para economia de tokens
    context_limited = context[:3000] if len(context) > 3000 else context
    answer_limited = answer[:2000] if len(answer) > 2000 else answer
    
    prompt = f"""Você é um avaliador especializado em avaliar a qualidade de respostas geradas por sistemas RAG (Retrieval-Augmented Generation).

Sua tarefa é avaliar a qualidade da resposta final considerando:
1. A pergunta original do usuário
2. O contexto fornecido (documentos recuperados)
3. A resposta gerada pelo sistema

PERGUNTA ORIGINAL DO USUÁRIO:
{question}

CONTEXTO FORNECIDO (Documentos Recuperados):
{context_limited}

RESPOSTA GERADA:
{answer_limited}

PROMPT PARA GERAR A RESPOSTA:
{PROMPT_TEMPLATE}

Avalie a qualidade da resposta considerando:
- **Relevância**: A resposta responde adequadamente à pergunta?
- **Precisão**: A resposta está correta e baseada no contexto fornecido?
- **Completude**: A resposta é completa e abrangente?
- **Clareza**: A resposta é clara e bem estruturada?
- **Uso do contexto**: A resposta utiliza adequadamente as informações do contexto?

Retorne sua avaliação no seguinte formato:
NOTA: [número de 1 a 5]
FEEDBACK: [explicação detalhada do porquê dessa nota, mencionando pontos fortes e fracos]

Onde a nota significa:
- 1 = Resposta muito ruim, inadequada ou incorreta
- 2 = Resposta ruim, parcialmente relevante, com problemas significativos
- 3 = Resposta adequada, responde à pergunta, mas com limitações
- 4 = Resposta boa, relevante, precisa e bem estruturada
- 5 = Resposta excelente, perfeitamente relevante, precisa, completa e clara
"""
    
    try:
        system_message = "Você é um avaliador objetivo e rigoroso de qualidade de respostas."
        system_message += " Retorne a nota e um feedback detalhado explicando sua avaliação."

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=500
        )
        
        result = response.choices[0].message.content.strip()
        
        lines = result.split('\n')
        score = None
        feedback = ""
        
        for line in lines:
            if line.upper().startswith('NOTA:'):
                score_str = line.split(':', 1)[1].strip()
                score = float(score_str.split()[0])
                score = max(0.0, min(5.0, score))
            elif line.upper().startswith('FEEDBACK:'):
                feedback = line.split(':', 1)[1].strip()
        
        # Se não encontrou no formato esperado, tenta extrair número
        if score is None:
            score = float(result.split()[0])
            score = max(0.0, min(5.0, score))
            feedback = "Feedback não disponível."
        
        return round(score, 2), feedback
        
    except Exception as e:
        print(f"Erro ao avaliar resposta com LLM: {e}")
        return 0, f"Erro ao avaliar: {str(e)}"
