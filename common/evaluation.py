from typing import List, Dict, Union, Tuple
from openai import OpenAI
from common.config import OPENAI_API_KEY

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

def llm_as_judge_evaluate_sync(
    question: str,
    retrieved_docs: List[Dict],
    model: str = "gpt-4o-mini"
) -> List[int]:
    if not retrieved_docs:
        return []
    
    if not client:
        return []
    
    relevances = []
    
    for doc in retrieved_docs:
        document_text = doc.get("document", "")[:1000]
        
        prompt = f"""Você é um avaliador especializado em avaliar a relevância semântica de documentos para uma pergunta específica.

Pergunta do usuário: {question}

Documento a avaliar:
{document_text}

Avalie a relevância semântica deste documento para responder à pergunta acima. 
Considere:
- Quão bem o documento responde ou se relaciona com a pergunta
- A qualidade e completude da informação
- A precisão e utilidade do conteúdo

Retorne APENAS um número inteiro de 1 a 5, onde:
- 1 = Relevância muito baixa
- 2 = Relevância baixa
- 3 = Relevância moderada
- 4 = Relevância alta
- 5 = Relevância muito alta/perfeita

Número (1-5):"""
        
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Você é um avaliador objetivo de relevância semântica. Retorne apenas números inteiros de 1 a 5."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=10
        )
        
        answer = response.choices[0].message.content.strip()
        relevance = int(float(answer.split()[0]))
        relevance = max(1, min(5, relevance))
        relevances.append(relevance)
    
    return relevances

def calculate_llm_judge_score(
    question: str,
    retrieved_docs: List[Dict],
    model: str = "gpt-4o-mini"
) -> int:
    if not retrieved_docs:
        return 1
    
    relevances = llm_as_judge_evaluate_sync(
        question=question,
        retrieved_docs=retrieved_docs,
        model=model
    )
    
    if not relevances:
        return 1
    
    avg_relevance = sum(relevances) / len(relevances)
    return int(round(avg_relevance))

def llm_as_judge_evaluate_response(
    question: str,
    context: str,
    answer: str,
    model: str = "gpt-4o-mini",
    return_feedback: bool = True
) -> Union[int, Tuple[int, str]]:
    if not answer or not question:
        return 1, "Resposta ou pergunta vazia."
        
    
    if not client:
        return 1, "API key não configurada."
    
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

Avalie a qualidade da resposta considerando:
- **Relevância**: A resposta responde adequadamente à pergunta?
- **Precisão**: A resposta está correta e baseada no contexto fornecido?
- **Completude**: A resposta é completa e abrangente?
- **Clareza**: A resposta é clara e bem estruturada?
- **Uso do contexto**: A resposta utiliza adequadamente as informações do contexto?

Retorne sua avaliação no seguinte formato:
NOTA: [número inteiro de 1 a 5]
FEEDBACK: [explicação detalhada do porquê dessa nota, mencionando pontos fortes e fracos]

Onde a nota significa:
- 1 = Resposta muito ruim, pouco relevante ou muito incompleta
- 2 = Resposta ruim, parcialmente relevante mas com problemas significativos
- 3 = Resposta adequada, responde à pergunta mas com algumas limitações
- 4 = Resposta boa, relevante, precisa e bem estruturada
- 5 = Resposta excelente, perfeitamente relevante, precisa, completa e clara"""
    
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
    
    lines = result.split("\n")
    score = None
    feedback = ""
    
    for line in lines:
        if line.upper().startswith("NOTA:"):
            score_str = line.split(":", 1)[1].strip()
            score = int(float(score_str.split()[0]))
            score = max(1, min(5, score))
        elif line.upper().startswith("FEEDBACK:"):
            feedback = line.split(":", 1)[1].strip()
    
    if score is None:
        score = int(float(result.split()[0]))
        score = max(1, min(5, score))
        feedback = "Feedback não disponível."
    
    return score, feedback
    
