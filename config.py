# Configurações para avaliação - podem ser listas para testar múltiplas combinações
OLLAMA_MODEL = ["llama3.2", "llama3:70b","mistral", "deepseek-r1", "phi3", "neural-chat", "solar", "moondream", "deepseek-r1:671b", "llama3.1:405b"]  # Lista de modelos para testar
CHUNK_SIZE = [1000, 1200]  # Lista de tamanhos de chunk para testar
CHUNK_OVERLAP = [100, 200]  # Lista de overlaps para testar
NUM_RETRIEVED_DOCS = [3, 4]  # Lista de números de documentos para testar

# Para uso no app.py, usa o primeiro valor de cada lista
OLLAMA_MODEL_SINGLE = OLLAMA_MODEL[0] if isinstance(OLLAMA_MODEL, list) else OLLAMA_MODEL
CHUNK_SIZE_SINGLE = CHUNK_SIZE[0] if isinstance(CHUNK_SIZE, list) else CHUNK_SIZE
CHUNK_OVERLAP_SINGLE = CHUNK_OVERLAP[0] if isinstance(CHUNK_OVERLAP, list) else CHUNK_OVERLAP
NUM_RETRIEVED_DOCS_SINGLE = NUM_RETRIEVED_DOCS[0] if isinstance(NUM_RETRIEVED_DOCS, list) else NUM_RETRIEVED_DOCS

# OpenAI API Key para LLM-as-a-Judge (pode ser None se não quiser usar)
# Configure no arquivo .env ou via variável de ambiente
import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente do arquivo .env
load_dotenv()

# Configura tokenizers para evitar warnings de paralelismo
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)

PROMPT_TEMPLATE = """
You are a programming assistant specialized in explaining concepts in a clear, didactic, and structured way.

You will receive:
- A context retrieved from a RAG (Retrieval-Augmented Generation) system
- A user's question related to this context

Your responsibilities:
1. Identify the domain/topic of the provided context.
2. Verify whether the user's question is relevant to this domain.
   - If the question is NOT relevant or cannot be answered using the given context, clearly state that the information is not available in the provided context.
3. If the question IS relevant, answer it **using only the information contained in the context**.
   - Do NOT use external knowledge.
   - Do NOT make assumptions beyond the context.

Guidelines for the answer:
- Provide a simple and clear response.
- without any additional text or comments.
- with simple and clear examples.
- withou any content outside the context or outside the question.
- withou any additional information.
- Keep the answer focused and technically accurate.
- Do NOT mention the RAG system, retrieval process, or internal reasoning.
- the answer shoud be short and to the point.
- analyze the question and the context to provide the most relevant answer.
- you should mention any information relevant to the question and the context, optionally or explicitly.

Context:
{context}

User Question:
{question}

Answer:

"""

REFERENCE_URLS = [
  'https://docs.pybricks.com/en/stable/hubs/primehub.html',
  'https://docs.pybricks.com/en/stable/pupdevices/dcmotor.html',
  # 'https://docs.pybricks.com/en/stable/pupdevices/motor.html',
]

# Arquivos locais para indexar
LOCAL_FILES = [
    'motor.txt',
    'primehub.txt',
]
  # 'https://docs.pybricks.com/en/stable/pupdevices/tiltsensor.html',
  # 'https://docs.pybricks.com/en/stable/pupdevices/infraredsensor.html',
  # 'https://docs.pybricks.com/en/stable/pupdevices/colordistancesensor.html',
  # 'https://docs.pybricks.com/en/stable/pupdevices/pfmotor.html',
  # 'https://docs.pybricks.com/en/stable/pupdevices/colorsensor.html',
  # 'https://docs.pybricks.com/en/stable/pupdevices/ultrasonicsensor.html',
  # 'https://docs.pybricks.com/en/stable/pupdevices/forcesensor.html',
  # 'https://docs.pybricks.com/en/stable/pupdevices/colorlightmatrix.html',
  # 'https://docs.pybricks.com/en/stable/pupdevices/light.html',
  # 'https://docs.pybricks.com/en/stable/pupdevices/remote.html',
  # 'https://docs.pybricks.com/en/stable/parameters/index.html',
  # 'https://docs.pybricks.com/en/stable/parameters/axis.html',
  # 'https://docs.pybricks.com/en/stable/parameters/button.html',
  # 'https://docs.pybricks.com/en/stable/parameters/color.html',
  # 'https://docs.pybricks.com/en/stable/parameters/direction.html',
  # 'https://docs.pybricks.com/en/stable/parameters/icon.html',
  # 'https://docs.pybricks.com/en/stable/parameters/port.html',
  # 'https://docs.pybricks.com/en/stable/parameters/side.html',
  # 'https://docs.pybricks.com/en/stable/parameters/stop.html',
  # 'https://docs.pybricks.com/en/stable/tools/index.html',
  # 'https://docs.pybricks.com/en/stable/robotics.html',
  # 'https://docs.pybricks.com/en/stable/signaltypes.html',
  # 'https://docs.pybricks.com/en/stable/micropython/builtins.html#',
  # 'https://docs.pybricks.com/en/stable/micropython/exceptions.html',
  # 'https://docs.pybricks.com/en/stable/micropython/micropython.html',
  # 'https://docs.pybricks.com/en/stable/micropython/uerrno.html',
  # 'https://docs.pybricks.com/en/stable/micropython/uio.html',
  # 'https://docs.pybricks.com/en/stable/micropython/ujson.html',
  # 'https://docs.pybricks.com/en/stable/micropython/umath.html',
  # 'https://docs.pybricks.com/en/stable/micropython/urandom.html',
  # 'https://docs.pybricks.com/en/stable/micropython/uselect.html',
  # 'https://docs.pybricks.com/en/stable/micropython/ustruct.html',
  # 'https://docs.pybricks.com/en/stable/micropython/usys.html'
# ]


PERGUNTAS_TESTE = [
    # Prime Hub e suas funcionalidades
    "Quais são os parâmetros aceitos na inicialização do Prime Hub?",
    "Como acender a luz de status do Prime Hub em uma cor específica?",
    "Quais funcionalidades estão disponíveis no Prime Hub e como utilizá-las?",
    "Como utilizar o Bluetooth do Prime Hub para enviar e receber dados?",
    "Quais sensores podem ser conectados ao Prime Hub?",
    "Como configurar o botão central do Prime Hub para executar uma função personalizada?",
    "Como ler os valores do IMU (Unidade de Medida Inercial) do Prime Hub?",
    "O que a função imu.heading() retorna e como utilizá-la corretamente?",
    "Como configurar a exibição de números no display do Prime Hub?",
    "Como exibir uma animação na matriz de LEDs do Prime Hub?",
    "Quais são as opções disponíveis para o controle da luz de status no Prime Hub?",

    # Motores e Sensores
    "Como controlar um motor DC sem sensor de rotação?",
    "Qual a diferença entre brake() e stop() ao parar um motor?",
    "Como usar um sensor de distância no Pybricks?",
    "Quais sensores podem medir a inclinação do Prime Hub?",
    "Como medir a força aplicada com um sensor de força no Pybricks?",

    # Comunicação e Integração
    "Como transmitir dados entre dois Prime Hubs usando Bluetooth?",
    "O que é ble.signal_strength() e como usá-lo?",
    "Como armazenar e recuperar dados na memória persistente do Prime Hub?",

    # Energia e Sistema
    "Como verificar a voltagem da bateria do Prime Hub?",
    "Como saber se o Prime Hub está sendo carregado?",
    "Quais são os motivos que podem levar o Prime Hub a reiniciar?",

    # Controle e Movimentação de Robôs
    "Como programar um robô para andar em linha reta usando o Pybricks?",
    "Como fazer um robô girar em torno do próprio eixo no Pybricks?"
]
