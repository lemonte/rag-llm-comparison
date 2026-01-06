import os
from dotenv import load_dotenv

load_dotenv()

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DOCUMENT_URLS = [
    "https://docs.pybricks.com/en/latest/hubs/primehub.html",
    "https://docs.pybricks.com/en/stable/pupdevices/dcmotor.html",
]

LOCAL_FILES = [
    "motor.txt",
    "primehub.txt",
]

PERGUNTAS_TESTE_1 = [
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
    "Como controlar um motor DC sem sensor de rotação?",
    "Qual a diferença entre brake() e stop() ao parar um motor?",
    "Como usar um sensor de distância no Pybricks?",
    "Quais sensores podem medir a inclinação do Prime Hub?",
    "Como medir a força aplicada com um sensor de força no Pybricks?",
    "Como transmitir dados entre dois Prime Hubs usando Bluetooth?",
    "O que é ble.signal_strength() e como usá-lo?",
    "Como armazenar e recuperar dados na memória persistente do Prime Hub?",
    "Como verificar a voltagem da bateria do Prime Hub?",
    "Como saber se o Prime Hub está sendo carregado?",
    "Quais são os motivos que podem levar o Prime Hub a reiniciar?",
    "Como programar um robô para andar em linha reta usando o Pybricks?",
    "Como fazer um robô girar em torno do próprio eixo no Pybricks?",
]

PERGUNTAS_TESTE_2 = [
    {"numero_questao": 1, "pergunta": "Quais são os parâmetros aceitos na inicialização do Prime Hub?"},
    {"numero_questao": 1, "pergunta": "Quais argumentos podem ser passados ao criar uma instância do Prime Hub?"},
    {"numero_questao": 1, "pergunta": "Que parâmetros são suportados ao instanciar o Prime Hub no Pybricks?"},
    {"numero_questao": 1, "pergunta": "Como funciona a inicialização do Prime Hub e quais opções podem ser configuradas?"},

    {"numero_questao": 2, "pergunta": "Como acender a luz de status do Prime Hub em uma cor específica?"},
    {"numero_questao": 2, "pergunta": "De que forma é possível alterar a cor da luz de status do Prime Hub?"},
    {"numero_questao": 2, "pergunta": "Como configurar a luz de status do Prime Hub para exibir uma cor desejada?"},
    {"numero_questao": 2, "pergunta": "Qual é o método para ligar a luz de status do Prime Hub com uma cor específica?"},

    {"numero_questao": 3, "pergunta": "Quais funcionalidades estão disponíveis no Prime Hub e como utilizá-las?"},
    {"numero_questao": 3, "pergunta": "Quais recursos o Prime Hub oferece e como acessá-los no código?"},
    {"numero_questao": 3, "pergunta": "O que o Prime Hub é capaz de fazer e como usar suas funções?"},
    {"numero_questao": 3, "pergunta": "Quais são as principais funcionalidades do Prime Hub e como aplicá-las na prática?"},

    {"numero_questao": 4, "pergunta": "Como utilizar o Bluetooth do Prime Hub para enviar e receber dados?"},
    {"numero_questao": 4, "pergunta": "Como estabelecer comunicação via Bluetooth usando o Prime Hub?"},
    {"numero_questao": 4, "pergunta": "De que maneira o Bluetooth do Prime Hub pode ser usado para troca de dados?"},
    {"numero_questao": 4, "pergunta": "Qual é o processo para enviar e receber informações via Bluetooth no Prime Hub?"},

    {"numero_questao": 5, "pergunta": "Quais sensores podem ser conectados ao Prime Hub?"},
    {"numero_questao": 5, "pergunta": "Que tipos de sensores são compatíveis com o Prime Hub?"},
    {"numero_questao": 5, "pergunta": "Quais sensores o Prime Hub suporta para conexão direta?"},
    {"numero_questao": 5, "pergunta": "Quais dispositivos de sensor podem ser utilizados com o Prime Hub?"},

    {"numero_questao": 6, "pergunta": "Como configurar o botão central do Prime Hub para executar uma função personalizada?"},
    {"numero_questao": 6, "pergunta": "Como associar uma ação personalizada ao botão central do Prime Hub?"},
    {"numero_questao": 6, "pergunta": "De que forma é possível programar o botão central para executar uma função específica?"},
    {"numero_questao": 6, "pergunta": "Como capturar o evento do botão central do Prime Hub e executar código personalizado?"},

    {"numero_questao": 7, "pergunta": "Como ler os valores do IMU (Unidade de Medida Inercial) do Prime Hub?"},
    {"numero_questao": 7, "pergunta": "De que forma é possível acessar os dados do IMU no Prime Hub?"},
    {"numero_questao": 7, "pergunta": "Como obter informações de aceleração e rotação do IMU do Prime Hub?"},
    {"numero_questao": 7, "pergunta": "Qual é o método para ler os sensores inerciais do Prime Hub?"},

    {"numero_questao": 8, "pergunta": "O que a função imu.heading() retorna e como utilizá-la corretamente?"},
    {"numero_questao": 8, "pergunta": "Qual é o valor retornado por imu.heading() e como interpretá-lo?"},
    {"numero_questao": 8, "pergunta": "Para que serve a função imu.heading() e como usá-la no controle do robô?"},
    {"numero_questao": 8, "pergunta": "Como funciona o imu.heading() e em quais situações ele deve ser utilizado?"},

    {"numero_questao": 9, "pergunta": "Como configurar a exibição de números no display do Prime Hub?"},
    {"numero_questao": 9, "pergunta": "Como mostrar valores numéricos no display do Prime Hub?"},
    {"numero_questao": 9, "pergunta": "Qual é a forma correta de exibir números na tela do Prime Hub?"},
    {"numero_questao": 9, "pergunta": "Como personalizar a exibição de números no display do Prime Hub?"},

    {"numero_questao": 10, "pergunta": "Como exibir uma animação na matriz de LEDs do Prime Hub?"},
    {"numero_questao": 10, "pergunta": "De que forma é possível criar e mostrar animações na matriz de LEDs do Prime Hub?"},
    {"numero_questao": 10, "pergunta": "Como programar uma sequência animada de LEDs no Prime Hub?"},
    {"numero_questao": 10, "pergunta": "Qual é o processo para exibir animações no display de LEDs do Prime Hub?"},

    {"numero_questao": 11, "pergunta": "Quais são as opções disponíveis para o controle da luz de status no Prime Hub?"},
    {"numero_questao": 11, "pergunta": "Quais modos e configurações existem para a luz de status do Prime Hub?"},
    {"numero_questao": 11, "pergunta": "Como a luz de status do Prime Hub pode ser controlada e personalizada?"},
    {"numero_questao": 11, "pergunta": "Quais recursos estão disponíveis para manipular a luz de status do Prime Hub?"},

    # Motores e Sensores
    {"numero_questao": 12, "pergunta": "Como controlar um motor DC sem sensor de rotação?"},
    {"numero_questao": 12, "pergunta": "De que forma é possível operar um motor DC sem encoder no Pybricks?"},
    {"numero_questao": 12, "pergunta": "Como acionar e controlar um motor DC sem feedback de rotação?"},
    {"numero_questao": 12, "pergunta": "Quais técnicas podem ser usadas para controlar um motor DC sem sensor?"},

    {"numero_questao": 13, "pergunta": "Qual a diferença entre brake() e stop() ao parar um motor?"},
    {"numero_questao": 13, "pergunta": "O que muda entre usar brake() e stop() em um motor?"},
    {"numero_questao": 13, "pergunta": "Quais são os efeitos de brake() e stop() no comportamento do motor?"},
    {"numero_questao": 13, "pergunta": "Em que situações devo usar brake() ou stop() para interromper um motor?"},

    {"numero_questao": 14, "pergunta": "Como usar um sensor de distância no Pybricks?"},
    {"numero_questao": 14, "pergunta": "De que forma é possível ler medições de um sensor de distância no Pybricks?"},
    {"numero_questao": 14, "pergunta": "Como configurar e utilizar um sensor de distância em projetos Pybricks?"},
    {"numero_questao": 14, "pergunta": "Qual é o procedimento para obter dados de um sensor de distância no Pybricks?"},

    {"numero_questao": 15, "pergunta": "Quais sensores podem medir a inclinação do Prime Hub?"},
    {"numero_questao": 15, "pergunta": "Que sensores permitem detectar inclinação no Prime Hub?"},
    {"numero_questao": 15, "pergunta": "Como o Prime Hub mede inclinação e quais sensores estão envolvidos?"},
    {"numero_questao": 15, "pergunta": "Quais componentes são usados para identificar a inclinação do Prime Hub?"},

    {"numero_questao": 16, "pergunta": "Como medir a força aplicada com um sensor de força no Pybricks?"},
    {"numero_questao": 16, "pergunta": "Como obter leituras de força usando um sensor de força no Pybricks?"},
    {"numero_questao": 16, "pergunta": "De que maneira o Pybricks permite medir força aplicada em um sensor?"},
    {"numero_questao": 16, "pergunta": "Qual é a forma correta de acessar os dados de um sensor de força no Pybricks?"},

    # Comunicação e Integração
    {"numero_questao": 17, "pergunta": "Como transmitir dados entre dois Prime Hubs usando Bluetooth?"},
    {"numero_questao": 17, "pergunta": "Como configurar a comunicação Bluetooth entre dois Prime Hubs?"},
    {"numero_questao": 17, "pergunta": "De que forma dois Prime Hubs podem trocar informações via Bluetooth?"},
    {"numero_questao": 17, "pergunta": "Qual é o método para enviar dados de um Prime Hub para outro usando Bluetooth?"},

    {"numero_questao": 18, "pergunta": "O que é ble.signal_strength() e como usá-lo?"},
    {"numero_questao": 18, "pergunta": "Para que serve a função ble.signal_strength() no Pybricks?"},
    {"numero_questao": 18, "pergunta": "Como interpretar e utilizar o valor retornado por ble.signal_strength()?"},
    {"numero_questao": 18, "pergunta": "Em quais casos o ble.signal_strength() pode ser útil em projetos Bluetooth?"},

    {"numero_questao": 19, "pergunta": "Como armazenar e recuperar dados na memória persistente do Prime Hub?"},
    {"numero_questao": 19, "pergunta": "Como salvar informações de forma persistente no Prime Hub?"},
    {"numero_questao": 19, "pergunta": "De que maneira é possível gravar e ler dados da memória do Prime Hub?"},
    {"numero_questao": 19, "pergunta": "Qual é o processo para armazenar dados que persistem após reinicialização do Prime Hub?"},

    # Energia e Sistema
    {"numero_questao": 20, "pergunta": "Como verificar a voltagem da bateria do Prime Hub?"},
    {"numero_questao": 20, "pergunta": "Como obter a tensão atual da bateria do Prime Hub?"},
    {"numero_questao": 20, "pergunta": "De que forma é possível monitorar a voltagem da bateria no Prime Hub?"},
    {"numero_questao": 20, "pergunta": "Qual comando permite verificar o nível de voltagem da bateria do Prime Hub?"},

    {"numero_questao": 21, "pergunta": "Como saber se o Prime Hub está sendo carregado?"},
    {"numero_questao": 21, "pergunta": "Como identificar se o Prime Hub está conectado ao carregador?"},
    {"numero_questao": 21, "pergunta": "Existe uma forma de verificar o estado de carregamento do Prime Hub?"},
    {"numero_questao": 21, "pergunta": "Como detectar via código se o Prime Hub está em processo de carga?"},

    {"numero_questao": 22, "pergunta": "Quais são os motivos que podem levar o Prime Hub a reiniciar?"},
    {"numero_questao": 22, "pergunta": "O que pode causar reinicializações inesperadas no Prime Hub?"},
    {"numero_questao": 22, "pergunta": "Quais fatores fazem o Prime Hub reiniciar automaticamente?"},
    {"numero_questao": 22, "pergunta": "Em quais situações o Prime Hub pode acabar reiniciando?"},

    # Controle e Movimentação de Robôs
    {"numero_questao": 23, "pergunta": "Como programar um robô para andar em linha reta usando o Pybricks?"},
    {"numero_questao": 23, "pergunta": "Como fazer um robô se deslocar em linha reta no Pybricks?"},
    {"numero_questao": 23, "pergunta": "Quais técnicas podem ser usadas para manter o robô andando reto no Pybricks?"},
    {"numero_questao": 23, "pergunta": "Como controlar os motores para garantir movimento em linha reta no Pybricks?"},

    {"numero_questao": 24, "pergunta": "Como fazer um robô girar em torno do próprio eixo no Pybricks?"},
    {"numero_questao": 24, "pergunta": "Como programar uma rotação no próprio eixo usando o Pybricks?"},
    {"numero_questao": 24, "pergunta": "De que forma é possível fazer o robô girar parado no Pybricks?"},
    {"numero_questao": 24, "pergunta": "Como controlar os motores para realizar um giro no próprio eixo no Pybricks?"}
]




PROMPT_TEMPLATE = """
Use o contexto abaixo para responder à pergunta. 
Se a informação estiver presente no contexto, inclua a citação usando [source_id].
Se não souber a resposta, diga que não sabe.
Responda no mesmo idioma da pergunta.

Contexto: {context}

Pergunta: {question}
"""

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_VECTOR_STORE_ID = os.getenv("OPENAI_VECTOR_STORE_ID")
OPENAI_FILE_SEARCH_MODEL = os.getenv("OPENAI_FILE_SEARCH_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

def get_chromadb_folder(model_name: str, chunk_size: int, k_documents: int, chunk_overlap: int) -> str:
    return f"chroma_db_storage_{model_name}_{chunk_size}_{k_documents}_{chunk_overlap}"

def get_metrics_file(model_name: str, chunk_size: int, k_documents: int, chunk_overlap: int) -> str:
    return f"metricas_rag_{model_name}_{chunk_size}_{k_documents}_{chunk_overlap}.xlsx"

def get_documents_file() -> str:
    return "documentos_processados.json"
