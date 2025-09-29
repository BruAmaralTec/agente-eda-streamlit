# agents.py
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI # VERIFICADO: Importação correta para OpenAI
from langchain_core.messages import ToolMessage

# Carrega as variáveis de ambiente
load_dotenv()

# Inicializa o modelo LLM que será usado por todos os agentes
# VERIFICADO: Usando ChatOpenAI com o modelo gpt-3.5-turbo
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

def analyst_agent(state):
    """
    O agente Analista Mestre. Recebe o estado atual e decide a próxima ação.
    """
    print("---EXECUTANDO O AGENTE ANALISTA (com OpenAI)---")
    
    # Vincula as ferramentas ao LLM para que ele possa decidir qual chamar.
    llm_with_tools = llm.bind_tools(state['tools'])
    
    # Invoca o LLM com o histórico da conversa.
    ai_response = llm_with_tools.invoke(state['messages'])
    
    print(f"Decisão do Agente Analista: {ai_response}")

    # Adiciona a resposta do AI (que pode ser uma mensagem ou uma chamada de ferramenta) ao estado.
    return {"messages": [ai_response]}