# graph.py
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.tools import BaseTool
import operator
import pandas as pd
from langchain_core.messages import ToolMessage

# Nó que executa as ferramentas (COM A LÓGICA CORRIGIDA)
class ToolNode:
    def __init__(self, tools: list[BaseTool]):
        self.tool_map = {tool.name: tool for tool in tools}

    def execute(self, state):
        print("---EXECUTANDO FERRAMENTA(S)---")
        last_message = state["messages"][-1]

        # Se não houver chamadas de ferramenta, não faz nada.
        if not last_message.tool_calls:
            return {}

        # CRIA UMA LISTA PARA GUARDAR AS RESPOSTAS DE TODAS AS FERRAMENTAS
        tool_messages = []
        
        # FAZ UM LOOP POR TODAS AS CHAMADAS DE FERRAMENTA QUE O AGENTE FEZ
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            
            if tool_name not in self.tool_map:
                # Se a ferramenta não for encontrada, retorna um erro educado
                error_message = f"Erro: A ferramenta '{tool_name}' não foi encontrada."
                tool_messages.append(ToolMessage(content=error_message, tool_call_id=tool_call['id']))
                continue

            # Executa a ferramenta e guarda o resultado
            tool_result = self.tool_map[tool_name].invoke(tool_call["args"])
            print(f"Resultado da Ferramenta '{tool_name}': {tool_result}")
            
            # Adiciona a mensagem de resultado à nossa lista de respostas
            tool_messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call['id']))
        
        # Retorna a lista completa de respostas
        return {"messages": tool_messages}

# Definição do Estado do Grafo
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    tools: list[BaseTool]

# Função Roteadora: decide o próximo passo
def router(state):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_executor"
    return END

# Função que cria e compila o grafo
def create_graph():
    workflow = StateGraph(AgentState)
    
    from agents import analyst_agent
    
    workflow.add_node("analyst", analyst_agent)
    workflow.add_node("tool_executor", lambda state: ToolNode(state['tools']).execute(state))

    workflow.set_entry_point("analyst")

    workflow.add_conditional_edges("analyst", router)
    workflow.add_edge("tool_executor", "analyst")

    graph = workflow.compile()
    print("Grafo compilado com sucesso!")
    return graph