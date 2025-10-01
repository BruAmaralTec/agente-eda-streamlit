# main.py
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from graph import create_graph, AgentState
from tools import (
    create_python_analyst_tool,
    plot_histogram,
    find_clusters,
    detect_outliers_zscore
)
from langchain_core.messages import HumanMessage, SystemMessage
import os

# Este bloco de código injeta um script que força o reload da página na primeira carga da sessão.
components.html(
    """
    <script>
    if (!sessionStorage.getItem('reloaded')) {
        sessionStorage.setItem('reloaded', 'true');
        window.location.reload();
    }
    </script>
    """,
    height=0  # O componente não precisa de espaço visual
)

# Configuração da página
st.set_page_config(page_title="Agente de E.D.A.", layout="wide")

# Título personalizado. Lembre-se de substituir "Bru Amaral".
st.title("🤖 Agente de E.D.A. do Bru Amaral")

# --- Configuração da Barra Lateral (Sidebar) ---
st.sidebar.image("assets/logo.png", use_container_width=True)
st.sidebar.header("Controles")
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV", type="csv")

# --- Inicialização do Estado da Sessão ---
if "graph" not in st.session_state:
    st.session_state.df = None
    st.session_state.graph = None
    st.session_state.tools = []
    st.session_state.messages = [
        SystemMessage(content="Você é um analista de dados expert. Use suas ferramentas para responder às perguntas do usuário sobre o dataframe `df`. Responda de forma concisa e direta, sempre em português.")
    ]

# --- Lógica de Carregamento e Preparação do Agente ---
if uploaded_file is not None and st.session_state.df is None:
    try:
        components.html("<script>sessionStorage.removeItem('reloaded');</script>", height=0)
        
        # Leitura robusta do arquivo CSV
        df = pd.read_csv(uploaded_file, engine='python', on_bad_lines='skip', sep=None)
        
        st.session_state.df = df
        
        # --- INÍCIO DA CORREÇÃO ---
        # 1. Usamos a "receita" para criar a ferramenta específica para este dataframe.
        python_tool = create_python_analyst_tool(df)
        
        # 2. Adicionamos a ferramenta pronta (o "bolo") à lista, junto com as outras.
        st.session_state.tools = [
            python_tool, # AQUI ESTÁ A FERRAMENTA PRONTA
            plot_histogram,
            find_clusters,
            detect_outliers_zscore
        ]
        # --- FIM DA CORREÇÃO ---
        
        # Cria o grafo do agente com o conjunto de ferramentas
        st.session_state.graph = create_graph()
        
        st.sidebar.success("Arquivo carregado e agente pronto!")
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar o arquivo: {e}")

# --- Exibição da Pré-visualização dos Dados ---
if st.session_state.df is not None:
    st.subheader("Pré-visualização dos Dados")
    st.dataframe(st.session_state.df.head())
else:
    st.info("Por favor, carregue um arquivo CSV na barra lateral para começar a análise.")

# --- Interface de Chat ---
st.subheader("Faça sua pergunta ao Agente")

# Exibe o histórico de mensagens
for msg in st.session_state.messages:
    if not isinstance(msg, SystemMessage):
        with st.chat_message(msg.type):
            if hasattr(msg, 'content') and msg.content:
                st.markdown(msg.content)
                # Se a mensagem contiver o gatilho de sucesso do gráfico, exibe a imagem
                if "Histograma gerado com sucesso" in msg.content:
                    if os.path.exists("plot.png"):
                        st.image("plot.png")

# Captura a entrada do usuário
if prompt := st.chat_input("Descreva os dados para mim."):
    if st.session_state.graph is None:
        st.warning("Por favor, carregue um arquivo CSV primeiro.")
    else:
        # Adiciona a mensagem do usuário ao histórico e à tela
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        # Invoca o agente e exibe a resposta
        with st.chat_message("assistant"):
            with st.spinner("O agente está pensando..."):
                # Prepara o estado inicial para a invocação do grafo
                initial_state: AgentState = {
                    "messages": list(st.session_state.messages),
                    "tools": st.session_state.tools,
                }
                
                # Executa o agente
                final_state = st.session_state.graph.invoke(initial_state)
                response_message = final_state['messages'][-1]
                
                # Exibe a resposta em texto
                st.markdown(response_message.content)
                
                # Lógica para exibir a imagem também na resposta em tempo real
                if "Histograma gerado com sucesso" in response_message.content:
                    if os.path.exists("plot.png"):
                        st.image("plot.png")
                
                # Atualiza o histórico de mensagens da sessão com o fluxo completo da conversa
                st.session_state.messages = final_state['messages']