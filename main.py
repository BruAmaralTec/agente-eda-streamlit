# main.py
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
from graph import create_graph, AgentState
from tools import create_python_analyst_tool, plot_histogram # IMPORTA A NOVA FERRAMENTA
from langchain_core.messages import HumanMessage, SystemMessage
import os

components.html(
    """
    <script>
    if (!sessionStorage.getItem('reloaded')) {
        sessionStorage.setItem('reloaded', 'true');
        window.location.reload();
    }
    </script>
    """,
    height=0
)

st.set_page_config(page_title="Agente de E.D.A Bruna Amaral.", layout="wide")

st.title("🤖 Agente Autonomo para Análise Exploratória de Dados (E.D.A")

st.sidebar.image("assets/logo.png", use_container_width=True)
st.sidebar.header("Controles")
uploaded_file = st.sidebar.file_uploader("Escolha um arquivo CSV", type="csv")

if "graph" not in st.session_state:
    st.session_state.df = None
    st.session_state.graph = None
    st.session_state.tools = []
    st.session_state.messages = [
        SystemMessage(content="Você é um analista de dados expert. Use suas ferramentas para responder às perguntas do usuário sobre o dataframe `df`. Responda de forma concisa e direta, sempre em português.")
    ]

if uploaded_file is not None and st.session_state.df is None:
    try:
        components.html("<script>sessionStorage.removeItem('reloaded');</script>", height=0)
        
        df = pd.read_csv(uploaded_file, engine='python', on_bad_lines='skip', sep=None)
        
        st.session_state.df = df
        
        python_tool = create_python_analyst_tool(df)
        
        # --- INÍCIO DA MUDANÇA ---
        # Adicionamos a nova ferramenta de plotagem à lista de ferramentas
        st.session_state.tools = [python_tool, plot_histogram]
        # --- FIM DA MUDANÇA ---
        
        st.session_state.graph = create_graph()
        
        st.sidebar.success("Arquivo carregado e agente pronto!")
    except Exception as e:
        st.sidebar.error(f"Erro ao carregar o arquivo: {e}")

if st.session_state.df is not None:
    st.subheader("Pré-visualização dos Dados")
    st.dataframe(st.session_state.df.head())
else:
    st.info("Por favor, carregue um arquivo CSV na barra lateral para começar a análise.")

st.subheader("Faça sua pergunta ao Agente")

for msg in st.session_state.messages:
    if not isinstance(msg, SystemMessage):
        with st.chat_message(msg.type):
            if hasattr(msg, 'content') and msg.content:
                st.markdown(msg.content)
                # --- INÍCIO DA MUDANÇA ---
                # Se a última mensagem foi a de sucesso do gráfico, exibe a imagem
                if "Histograma gerado com sucesso" in msg.content:
                    if os.path.exists("plot.png"):
                        st.image("plot.png")
                # --- FIM DA MUDANÇA ---

if prompt := st.chat_input("Descreva os dados para mim."):
    if st.session_state.graph is None:
        st.warning("Por favor, carregue um arquivo CSV primeiro.")
    else:
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("O agente está pensando..."):
                initial_state: AgentState = {
                    "messages": list(st.session_state.messages),
                    "tools": st.session_state.tools,
                }
                
                final_state = st.session_state.graph.invoke(initial_state)
                response_message = final_state['messages'][-1]
                st.markdown(response_message.content)
                
                # --- INÍCIO DA MUDANÇA ---
                # Lógica para exibir a imagem também na resposta em tempo real
                if "Histograma gerado com sucesso" in response_message.content:
                    if os.path.exists("plot.png"):
                        st.image("plot.png")
                # --- FIM DA MUDANÇA ---
                
                st.session_state.messages = final_state['messages']