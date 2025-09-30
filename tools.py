# tools.py
from langchain_core.tools import tool
from langchain_experimental.tools import PythonAstREPLTool
import pandas as pd
import matplotlib.pyplot as plt

# Ferramenta principal para análise de dados via REPL
def create_python_analyst_tool(df: pd.DataFrame) -> PythonAstREPLTool:
    """Cria a ferramenta PythonAstREPLTool com o dataframe no escopo local."""
    local_vars = {"df": df}
    python_analyst_tool = PythonAstREPLTool(
        name="python_analyst_repl",
        description="""Ferramenta segura para executar comandos Python de linha única para analisar e consultar o dataframe `df` já carregado. Ideal para cálculos, estatísticas e exploração de dados (ex: `df.describe()`, `df.head()`, `df['coluna'].value_counts()`). Não pode ser usada para criar gráficos ou modificar arquivos.""",
        locals=local_vars
    )
    return python_analyst_tool

# --- INÍCIO DA NOVA FERRAMENTA DE GRÁFICO ---
@tool
def plot_histogram(column_name: str, max_value: int = None) -> str:
    """
    Gera e salva um histograma para uma coluna específica do dataframe 'df'.
    Use esta ferramenta para visualizar a distribuição de uma variável numérica.
    Opcionalmente, um valor máximo pode ser definido para focar em uma faixa de dados.
    A ferramenta salva o gráfico como 'plot.png'.
    """
    # Esta ferramenta acessa o dataframe que está no escopo da sessão do Streamlit.
    # Em uma aplicação real, passaríamos o 'df' como argumento.
    # Por simplicidade aqui, vamos assumir que ele está acessível ou recarregá-lo.
    # NOTA: Para que esta ferramenta funcione no deploy do Streamlit,
    # o dataframe precisa estar no `st.session_state`.
    import streamlit as st
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Erro: O dataframe não foi encontrado na sessão."

    df = st.session_state.df
    
    if column_name not in df.columns:
        return f"Erro: A coluna '{column_name}' não existe no dataframe."

    try:
        plt.figure(figsize=(10, 6))
        data_to_plot = df[column_name]
        
        if max_value is not None:
            data_to_plot = data_to_plot[data_to_plot <= max_value]

        data_to_plot.hist(bins=50)
        plt.title(f"Histograma da Coluna '{column_name}'")
        plt.xlabel(column_name)
        plt.ylabel("Frequência")
        plt.grid(axis='y', alpha=0.75)

        # Salva a figura em um arquivo que o Streamlit pode exibir
        plt.savefig("plot.png")
        plt.close() # Fecha a figura para liberar memória

        return "Histograma gerado com sucesso e salvo como plot.png."
    except Exception as e:
        return f"Ocorreu um erro ao gerar o gráfico: {e}"
# --- FIM DA NOVA FERRAMENTA DE GRÁFICO ---