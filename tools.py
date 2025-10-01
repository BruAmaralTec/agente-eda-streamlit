# tools.py
from langchain_core.tools import tool
from langchain_experimental.tools import PythonAstREPLTool
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import zscore
import streamlit as st

@tool
def create_python_analyst_tool(df: pd.DataFrame) -> PythonAstREPLTool:
    """
    Cria a ferramenta PythonAstREPLTool com o dataframe no escopo local.
    Esta é a ferramenta principal para consultas e análises de dados rápidas.
    """
    local_vars = {"df": df}
    python_analyst_tool = PythonAstREPLTool(
        name="python_analyst_repl",
        description="""Ferramenta segura para executar comandos Python de linha única para analisar e consultar o dataframe `df` já carregado. Ideal para cálculos, estatísticas e exploração de dados (ex: `df.describe()`, `df.head()`, `df['coluna'].value_counts()`). Não pode ser usada para criar gráficos ou modificar arquivos.""",
        locals=local_vars
    )
    return python_analyst_tool

@tool
def plot_histogram(column_name: str, max_value: int = None) -> str:
    """
    Gera e salva um histograma para uma coluna específica do dataframe 'df'.
    Use esta ferramenta para visualizar a distribuição de uma variável numérica.
    Opcionalmente, um valor máximo pode ser definido para focar em uma faixa de dados.
    A ferramenta salva o gráfico como 'plot.png'.
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Erro: O dataframe não foi encontrado na sessão."
    
    df = st.session_state.df
    
    if column_name not in df.columns:
        return f"Erro: A coluna '{column_name}' não existe no dataframe."
        
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return f"Erro: A coluna '{column_name}' não é numérica e não pode ser plotada em um histograma."

    try:
        plt.figure(figsize=(10, 6))
        data_to_plot = df[column_name].dropna()
        
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

@tool
def find_clusters(n_clusters: int = 5) -> str:
    """
    Realiza uma análise de clustering K-Means nos dados numéricos do dataframe 'df'
    para encontrar agrupamentos naturais. Os dados são padronizados antes do clustering.
    Retorna um resumo da contagem de pontos de dados em cada cluster encontrado.
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Erro: O dataframe não foi encontrado na sessão."
    
    df = st.session_state.df
    
    try:
        # Seleciona apenas colunas numéricas para o clustering
        df_numeric = df.select_dtypes(include=['number'])
        
        if df_numeric.empty:
            return "Erro: Não foram encontradas colunas numéricas no dataframe para realizar o clustering."
        
        # Padroniza os dados (importante para K-Means)
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_numeric)
        
        # Executa o algoritmo K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        kmeans.fit(df_scaled)
        
        # Conta quantos pontos de dados caíram em cada cluster
        cluster_counts = pd.Series(kmeans.labels_).value_counts().sort_index()
        
        summary = f"Análise de cluster K-Means concluída com sucesso para {n_clusters} clusters.\n"
        summary += "A distribuição de pontos de dados por cluster é a seguinte:\n"
        for cluster, count in cluster_counts.items():
            summary += f"- Cluster {cluster}: {count} pontos de dados\n"
            
        return summary
    except Exception as e:
        return f"Ocorreu um erro ao realizar a análise de cluster: {e}"

@tool
def detect_outliers_zscore(column_name: str, threshold: float = 3.0) -> str:
    """
    Detecta outliers em uma coluna numérica do dataframe 'df' usando o método Z-score.
    Um outlier é um ponto de dados cujo Z-score em valor absoluto é maior que o threshold (padrão 3.0).
    Retorna um resumo da contagem e porcentagem de outliers encontrados.
    """
    if 'df' not in st.session_state or st.session_state.df is None:
        return "Erro: O dataframe não foi encontrado na sessão."
    
    df = st.session_state.df
    
    if column_name not in df.columns:
        return f"Erro: A coluna '{column_name}' não existe no dataframe."
        
    if not pd.api.types.is_numeric_dtype(df[column_name]):
        return f"Erro: A coluna '{column_name}' não é numérica e não pode ser analisada com Z-score."
        
    try:
        # Pega a coluna e remove valores nulos que podem atrapalhar o cálculo
        col_data = df[column_name].dropna()
        
        if col_data.empty:
            return f"A coluna '{column_name}' não possui dados válidos para análise."
            
        # Calcula os Z-scores para a coluna
        z_scores = zscore(col_data)
        
        # Identifica os outliers (onde o valor absoluto do Z-score > threshold)
        outliers = np.abs(z_scores) > threshold
        
        # Conta quantos outliers foram encontrados
        num_outliers = np.sum(outliers)
        total_points = len(col_data)
        percentage_outliers = (num_outliers / total_points) * 100 if total_points > 0 else 0
        
        summary = (
            f"Análise de outliers com Z-score concluída para a coluna '{column_name}' (usando um threshold de {threshold}):\n"
            f"- Foram encontrados {num_outliers} outliers.\n"
            f"- Isso representa {percentage_outliers:.2f}% do total de pontos de dados válidos na coluna."
        )
        
        return summary
    except Exception as e:
        return f"Ocorreu um erro ao detectar outliers: {e}"