# tools.py
from langchain_core.tools import tool
from langchain_experimental.tools import PythonAstREPLTool
import pandas as pd

def create_python_analyst_tool(df: pd.DataFrame) -> PythonAstREPLTool:
    """Cria a ferramenta PythonAstREPLTool com o dataframe no escopo local."""
    local_vars = {"df": df}
    python_analyst_tool = PythonAstREPLTool(
        name="python_analyst_repl",
        description="""Ferramenta segura para executar comandos Python de linha única para analisar e consultar o dataframe `df` já carregado. Ideal para cálculos, estatísticas e exploração de dados (ex: `df.describe()`, `df.head()`, `df['coluna'].value_counts()`). Não pode ser usada para criar gráficos ou modificar arquivos.""",
        locals=local_vars
    )
    return python_analyst_tool

@tool
def code_sandbox_executor(code: str) -> str:
    """
    (A SER IMPLEMENTADO) Ambiente de teste para executar scripts Python completos de múltiplas linhas.
    """
    return "Execução em sandbox ainda não implementada."