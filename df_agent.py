from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
import pandas as pd

from langchain.agents.agent_types import AgentType


df = pd.read_csv("df_rent.csv")


agent = create_pandas_dataframe_agent(
    ChatOpenAI(model="gpt-3.5-turbo-0613"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)
agent.invoke("Quantas linhas há no conjunto de dados?")