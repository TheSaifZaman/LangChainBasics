"""
ReAct-style agent with a custom tool.

What this shows
- How to declare a simple `@tool` (get_system_time) for the agent to call.
- How to use a standard ReAct prompt from LangChain Hub (`hwchase17/react`).
- How to create the agent and run it via `AgentExecutor.invoke()`.

Notes
- The example tool returns your local system time; ensure your system clock/timezone
  reflect the location you expect. The model may reason about location vs. timezone
  as part of the ReAct chain-of-thought.

Run
    python 05_agents/01_agents_basics.py
"""

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
import datetime
from langchain.agents import tool

load_dotenv()

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S"):
    """ Returns the current date and time in the specified format """

    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime(format)
    return formatted_time

# Backed by your chosen LLM
llm = ChatOpenAI(model="gpt-4")

# A simple question that may require reasoning about timezones
query = "What is the current time in London? (You are in India). Just show the current time and not the date"

# Pull a standard ReAct prompt
prompt_template = hub.pull("hwchase17/react")

# Register tools the agent can use
tools = [get_system_time]

# Create a ReAct agent and an executor around it
agent = create_react_agent(llm, tools, prompt_template)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent
agent_executor.invoke({"input": query})
