"""
Batch inference with LCEL chains.

What this shows
- Use `.batch()` to run the same chain over a list of inputs efficiently.
- Same composition pattern: `prompt | model | StrOutputParser()`.

Run
    python 03_chains/06_batch_inference.py
"""

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise assistant that answers in one sentence."),
    ("human", "Summarize: {text}"),
])

chain = prompt | model | StrOutputParser()

inputs = [
    {"text": "LangChain helps build LLM apps with composable building blocks."},
    {"text": "RAG combines document retrieval with generation to ground outputs."},
    {"text": "Agents can use tools to solve tasks that require actions."},
]

results = chain.batch(inputs)
for i, r in enumerate(results, 1):
    print(f"{i}. {r}")
