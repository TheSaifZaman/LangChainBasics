"""
Basic chat model invocation with LangChain.

What this shows
- How to initialize a chat model (`ChatOpenAI`).
- How to call `invoke()` with a simple string prompt.
- That `invoke()` returns a `AIMessage`-like object; the actual text lives in `result.content`.

Run
    python 01_chat_models/01_basic_invoke.py

Prerequisites
- Set `OPENAI_API_KEY` in your `.env` (see project README).
"""
 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Initialize a default ChatOpenAI model (you can switch to gpt-4o/gpt-4o-mini)
llm = ChatOpenAI(model="gpt-4")

# Invoke the model with a simple prompt
result = llm.invoke("What is the current time in India?")

# You can inspect the whole result object or just print result.content.
print(result)
# print(result.content)
