"""
Timeouts and retries with ChatOpenAI.

What this shows
- Configure per-call `timeout` and built-in `max_retries` on the LLM.
- Handle exceptions gracefully and show a simple retry-aware flow.

Run
    python 01_chat_models/07_timeout_retry.py
"""

import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Configure a model with a short timeout and a couple of retries
llm = ChatOpenAI(model="gpt-4o-mini", timeout=5, max_retries=2)

prompt = "Give me one fun fact about honeybees in <= 20 words."

try:
    t0 = time.time()
    result = llm.invoke(prompt)
    dt = time.time() - t0
    print(f"Took {dt:.2f}s\n{result.content}")
except Exception as e:
    print("Request failed after retries:", e)
