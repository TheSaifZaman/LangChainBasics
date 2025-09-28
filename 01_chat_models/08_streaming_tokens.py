"""
Streaming tokens from ChatOpenAI.

What this shows
- Use `.stream()` to iterate over the model's incremental outputs.
- Useful for UX where you want to display partial responses.

Run
    python 01_chat_models/08_streaming_tokens.py
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = "Explain the concept of map-reduce in 3 short bullet points."

print("Streaming response:\n")
for chunk in llm.stream(prompt):
    # Each chunk is a message-like object; print partial content as it arrives
    print(chunk.content, end="", flush=True)

print()  # final newline
