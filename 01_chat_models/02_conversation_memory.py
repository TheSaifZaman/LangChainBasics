"""
Simple conversation using explicit message types.

What this shows
- How to construct a conversation with `SystemMessage` and `HumanMessage`.
- How `ChatOpenAI.invoke()` accepts a list of messages and returns a message object.
- Where to find the text output (`result.content`).
"""

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")  # try gpt-4o-mini for lower cost/speed

messages = [
    SystemMessage("You are an expert in social media content strategy"), 
    HumanMessage("Give a short tip to create engaging posts on Instagram"), 
]

result = llm.invoke(messages)  # returns an AIMessage-like object

print(result.content)