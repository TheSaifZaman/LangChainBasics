"""
Few-shot prompting and partial variables with ChatPromptTemplate.

What this shows
- Build a prompt with examples (few-shot) to guide the model.
- Use `partial()` to set fixed variables once, while leaving others dynamic at run time.

Run
    python 02_prompting/02_few_shot_and_partial.py
"""

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini")

# Few-shot examples (system + human + ai triplets can also be used)
examples = [
    ("human", "Classify: 'I love this product!'") ,
    ("ai", "label=positive, reason=expresses strong liking."),
    ("human", "Classify: 'It's okay, not great.'"),
    ("ai", "label=neutral, reason=ambivalent statement."),
]

base_messages = [
    ("system", "You are a sentiment classifier. Always return 'label=<label>, reason=<short_reason>'."),
    *examples,
    ("human", "Classify: '{text}'"),
]

prompt = ChatPromptTemplate.from_messages(base_messages)

# Partial variables: lock the system instruction once, keep text dynamic
# (Demonstration: in this simple example, there are no extra variables to partial.)

result = (prompt | llm).invoke({"text": "The delivery was late and the support was unhelpful."})
print(result.content)
