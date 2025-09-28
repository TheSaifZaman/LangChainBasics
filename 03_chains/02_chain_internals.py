"""
Peek under the hood with RunnableLambda and RunnableSequence.

What this shows
- How to build a chain step-by-step using `RunnableLambda` and `RunnableSequence`.
- Each step transforms the data: format prompt -> invoke model -> parse content.
- `invoke()` runs the composed sequence and returns the final value.
"""

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4")

# Define prompt templates
prompt_template = ChatPromptTemplate.from_messages(
     [
        ("system", "You love facts and you tell facts about {animal}"),
        ("human", "Tell me {count} facts."),
    ]
)

# Create individual runnables (steps in the chain)
# 1) format_prompt: fills the template and returns a PromptValue
format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
# 2) invoke_model: calls the model with the prompt's messages and returns an AIMessage
invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
# 3) parse_output: extracts the string content from the AIMessage
parse_output = RunnableLambda(lambda x: x.content)

# Create the RunnableSequence (equivalent to the LCEL chain)
chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# Run the chain
response = chain.invoke({"animal": "cat", "count": 2})

# Output
print(response)
