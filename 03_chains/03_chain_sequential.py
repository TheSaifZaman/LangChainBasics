"""
Sequential composition of multiple steps using LCEL.

What this shows
- Compose multiple stages: generate -> parse -> reformat -> translate -> parse.
- Use `RunnableLambda` to adapt/prepare intermediate values.
- `StrOutputParser` extracts plain text from model outputs between stages.

Run
    python 03_chains/03_chain_sequential.py
"""

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define prompt templates
animal_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You like telling facts and you tell facts about {animal}."),
        ("human", "Tell me {count} facts."),
    ]
)

# Define a prompt template for translation to French
translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provided text into {language}."),
        ("human", "Translate the following text to {language}: {text}"),
    ]
)

# Define additional processing steps using RunnableLambda
# Example extra step showing you can transform the generated text
count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

# Prepare a dict matching the variables expected by the translation prompt
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "french"})


# Create the combined chain using LangChain Expression Language (LCEL)
# Note: You can insert `count_words` anywhere if you want to inspect/augment output.
# 1. Generate animal facts
# 2. Parse the output to extract plain text
# 3. Prepare the text for translation
# 4. Translate the text to French
# 5. Parse the output to extract plain text
chain = (
    animal_facts_template
    | model
    | StrOutputParser()
    | prepare_for_translation
    | translation_template
    | model
    | StrOutputParser()
)

# Run the chain
result = chain.invoke({"animal": "cat", "count": 2})

# Output
print(result)
