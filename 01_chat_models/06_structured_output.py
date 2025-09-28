"""
Structured output with Pydantic using ChatOpenAI.with_structured_output().

What this shows
- Define a Pydantic schema for the model to target.
- Use `llm.with_structured_output(Schema)` so `.invoke()` returns a parsed object.
- Great for downstream reliability (no brittle regex/JSON parsing).

Run
    python 01_chat_models/06_structured_output.py
"""

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

load_dotenv()

class WeatherInfo(BaseModel):
    location: str = Field(..., description="City or location name")
    temperature_c: float = Field(..., description="Temperature in Celsius")
    condition: str = Field(..., description="Short weather condition e.g., Sunny")

# Use a fast, cost-effective model for structured output
llm = ChatOpenAI(model="gpt-4o-mini")
structured_llm = llm.with_structured_output(WeatherInfo)

prompt = (
    "Extract the structured weather info from this text: "
    "'In Dhaka it is about 31C and humid; mostly cloudy.'"
)

result = structured_llm.invoke(prompt)
print(result)
print("Location:", result.location)
print("Temp C:", result.temperature_c)
print("Condition:", result.condition)
