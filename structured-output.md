# Structured Output
Structured output refers to returning results from a language model (LLM) in a predictable, machine-readable format—such as a Pydantic model, JSON, or a predefined schema. This is important when you want to extract data from text, generate objects for downstream processing, or validate user inputs/output using a fixed schema.

## Why Use Structured Output?
- You need consistency and validity in the format returned by the LLM.
- You want to extract specific fields from a natural language prompt (e.g., name, date, location).
- You need to process the output in other systems (e.g., pass a dictionary to a database, call an API).

## Techniques for Structured Output
1. **Pydantic Output Parsers** – Define a schema using Python's pydantic models.
2. **Structured Output Parser** – LangChain utility to enforce output format.
3. **Function Calling** (OpenAI, Anthropic) – Uses function signatures as schemas.
4. **Output Fixing Parser** – Helps fix invalid outputs that deviate from the schema.

## Examplpe
**1. Define Pyndantic Schema:**
```py
from pydantic import BaseModel, Field
from typing import List

class PersonInfo(BaseModel):
    name: str = Field(..., description="Full name of the person")
    age: int = Field(..., description="Age in years")
    hobbies: List[str] = Field(..., description="List of hobbies")
```
**2. Setup:**
```py
# Create parser using the schema
parser = PydanticOutputParser(pydantic_object=PersonInfo)

# Format prompt
formatted_prompt = prompt.format_messages(
    input_text=input_text,
    format_instructions=parser.get_format_instructions()
)
```
**3. Generate Output:**
```py
response = llm(formatted_prompt)
parsed_output = parser.parse(response.content)

print(parsed_output)
```
