import os
from langchain_google_genai import GoogleGenerativeAI

os.environ["GOOGLE_API_KEY"] = "AIzaSyA3NDl2Au7apFilX0UHyGxqvm21M3Xw5bY"

# Instantiate the plain LLM
llm = GoogleGenerativeAI(model="gemini-2.0-flash")

# Send a simple string prompt
prompt = "List three benefits of drinking green tea."
response = llm.invoke(prompt)

print("LLM response:\n", response)