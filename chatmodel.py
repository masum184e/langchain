import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

os.environ["GOOGLE_API_KEY"] = "AIzaSyA3NDl2Au7apFilX0UHyGxqvm21M3Xw5bY"

# Instantiate the chat model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Define the structured messages
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What are the health benefits of green tea?")
]

# Get response
response = llm.invoke(messages)
print("ChatModel response:\n", response.content)