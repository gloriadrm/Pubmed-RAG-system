import os
from dotenv import load_dotenv
load_dotenv(override=True)

api_key = os.environ.get("OPENAI_API_KEY")
print(f"API key encontrada: {'Sí' if api_key else 'NO — falta en .env'}")
if api_key:
    print(f"Key empieza por: {api_key[:8]}...")

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
response = llm.invoke("Di solo la palabra: funciona")
print(f"Respuesta OpenAI: {response.content}")
