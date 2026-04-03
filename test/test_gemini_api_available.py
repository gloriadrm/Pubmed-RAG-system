import os
from dotenv import load_dotenv
load_dotenv(override=True)

api_key = os.environ.get("GOOGLE_API_KEY")
print(f"API key encontrada: {'Sí' if api_key else 'NO — falta en .env'}")
if api_key:
    print(f"Key empieza por: {api_key[:8]}...")

from langchain_google_genai import ChatGoogleGenerativeAI
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
response = llm.invoke("Di solo la palabra: funciona")
print(f"Respuesta Gemini: {response.content}")