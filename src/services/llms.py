"""
llms.py
-------
Instancia del LLM para la cadena RAG.
"""

# Usa llama3.1:8b ejecutado localmente a través de Ollama.
# Ollama debe estar corriendo en localhost:11434 antes de lanzar la API.
# from langchain_ollama import ChatOllama
# from src.config import LLM_MODEL, LLM_TEMPERATURE, OLLAMA_BASE_URL

# llm_langchain = ChatOllama(
#     model=LLM_MODEL,
#     temperature=LLM_TEMPERATURE,
#     base_url=OLLAMA_BASE_URL,
# )

# Gemini 2.0 Flash a través de la API de Google.
# Requiere GOOGLE_API_KEY en el entorno (.env).
from langchain_google_genai import ChatGoogleGenerativeAI
from src.config import GEMINI_MODEL, LLM_TEMPERATURE

llm_langchain = ChatGoogleGenerativeAI(
    model=GEMINI_MODEL,
    temperature=LLM_TEMPERATURE,
)