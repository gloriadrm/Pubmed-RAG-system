"""
test_openai_api_available.py
-----------------------------
Comprueba que OPENAI_API_KEY está configurada y que el modelo responde —
hace una llamada real (de pago) a la API de OpenAI. Por eso todo el código
va dentro de main(), guardado tras if __name__ == "__main__": una simple
importación (p. ej. al recolectar tests con pytest sobre todo el directorio
test/) no debe disparar una llamada facturable.

Uso:
    python test/test_openai_api_available.py
"""

import os
from dotenv import load_dotenv


def main():
    load_dotenv(override=True)

    api_key = os.environ.get("OPENAI_API_KEY")
    print(f"API key encontrada: {'Sí' if api_key else 'NO — falta en .env'}")
    if api_key:
        print(f"Key empieza por: {api_key[:8]}...")

    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
    response = llm.invoke("Di solo la palabra: funciona")
    print(f"Respuesta OpenAI: {response.content}")


if __name__ == "__main__":
    main()
