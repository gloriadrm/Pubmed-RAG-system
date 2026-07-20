"""
llms.py
-------
Instancia del LLM para la cadena RAG.

El proveedor activo se selecciona mediante la variable de entorno
LLM_PROVIDER (ver src/config.py, default "openai"; valores soportados:
"openai" | "gemini" | "ollama"). Cada builder valida únicamente las
variables de entorno que su propio proveedor necesita — seleccionar
"openai" nunca exige una clave de Gemini, y viceversa. Añadir un
proveedor nuevo solo requiere registrar un builder en _PROVIDERS — el
resto del pipeline (src/rag/chain.py) consume `llm_langchain` como un
BaseChatModel de LangChain y no conoce el proveedor concreto.
"""

import os

from src.config import LLM_PROVIDER, LLM_TEMPERATURE


def _require_env(*names: str, provider: str):
    """
    Devuelve el valor de la primera variable de entorno presente en `names`.
    Lanza un error claro (nombrando las variables aceptadas) si ninguna está
    configurada.
    """
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    joined = " o ".join(names)
    raise ValueError(
        f"Falta la variable de entorno {joined}, requerida para LLM_PROVIDER='{provider}'."
    )


def _build_openai():
    """OpenAI (modelo configurable vía OPENAI_MODEL). Requiere OPENAI_API_KEY."""
    from langchain_openai import ChatOpenAI
    from src.config import OPENAI_MODEL

    _require_env("OPENAI_API_KEY", provider="openai")

    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=LLM_TEMPERATURE,
    )


def _build_gemini():
    """
    Gemini (modelo configurable vía GEMINI_MODEL). Requiere GEMINI_API_KEY;
    GOOGLE_API_KEY se acepta como alias heredado si GEMINI_API_KEY no está
    definida (prioridad: GEMINI_API_KEY primero).
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    from src.config import GEMINI_MODEL

    api_key = _require_env("GEMINI_API_KEY", "GOOGLE_API_KEY", provider="gemini")

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        temperature=LLM_TEMPERATURE,
        google_api_key=api_key,
    )


def _build_ollama():
    """Ollama local (modelo configurable vía OLLAMA_LLM_MODEL). No requiere API key."""
    from langchain_ollama import ChatOllama
    from src.config import OLLAMA_BASE_URL, OLLAMA_LLM_MODEL

    return ChatOllama(
        model=OLLAMA_LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        base_url=OLLAMA_BASE_URL,
    )


_PROVIDERS = {
    "openai": _build_openai,
    "gemini": _build_gemini,
    "ollama": _build_ollama,
}


def get_llm():
    """Factoría: instancia el LLM activo según LLM_PROVIDER."""
    try:
        builder = _PROVIDERS[LLM_PROVIDER]
    except KeyError:
        raise ValueError(
            f"Proveedor LLM desconocido: '{LLM_PROVIDER}'. "
            f"Proveedores soportados: {', '.join(sorted(_PROVIDERS))}."
        )
    return builder()


llm_langchain = get_llm()
