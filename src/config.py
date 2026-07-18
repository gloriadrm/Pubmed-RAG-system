"""
config.py
---------
Configuración centralizada del sistema RAG biomédico.
Todos los módulos importan sus constantes desde aquí.
"""

from pathlib import Path

# -------------- PATHS --------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"

# -------------- QDRANT --------------
# Las URLs se pueden sobreescribir con variables de entorno (útil en Docker)

import os

QDRANT_URL      = os.environ.get("QDRANT_URL",      "http://localhost:6333")
COLLECTION_NAME = "biomedical_rag"
VECTOR_SIZE     = 1024

# -------------- EMBEDDINGS --------------

EMBEDDING_MODEL = "BAAI/bge-m3"

# -------------- PUBMED / NCBI --------------

PUBMED_UPDATE_BASE = "https://ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/"
OA_CSV_URL         = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_file_list.csv"

LOCAL_OA_CSV          = DATA_DIR / "oa_file_list.csv"
PUBMED_LAST_UPDATE    = DATA_DIR / "pubmed_last_update.txt"

# -------------- INGESTA / LICENCIAS --------------

# Licencias PMC que permiten reproducción del texto completo.
# Se excluyen variantes ND (No Derivatives) y NO-CC CODE.
ALLOWED_PMC_LICENSES = {
    "CC BY",
    "CC BY-SA",
    "CC BY-NC",
    "CC BY-NC-SA",
    "CC0",
}

# -------------- CHUNKING --------------

CHUNK_TOKEN_LIMIT = 1500   # secciones por debajo de este umbral → un solo chunk
CHUNK_SIZE        = 1000   # tokens por chunk cuando hay que subdividir
CHUNK_OVERLAP     = 100    # solapamiento entre chunks consecutivos

# -------------- RAG / LLM --------------

LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS  = 2048

# Proveedor LLM activo para clasificación y generación (ver src/services/llms.py).
# Valores soportados: "openai" | "gemini" | "ollama". Cambiar de proveedor solo
# requiere esta variable de entorno — el resto del pipeline (chain.py) es
# agnóstico al proveedor concreto.
#
# Solo son obligatorias las variables del proveedor seleccionado (la
# validación ocurre dentro de cada builder en llms.py, no aquí — este módulo
# no falla al importarse aunque falten claves de proveedores no usados).
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openai").strip().lower()

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-mini")

GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

# OLLAMA_BASE_URL se sobrescribe en Docker (ver docker-compose.yml) porque
# Ollama corre en el host, no en un contenedor de este stack.
OLLAMA_BASE_URL  = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL = os.environ.get("OLLAMA_LLM_MODEL", "llama3.1:8b")

RETRIEVAL_K_THEMATIC    = 6   # abstracts PubMed → varios papers, suficiente con 6
RETRIEVAL_K_SPECIFIC    = 10   # chunks de un paper concreto → necesitamos más secciones
RETRIEVAL_K_TRANSVERSAL = 6   # chunks de todos los papers → diversidad > cantidad
