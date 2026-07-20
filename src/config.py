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
PUBMED_LAST_UPDATE = DATA_DIR / "pubmed_last_update.txt"

# Tope de update files procesados por ejecución de run_pubmed_update(). NLM
# publica varios ficheros al día (~100-200MB descomprimidos cada uno); tras
# un periodo sin actualizar, el backlog pendiente puede ser de decenas o
# cientos de ficheros — procesarlos todos en una sola petición síncrona de
# /corpus/update tardaría muchos minutos u horas, muy por encima de lo
# razonable para un botón de UI. El progreso se guarda fichero a fichero
# (save_last_processed_file), así que sucesivas ejecuciones retoman donde
# quedó la anterior sin perder ni repetir trabajo.
#
# Medido en vivo: incluso 3 ficheros tardaron ~12 min en este entorno (el
# parseo XML con ElementTree es CPU-bound puro-Python, no libera el GIL —
# bloquea también el resto de peticiones a la API mientras corre). 1 fichero
# mantiene cada clic del botón dentro de "varios minutos" de verdad.
PUBMED_MAX_FILES_PER_RUN = 1

# -------------- PMC OPEN ACCESS (Cloud Service en AWS) --------------
#
# NCBI retiró oa_file_list.csv del FTP Service en 2026 (ver anuncio:
# https://ncbiinsights.ncbi.nlm.nih.gov/2026/02/12/pmc-article-dataset-distribution-services/).
# El reemplazo no es un CSV único con licencia/fecha — es un bucket S3
# público (pmc-oa-opendata) con un JSON de metadata individual por
# artículo, más PMC-ids.csv.gz (bulk, no deprecado) para el mapeo
# PMID→PMCID. Ver src/data_actualization/pmc_oa_client.py.

PMC_IDS_URL       = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/PMC-ids.csv.gz"
LOCAL_PMC_IDS_CSV = DATA_DIR / "PMC-ids.csv"

PMC_OA_METADATA_BASE_URL = "https://pmc-oa-opendata.s3.amazonaws.com/metadata"

# -------------- ACTUALIZACIÓN DEL CORPUS --------------

# Marca de tiempo de la última ejecución completa (PubMed + PMC) de
# POST /corpus/update — expuesta en GET /corpus/status. No existe hasta la
# primera ejecución.
LAST_CORPUS_UPDATE_FILE = DATA_DIR / "last_corpus_update.txt"

# Estado del job de actualización en curso/más reciente — único punto de
# coordinación entre el proceso de la API y el proceso worker independiente
# que ejecuta el trabajo (ver src/data_actualization/update_job.py).
UPDATE_JOB_STATUS_FILE = DATA_DIR / "update_job_status.json"

# Protección ligera del botón "Actualizar corpus" del frontend: al ser un
# endpoint sin autenticación (proyecto de portfolio, no producción), este
# cooldown evita que compartir la URL permita lanzar decenas de
# actualizaciones reales seguidas. Ver POST /corpus/update en src/api/main.py.
CORPUS_UPDATE_COOLDOWN_SECONDS = 300

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
