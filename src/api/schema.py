"""
schema.py
---------
Modelos Pydantic para los endpoints de la API RAG biomédica.
"""

from typing import Optional
from pydantic import BaseModel, field_validator

from src.rag.structures import normalize_pmc_id


class QueryRequest(BaseModel):
    question: str
    # Opcional: si el usuario ya sabe el PMC ID puede pasarlo directamente y el
    # sistema lo usa con prioridad absoluta (ver src/rag/chain.py) — se valida y
    # normaliza aquí, en la frontera de la API, para no degradar silenciosamente
    # ante un formato inválido (ej. "PMC12ABC" nunca cae a resolución automática:
    # la petición se rechaza con 422 antes de llegar al clasificador o a Qdrant).
    pmc_id: Optional[str] = None

    @field_validator("pmc_id")
    @classmethod
    def _validate_pmc_id(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        return normalize_pmc_id(value)


class SourceDoc(BaseModel):
    title:   str
    section: Optional[str] = None
    pmc_id:  Optional[str] = None
    pm_id:   Optional[str] = None   # PMID de PubMed (siempre presente en docs source=pubmed)
    source:  str            # "pubmed" | "pmc"
    excerpt: Optional[str] = None   # fragmento de texto realmente recuperado (page_content limpio)


class QueryResponse(BaseModel):
    question:           str
    answer:              str
    query_type:          str    # clasificación original del router (thematic_summary | specific_query | transversal_query | none)
    retrieval_strategy:   str    # estrategia de retrieval REALMENTE ejecutada (puede diferir de query_type)
    pmc_id:              Optional[str]        # pmc_id resuelto (si aplica)
    reason:               str    # justificación de la clasificación del router (sin relación con retrieval_note)
    retrieval_note:       Optional[str] = None   # explica degradaciones, pmc_id sin chunks, o 0 resultados
    sources:              list[SourceDoc]      # documentos usados como contexto


class IngestRequest(BaseModel):
    query: str
    n:     int = 10


class IngestResponse(BaseModel):
    query:                  str
    articles_found:         int
    pubmed_indexed:         int
    pmc_found:              int   # artículos con pmc_id resuelto (antes del filtro de licencia)
    pmc_excluded_license:   int   # excluidos por licencia no permitida
    xml_downloaded:         int   # artículos cuyo full text se descargó realmente
    chunks_created:         int
    log:                    list[str]   # líneas de log legibles, para un desplegable


class LicenseCount(BaseModel):
    license: str
    count:   int


class CorpusStatusResponse(BaseModel):
    pubmed_count:      int
    pmc_article_count: int   # PMC IDs distintos con texto completo indexado
    pmc_chunk_count:   int   # total de chunks PMC (varios por artículo)
    licenses:          list[LicenseCount]
    last_updated:      Optional[str] = None   # None si /corpus/update nunca se ha ejecutado


class UpdateJobStartedResponse(BaseModel):
    job_id: str
    state:  str   # siempre "running" — devuelto solo al arrancar el job con éxito


class UpdateResult(BaseModel):
    pubmed_checked:       int   # PMIDs indexados revisados contra los update files de NLM
    pubmed_updated:       int
    pubmed_deleted:       int
    pmc_checked:          int   # PMC IDs indexados revisados contra su metadata OA actual
    pmc_reingested:       int
    pmc_retracted:        int
    pmc_removed_license:  int   # retirados por licencia ya no permitida
    pmc_removed_gone:     int   # retirados por ya no estar en el PMC Open Access Subset


class UpdateJobStatusResponse(BaseModel):
    state:        str                        # idle | running | completed | failed
    job_id:       Optional[str] = None
    phase:        Optional[str] = None        # pubmed | pmc | None (solo con state=running)
    started_at:   Optional[str] = None
    finished_at:  Optional[str] = None
    result:       Optional[UpdateResult] = None
    error:        Optional[str] = None        # solo con state=failed
    log:          list[str] = []


class HealthResponse(BaseModel):
    status:             str
    qdrant:             str
    embedding_provider: str   # modelo de embeddings (fijo, no configurable vía LLM_PROVIDER)
    embedding_status:   str   # loaded | not loaded | error (...)
    llm_provider:       str   # proveedor LLM activo (openai | gemini | ollama)
    llm_model:          str   # modelo concreto del proveedor activo (config, no verificado en cada request)
    llm_status:         str   # configured | reachable | misconfigured (...)
