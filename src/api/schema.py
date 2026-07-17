"""
schema.py
---------
Modelos Pydantic para los endpoints de la API RAG biomédica.
"""

from typing import Optional
from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    # Opcional: si el usuario ya sabe el PMC ID puede pasarlo directamente
    # y el sistema salta la etapa de resolución
    pmc_id: Optional[str] = None


class SourceDoc(BaseModel):
    title:   str
    section: Optional[str] = None
    pmc_id:  Optional[str] = None
    pm_id:   Optional[str] = None   # PMID de PubMed (siempre presente en docs source=pubmed)
    source:  str            # "pubmed" | "pmc"


class QueryResponse(BaseModel):
    question:   str
    answer:     str
    query_type: str                  # thematic_summary | specific_query | transversal_query | none
    pmc_id:     Optional[str]        # pmc_id resuelto (si aplica)
    reason:     str                  # justificación del router
    sources:    list[SourceDoc]      # documentos usados como contexto


class IngestRequest(BaseModel):
    query: str
    n:     int = 10


class IngestResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    llm:    str
