"""
main.py
-------
API FastAPI del sistema RAG biomédico.

Endpoints:
  GET  /health          → estado de Qdrant y Ollama
  POST /query           → pipeline RAG completo (clasificador → retrieval → LLM)
  POST /ingest          → lanza ingesta temática de nuevos artículos

Lanzar con:
    uvicorn src.api.main:app --reload --port 8000
"""

import os
import requests as http_requests
from fastapi import FastAPI, HTTPException

from src.api.schema import (
    QueryRequest, QueryResponse, SourceDoc,
    IngestRequest, IngestResponse,
    HealthResponse,
)
from src.config import QDRANT_URL
from src.rag.chain import rag_chain

app = FastAPI(
    title="Biomedical RAG API",
    description="Sistema RAG sobre artículos PubMed + PMC para consultas biomédicas.",
    version="1.0.0",
)


# -------------- HEALTH --------------

@app.get("/health", response_model=HealthResponse)
def health():
    """Comprueba que Qdrant está accesible y que el proveedor LLM activo está configurado."""

    # Qdrant
    try:
        r = http_requests.get(f"{QDRANT_URL}/healthz", timeout=3)
        qdrant_status = "ok" if r.status_code == 200 else f"error {r.status_code}"
    except Exception as e:
        qdrant_status = f"unreachable ({e})"

    # LLM activo — actualmente Gemini (ver src/services/llms.py). Ollama está
    # comentado y no es el proveedor en uso, por eso no se comprueba aquí.
    if os.environ.get("GOOGLE_API_KEY"):
        llm_status = "gemini"
    else:
        llm_status = "gemini (missing GOOGLE_API_KEY)"

    overall = "ok" if qdrant_status == "ok" and llm_status == "gemini" else "degraded"

    return HealthResponse(status=overall, qdrant=qdrant_status, llm=llm_status)


# -------------- QUERY (RAG principal) --------------

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Pipeline RAG completo:
      1. El router clasifica la pregunta (thematic_summary / specific_query / transversal_query / none)
      2. Se recuperan documentos de Qdrant según el tipo
      3. El LLM genera la respuesta basada en el contexto

    Si se pasa pmc_id en el request, se inyecta en la pregunta para forzar specific_query
    sobre ese artículo concreto.
    """
    question = request.question

    # Si el usuario pasa pmc_id directamente, lo inyectamos en la pregunta
    # para que el clasificador lo detecte, o lo añadimos al input del chain
    chain_input = {"question": question}
    if request.pmc_id:
        chain_input["question"] = f"{question} (PMC ID: {request.pmc_id})"

    try:
        result = await rag_chain.ainvoke(chain_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Extraer info del clasificador
    source_sel = result.get("source")
    query_type = source_sel.query_type if source_sel else "unknown"
    pmc_id_resolved = source_sel.pmc_id if source_sel else request.pmc_id
    reason = source_sel.reason if source_sel else ""

    # Construir lista de fuentes desde los documentos recuperados
    sources = []
    for doc in result.get("source_context") or []:
        meta = doc.metadata
        sources.append(SourceDoc(
            title=meta.get("title", ""),
            section=meta.get("section"),
            pmc_id=meta.get("pmc_id"),
            pm_id=meta.get("pm_id"),
            source=meta.get("source", ""),
        ))

    return QueryResponse(
        question=request.question,
        answer=result.get("answer", ""),
        query_type=query_type,
        pmc_id=pmc_id_resolved,
        reason=reason,
        sources=sources,
    )


# -------------- INGEST --------------

@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """
    Lanza el pipeline de ingesta para una query temática.
    Útil para añadir nuevos artículos sin necesidad de tocar el código.
    """
    try:
        from src.data_actualization.oa_updater import get_oa_df
        from src.ingest.pipeline import ingest_query
        from src.ingest.indexer import create_collection

        create_collection(recreate=False)
        oa_df = get_oa_df()
        ingest_query(query=request.query, n=request.n, oa_df=oa_df)
        return IngestResponse(message=f"Ingesta completada para query: '{request.query}'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
