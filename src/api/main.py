"""
main.py
-------
API FastAPI del sistema RAG biomédico.

Endpoints:
  GET  /health          → estado de Qdrant, embeddings (BGE-M3) y del proveedor LLM activo
  POST /query           → pipeline RAG completo (clasificador → retrieval → LLM)
  POST /ingest          → lanza ingesta temática de nuevos artículos
  GET  /                → frontend estático (demo web)

Lanzar con:
    uvicorn src.api.main:app --reload --port 8000

Nota sobre LLM_PROVIDER: el proveedor LLM se resuelve una única vez al
importar src/services/llms.py (arranque del proceso). Cambiar LLM_PROVIDER
en .env no tiene efecto en un proceso ya corriendo — hace falta reiniciar
la API (en Docker: `docker compose up -d --force-recreate api`).
"""

import os

import requests as http_requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from src.api.schema import (
    QueryRequest, QueryResponse, SourceDoc,
    IngestRequest, IngestResponse,
    HealthResponse,
)
from src.config import QDRANT_URL, OLLAMA_BASE_URL, LLM_PROVIDER
from src.rag.chain import rag_chain

app = FastAPI(
    title="Biomedical RAG API",
    description="Sistema RAG sobre artículos PubMed + PMC para consultas biomédicas.",
    version="1.0.0",
)


# -------------- HEALTH --------------

@app.get("/health", response_model=HealthResponse)
def health():
    """
    Comprueba Qdrant, el modelo de embeddings (BGE-M3, local) y el estado de
    configuración/alcance del proveedor LLM activo (LLM_PROVIDER) — tres
    componentes independientes, deliberadamente no conflados entre sí:
    Ollama puede estar caído sin afectar a los embeddings (BGE-M3 corre en
    proceso, vía sentence-transformers, sin red) ni al LLM si el proveedor
    activo es OpenAI o Gemini.

    No hace ninguna llamada de generación de pago ni recalcula embeddings en
    cada petición: para OpenAI/Gemini solo valida presencia de credenciales;
    para Ollama hace ping a /api/tags (gratis, local); para embeddings
    comprueba que el modelo ya está cargado en memoria (se carga una única
    vez al arrancar el proceso — un encode() real de verificación vive en
    test/test_health.py, no aquí, para mantener /health barato).
    """

    # Qdrant
    try:
        r = http_requests.get(f"{QDRANT_URL}/healthz", timeout=3)
        qdrant_status = "ok" if r.status_code == 200 else f"error {r.status_code}"
    except Exception as e:
        qdrant_status = f"unreachable ({e})"

    # Embeddings (BGE-M3) — modelo local, no depende de Ollama ni de ningún
    # servicio de red. Se carga una única vez al importar
    # src.services.embeddings (arranque del proceso); aquí solo se comprueba
    # que esa instancia existe, sin volver a ejecutar inferencia en cada
    # petición a /health.
    embedding_provider = "bge-m3"
    try:
        from src.services.embeddings import embedding_model
        embedding_status = "loaded" if embedding_model is not None else "not loaded"
    except Exception as e:
        embedding_status = f"error ({e})"

    # LLM activo
    if LLM_PROVIDER == "openai":
        llm_status = "configured" if os.environ.get("OPENAI_API_KEY") else "misconfigured (missing OPENAI_API_KEY)"
    elif LLM_PROVIDER == "gemini":
        has_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        llm_status = "configured" if has_key else "misconfigured (missing GEMINI_API_KEY or GOOGLE_API_KEY)"
    elif LLM_PROVIDER == "ollama":
        try:
            r = http_requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
            llm_status = "reachable" if r.status_code == 200 else f"error {r.status_code}"
        except Exception as e:
            llm_status = f"unreachable ({e})"
    else:
        llm_status = f"misconfigured (unknown provider '{LLM_PROVIDER}')"

    embedding_ok = embedding_status == "loaded"
    llm_ok = llm_status in ("configured", "reachable")
    overall = "ok" if qdrant_status == "ok" and embedding_ok and llm_ok else "degraded"

    return HealthResponse(
        status=overall,
        qdrant=qdrant_status,
        embedding_provider=embedding_provider,
        embedding_status=embedding_status,
        llm_provider=LLM_PROVIDER,
        llm_status=llm_status,
    )


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

# -------------- FRONTEND ESTÁTICO --------------
# Debe montarse el último: como catch-all en "/", cualquier ruta de API
# registrada arriba (/health, /query, /ingest, /docs) tiene prioridad.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
