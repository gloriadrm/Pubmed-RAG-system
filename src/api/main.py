"""
main.py
-------
API FastAPI del sistema RAG biomédico.

Endpoints:
  GET  /health                 → estado de Qdrant, embeddings (BGE-M3) y del proveedor LLM activo
  POST /query                  → pipeline RAG completo (clasificador → retrieval → LLM)
  POST /ingest                 → lanza ingesta temática de nuevos artículos (amplía el corpus)
  GET  /corpus/status          → conteos del corpus indexado, licencias, última actualización
  POST /corpus/update          → lanza en segundo plano la revisión del corpus (nunca lo amplía),
                                  responde de inmediato con {job_id, state}
  GET  /corpus/update/status   → estado del job de actualización (running/completed/failed)
  GET  /                       → frontend estático (demo web)

Lanzar con:
    uvicorn src.api.main:app --reload --port 8000

Nota sobre LLM_PROVIDER: el proveedor LLM se resuelve una única vez al
importar src/services/llms.py (arranque del proceso). Cambiar LLM_PROVIDER
en .env no tiene efecto en un proceso ya corriendo — hace falta reiniciar
la API (en Docker: `docker compose up -d --force-recreate api`).
"""

import os
import threading
import uuid
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from multiprocessing import get_context

import requests as http_requests
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from src.api.schema import (
    QueryRequest, QueryResponse, SourceDoc,
    IngestRequest, IngestResponse,
    CorpusStatusResponse, LicenseCount,
    UpdateJobStartedResponse, UpdateJobStatusResponse,
    HealthResponse,
)
from src.config import (
    QDRANT_URL, OLLAMA_BASE_URL, LLM_PROVIDER,
    OPENAI_MODEL, GEMINI_MODEL, OLLAMA_LLM_MODEL,
    LAST_CORPUS_UPDATE_FILE, CORPUS_UPDATE_COOLDOWN_SECONDS,
)
from src.data_actualization.update_job import (
    read_job_status, start_job_status, run_update_job, mark_interrupted_if_running,
)
from src.rag.chain import rag_chain, clean_document_content

# El parseo de los update files de PubMed es CPU-bound (ElementTree puro
# Python, no libera el GIL) — ejecutarlo en el propio proceso de uvicorn,
# aunque fuera en un thread, bloquearía /health, /query y el resto de
# peticiones durante todo el tiempo que dure. Un único proceso worker
# aparte evita esto sin necesitar Celery/Redis para una demo de portfolio:
# ver src/data_actualization/update_job.py y POST /corpus/update más abajo.
#
# TODO: Python está desaconsejando progresivamente fork() en procesos
# multihilo (uvicorn lo es) — ver la DeprecationWarning que emite
# multiprocessing.popen_fork al usarlo así (riesgo teórico de deadlock en
# el hijo si el fork ocurre mientras otro hilo tiene un lock tomado). No
# se ha observado ningún problema en las pruebas ni en uso real, y "spawn"
# obligaría a que run_update_job() y sus dependencias fueran íntegramente
# reimportables sin estado heredado del padre (más lento, sin reutilizar el
# modelo BGE-M3 ya cargado en memoria). Si este proyecto crece más allá de
# una demo de portfolio, sustituir por mp_context=get_context("spawn") o por
# un worker dedicado fuera del proceso de la API.
_update_executor = ProcessPoolExecutor(max_workers=1, mp_context=get_context("fork"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Si el proceso se reinició con un job marcado "running" en disco, ese
    # job murió sin terminar — lo marcamos "failed" para no dejar al
    # frontend creyendo indefinidamente que sigue en curso.
    mark_interrupted_if_running()
    yield
    _update_executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="Biomedical RAG API",
    description="Sistema RAG sobre artículos PubMed + PMC para consultas biomédicas.",
    version="1.0.0",
    lifespan=lifespan,
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
    llm_model = {"openai": OPENAI_MODEL, "gemini": GEMINI_MODEL, "ollama": OLLAMA_LLM_MODEL}.get(LLM_PROVIDER, "?")
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
        llm_model=llm_model,
        llm_status=llm_status,
    )


# -------------- QUERY (RAG principal) --------------

@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Pipeline RAG completo:
      1. El router clasifica la pregunta original, sin modificar
         (thematic_summary / specific_query / transversal_query / none)
      2. Se recupera contexto de Qdrant según la estrategia efectiva — que puede
         diferir de la clasificación si se pasó pmc_id explícito (prioridad
         absoluta, incluso sobre query_type='none') o si specific_query degrada
         a transversal_query por no resolver un artículo único. Ver
         src/rag/chain.py para la prioridad completa y src/rag/structures.py
         para RetrievalResult.
      3. El LLM genera la respuesta basada en el contexto realmente recuperado

    request.pmc_id (si se proporciona) ya viene validado y normalizado por
    QueryRequest — viaja al chain como campo estructurado, nunca se inyecta
    como texto en la pregunta.
    """
    chain_input = {"question": request.question, "request_pmc_id": request.pmc_id}

    try:
        result = await rag_chain.ainvoke(chain_input)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Clasificación original del router (nunca sobrescrita)
    source_sel = result.get("source")
    query_type = source_sel.query_type if source_sel else "unknown"
    reason = source_sel.reason if source_sel else ""

    # Estrategia de retrieval realmente ejecutada (puede diferir de query_type)
    retrieval = result.get("source_context")
    retrieval_strategy = retrieval.retrieval_strategy if retrieval else "none"
    pmc_id_resolved = retrieval.resolved_pmc_id if retrieval else request.pmc_id
    retrieval_note = retrieval.retrieval_note if retrieval else None
    docs = retrieval.docs if retrieval else []

    # Construir lista de fuentes desde los documentos recuperados
    sources = []
    for doc in docs:
        meta = doc.metadata
        sources.append(SourceDoc(
            title=meta.get("title", ""),
            section=meta.get("section"),
            pmc_id=meta.get("pmc_id"),
            pm_id=meta.get("pm_id"),
            source=meta.get("source", ""),
            excerpt=clean_document_content(doc),
        ))

    return QueryResponse(
        question=request.question,
        answer=result.get("answer", ""),
        query_type=query_type,
        retrieval_strategy=retrieval_strategy,
        pmc_id=pmc_id_resolved,
        reason=reason,
        retrieval_note=retrieval_note,
        sources=sources,
    )


# -------------- INGEST (amplía el corpus) --------------

@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest):
    """
    Lanza el pipeline de ingesta para una query temática — añade
    conocimiento nuevo al corpus. Útil para indexar artículos sin
    necesidad de tocar el código.
    """
    try:
        from src.data_actualization.oa_updater import get_oa_df
        from src.ingest.pipeline import ingest_query
        from src.ingest.indexer import create_collection

        create_collection(recreate=False)
        oa_df = get_oa_df()
        summary = ingest_query(query=request.query, n=request.n, oa_df=oa_df)
        return IngestResponse(query=request.query, **summary)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# -------------- CORPUS: estado y actualización (mantiene, no amplía) --------------

def _get_corpus_status() -> CorpusStatusResponse:
    from collections import Counter
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from src.ingest.indexer import client, COLLECTION_NAME

    pubmed_count = client.count(
        collection_name=COLLECTION_NAME,
        count_filter=Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value="pubmed"))]),
    ).count

    pmc_chunk_count = 0
    pmc_licenses: dict[str, str] = {}   # pmc_id -> license (dedup por artículo)
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=[FieldCondition(key="metadata.source", match=MatchValue(value="pmc"))]),
            limit=256,
            offset=offset,
            with_payload=["metadata"],
            with_vectors=False,
        )
        for point in points:
            pmc_chunk_count += 1
            md = point.payload.get("metadata", {})
            pmc_id = md.get("pmc_id")
            if pmc_id:
                pmc_licenses[pmc_id] = md.get("license") or "desconocida"
        if offset is None:
            break

    license_counts = Counter(pmc_licenses.values())
    licenses = [
        LicenseCount(license=lic, count=n)
        for lic, n in sorted(license_counts.items(), key=lambda kv: -kv[1])
    ]

    last_updated = LAST_CORPUS_UPDATE_FILE.read_text().strip() if LAST_CORPUS_UPDATE_FILE.exists() else None

    return CorpusStatusResponse(
        pubmed_count=pubmed_count,
        pmc_article_count=len(pmc_licenses),
        pmc_chunk_count=pmc_chunk_count,
        licenses=licenses,
        last_updated=last_updated,
    )


@app.get("/corpus/status", response_model=CorpusStatusResponse)
def corpus_status():
    """Conteos del corpus indexado (PubMed, PMC, chunks, licencias) y
    fecha de la última ejecución completa de /corpus/update. Solo lectura,
    barato — sin cooldown ni bloqueo."""
    try:
        return _get_corpus_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Protección ligera del arranque del job: no hay autenticación (proyecto de
# portfolio), así que este lock (solo protege el check-y-arranque, no el
# trabajo en sí) + un cooldown tras cada ejecución evitan que compartir la
# URL permita lanzar decenas de actualizaciones reales seguidas contra
# NCBI/Qdrant. Estado en memoria del proceso — se resetea si la API se
# reinicia, lo cual es aceptable para este propósito. El estado del job en
# sí (running/completed/failed) vive en disco, no aquí — ver update_job.py.
_start_lock = threading.Lock()


@app.post("/corpus/update", response_model=UpdateJobStartedResponse)
def start_corpus_update():
    """
    Lanza la revisión del corpus ya indexado en un proceso worker aparte y
    responde inmediatamente — nunca mantiene la petición HTTP abierta
    mientras dura el trabajo real (puede tardar minutos). Consultar el
    progreso con GET /corpus/update/status.

      - PubMed: aplica revisiones/borrados de los update files de NLM
        pendientes desde la última ejecución (como mucho
        PUBMED_MAX_FILES_PER_RUN ficheros por ejecución).
      - PMC: consulta la metadata OA actual de cada artículo ya indexado
        (licencia, retractación, cambios) y aplica bajas/reingestas.

    Nunca amplía el corpus — eso es responsabilidad exclusiva de /ingest.
    """
    with _start_lock:
        status = read_job_status()
        if status.get("state") == "running":
            raise HTTPException(status_code=409, detail="Ya hay una actualización en curso.")

        if status.get("finished_at"):
            finished = datetime.fromisoformat(status["finished_at"])
            elapsed = (datetime.now(timezone.utc) - finished).total_seconds()
            if elapsed < CORPUS_UPDATE_COOLDOWN_SECONDS:
                remaining = int(CORPUS_UPDATE_COOLDOWN_SECONDS - elapsed)
                raise HTTPException(
                    status_code=429,
                    detail=f"Actualización reciente. Vuelve a intentarlo en {remaining}s.",
                )

        job_id = str(uuid.uuid4())
        # Escrito de forma SÍNCRONA aquí, antes de someter el trabajo al
        # executor: así una segunda petición que llegue mientras el job
        # todavía no ha arrancado en el worker ya ve "running", no "idle".
        start_job_status(job_id)

    _update_executor.submit(run_update_job, job_id)
    return UpdateJobStartedResponse(job_id=job_id, state="running")


@app.get("/corpus/update/status", response_model=UpdateJobStatusResponse)
def corpus_update_status():
    """Estado del job de actualización más reciente (running/completed/
    failed) — barato, solo lee el fichero de estado. Pensado para polling
    desde el frontend mientras el job está en curso."""
    return UpdateJobStatusResponse(**read_job_status())


class NoCacheStaticFiles(StaticFiles):
    """Sin Cache-Control explícito, los navegadores aplican caché heurística
    (RFC 7234 §4.2.2) sobre JS/CSS aunque cambien entre despliegues, sirviendo
    versiones obsoletas sin que el usuario lo note. Forzamos revalidación
    (ETag/Last-Modified) en cada petición en lugar de cachear a ciegas."""

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


# -------------- FRONTEND ESTÁTICO --------------
# Debe montarse el último: como catch-all en "/", cualquier ruta de API
# registrada arriba (/health, /query, /ingest, /docs) tiene prioridad.
app.mount("/", NoCacheStaticFiles(directory="static", html=True), name="static")
