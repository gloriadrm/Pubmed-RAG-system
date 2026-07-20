"""
chain.py
--------
Pipeline RAG biomédico usando LangChain LCEL.

Flujo:
  1. classifier_chain  → LLM clasifica la pregunta (SourceSelection)
  2. get_context        → retrieval real en Qdrant (RetrievalResult)
  3. format_docs        → formatea los documentos recuperados como contexto para el LLM
  4. RunnableBranch     → respuesta RAG si está en scope, mensaje de fuera de dominio si no

Arquitectura LCEL:
  RunnablePassthrough.assign(source=classifier_chain)
  → .assign(source_context=get_context)
  → .assign(context=format_docs)
  → RunnableBranch(in_scope → answer_chain, out_of_scope → no_scope_chain)

Prioridad del pmc_id (ver get_context):
  1. request_pmc_id  — pasado estructuradamente en el input del chain (QueryRequest.pmc_id,
                        ya normalizado/validado en la frontera de la API). Tiene prioridad
                        absoluta: fuerza retrieval_strategy='specific_query' incluso si el
                        router clasificó la pregunta como 'none' — is_in_scope() decide el
                        branch final según RetrievalResult.retrieval_strategy, no según
                        source.query_type.
  2. source.pmc_id   — extraído por el clasificador si la pregunta lo menciona explícitamente.
  3. _resolve_pmc_id — autor → título → búsqueda semántica, solo si los dos anteriores fallan.
  Si ninguno resuelve un pmc_id, specific_query degrada a transversal_query (con retrieval_note).

Nota sobre filtros Qdrant:
  Todos los campos del payload están bajo el dict anidado "metadata".
  Los filtros usan dot notation: metadata.source, metadata.pmc_id, etc.
  doc.metadata (LangChain) ya ES ese dict, así que doc.metadata.get("pmc_id") funciona directo.
"""

from operator import itemgetter
from typing import Optional

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.config import RETRIEVAL_K_THEMATIC, RETRIEVAL_K_SPECIFIC, RETRIEVAL_K_TRANSVERSAL
from src.rag.prompts import classifier_prompt, rag_prompt, out_of_scope_prompt
from src.rag.structures import SourceSelection, RetrievalResult, normalize_pmc_id
from src.services.vector_store import qdrant_store
from src.services.llms import llm_langchain


# -------------- CLASSIFIER --------------

classifier_chain = classifier_prompt | llm_langchain.with_structured_output(SourceSelection)


# -------------- HELPERS --------------

def _try_normalize_pmc_id(value: Optional[str]) -> Optional[str]:
    """
    Como normalize_pmc_id, pero nunca lanza — usada dentro del pipeline (no en
    la frontera de la API) para valores que ya deberían venir bien formados
    (extraídos por el clasificador o resueltos desde Qdrant). Si por lo que
    sea no lo están, se tratan como ausentes en vez de romper la petición.
    """
    if not value:
        return None
    try:
        return normalize_pmc_id(value)
    except ValueError:
        return None


def clean_document_content(doc: Document) -> str:
    """
    Limpia el page_content de un documento recuperado, quitando el prefijo
    redundante de título que llevan los embedding_text originales. Los dos
    formatos de origen usan separadores distintos (ver src/ingest/pmc.py y
    src/ingest/pubmed.py):
      PMC:    "título | sección | texto"  (separado por " | ")  → "texto"
      PubMed: "título abstract"           (unidos por un espacio, sin " | ")
              → se usa metadata['title'] para localizar y quitar el prefijo
    """
    content = doc.page_content
    if " | " in content:
        return content.split(" | ", 2)[-1].strip()

    title = doc.metadata.get("title")
    if title and content.startswith(title):
        return content[len(title):].strip()

    return content.strip()


# -------------- RETRIEVAL --------------

async def _resolve_pmc_id(question: str, source_sel: SourceSelection) -> str | None:
    """
    Intenta identificar el pmc_id del artículo cuando no vino ni en el request
    ni extraído del texto por el clasificador. Se ejecuta solo para
    specific_query sin pmc_id ya conocido.

    Estrategia en orden de prioridad:
      1. Filtro por autor en Qdrant (si el LLM extrajo author_last_name)
      2. Búsqueda semántica con el título exacto extraído por el LLM (paper_title)
         → consultar PubMed con el título como query: debería recuperar ese artículo en top-1
      3. Fallback: búsqueda semántica con la pregunta completa en PubMed
    """
    from src.ingest.indexer import client as qdrant_client
    from src.config import COLLECTION_NAME

    pubmed_filter = Filter(
        must=[FieldCondition(key="metadata.source", match=MatchValue(value="pubmed"))]
    )

    # --- 1. Filtro por autor ---
    if source_sel.author_last_name:
        conditions = [
            FieldCondition(key="metadata.source", match=MatchValue(value="pubmed")),
            FieldCondition(
                key="metadata.authors[].last_name",
                match=MatchValue(value=source_sel.author_last_name)
            ),
        ]
        if source_sel.author_name:
            conditions.append(
                FieldCondition(
                    key="metadata.authors[].name",
                    match=MatchValue(value=source_sel.author_name)
                )
            )

        results, _ = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(must=conditions),
            limit=1,
            with_payload=["metadata"],
            with_vectors=False,
        )
        if results:
            pmc_id = results[0].payload.get("metadata", {}).get("pmc_id")
            if pmc_id:
                return pmc_id

    # --- 2. Búsqueda semántica por título extraído ---
    # Usar el título como query es más preciso que usar la pregunta completa:
    # el embedding del título estará muy cerca del embedding del abstract que lo contiene.
    if source_sel.paper_title:
        title_docs = await qdrant_store.asimilarity_search(
            source_sel.paper_title, k=1, filter=pubmed_filter
        )
        if title_docs:
            pmc_id = title_docs[0].metadata.get("pmc_id")
            if pmc_id:
                return pmc_id

    # --- 3. Fallback: búsqueda semántica con la pregunta completa ---
    fallback_docs = await qdrant_store.asimilarity_search(question, k=1, filter=pubmed_filter)
    if fallback_docs:
        return fallback_docs[0].metadata.get("pmc_id")

    return None


async def _search_thematic(question: str) -> list[Document]:
    query_filter = Filter(
        must=[FieldCondition(key="metadata.source", match=MatchValue(value="pubmed"))]
    )
    return await qdrant_store.asimilarity_search(question, k=RETRIEVAL_K_THEMATIC, filter=query_filter)


async def _search_transversal(question: str) -> list[Document]:
    query_filter = Filter(
        must=[FieldCondition(key="metadata.source", match=MatchValue(value="pmc"))]
    )
    return await qdrant_store.amax_marginal_relevance_search(
        question, k=RETRIEVAL_K_TRANSVERSAL, fetch_k=RETRIEVAL_K_TRANSVERSAL * 3, filter=query_filter
    )


async def _retrieve_specific(pmc_id: str, question: str, override_note: Optional[str] = None) -> RetrievalResult:
    """
    Búsqueda specific_query sobre un único artículo ya identificado (venga del
    request, del clasificador o de _resolve_pmc_id). Único punto que ejecuta
    esta búsqueda filtrada — evita duplicarla entre las distintas vías de origen.
    """
    query_filter = Filter(
        must=[
            FieldCondition(key="metadata.source", match=MatchValue(value="pmc")),
            FieldCondition(key="metadata.pmc_id", match=MatchValue(value=pmc_id)),
        ]
    )
    docs = await qdrant_store.asimilarity_search(question, k=RETRIEVAL_K_SPECIFIC, filter=query_filter)

    if not docs:
        # El artículo se identificó (pmc_id resuelto), pero no hay chunks de texto
        # completo indexados para él — típicamente porque su licencia bloqueó la
        # descarga en el pipeline de ingesta (ver src/ingest/pipeline.py). Distinto
        # de "no se pudo identificar ningún artículo" (ver degradación más abajo).
        note = f"El artículo {pmc_id} fue identificado, pero su texto completo no está indexado."
    else:
        note = override_note

    return RetrievalResult(docs=docs, retrieval_strategy="specific_query", resolved_pmc_id=pmc_id, retrieval_note=note)


async def get_context(input_dict) -> RetrievalResult:
    """
    Retrieval real en Qdrant. Devuelve un RetrievalResult que puede diferir de
    la clasificación original del router (input_dict["source"].query_type) —
    ver la prioridad de pmc_id documentada en el docstring del módulo.
    """
    question = input_dict["question"]
    source_sel: SourceSelection = input_dict["source"]
    query_type = source_sel.query_type
    request_pmc_id = input_dict.get("request_pmc_id")  # ya normalizado por QueryRequest, o None

    # Prioridad 1: pmc_id explícito del request — máxima prioridad, incluso
    # sobre query_type='none'. Es un parámetro estructurado y determinista;
    # no debe depender de que el clasificador probabilístico esté de acuerdo.
    if request_pmc_id:
        override_note = None
        if query_type != "specific_query":
            override_note = (
                f"Se priorizó el PMC ID proporcionado explícitamente en la petición "
                f"(el router había clasificado la pregunta como '{query_type}')."
            )
        return await _retrieve_specific(request_pmc_id, question, override_note=override_note)

    if query_type == "none":
        return RetrievalResult(retrieval_strategy="none")

    if query_type == "thematic_summary":
        docs = await _search_thematic(question)
        note = None if docs else "No se encontraron abstracts PubMed relevantes para la consulta."
        return RetrievalResult(docs=docs, retrieval_strategy="thematic_summary", retrieval_note=note)

    if query_type == "transversal_query":
        docs = await _search_transversal(question)
        note = None if docs else "No se encontraron fragmentos de texto completo PMC relevantes para la consulta."
        return RetrievalResult(docs=docs, retrieval_strategy="transversal_query", retrieval_note=note)

    # query_type == "specific_query"
    # Prioridad 2: pmc_id extraído por el clasificador del propio texto de la pregunta.
    classifier_pmc_id = _try_normalize_pmc_id(source_sel.pmc_id)
    if classifier_pmc_id:
        return await _retrieve_specific(classifier_pmc_id, question)

    # Prioridad 3: resolución automática (autor → título → semántica).
    resolved = _try_normalize_pmc_id(await _resolve_pmc_id(question, source_sel))
    if resolved:
        return await _retrieve_specific(resolved, question)

    # Ninguna vía resolvió un pmc_id → degradación explícita a transversal_query.
    docs = await _search_transversal(question)
    return RetrievalResult(
        docs=docs,
        retrieval_strategy="transversal_query",
        retrieval_note="No se pudo identificar un artículo PMC único; se realizó una búsqueda transversal.",
    )


def format_docs(input_dict) -> str:
    retrieval: RetrievalResult = input_dict["source_context"]
    docs = retrieval.docs
    if not docs:
        return "No se encontraron documentos relevantes en la base de datos."

    parts = []
    for i, doc in enumerate(docs, 1):
        meta    = doc.metadata
        title   = meta.get("title", "Artículo sin título")
        section = meta.get("section", "")
        header  = f"[{i}] [{title} — {section}]" if section else f"[{i}] [{title}]"
        parts.append(f"{header}\n{clean_document_content(doc)}")

    return "\n\n---\n\n".join(parts)


# -------------- BRANCHING --------------

def is_in_scope(input_dict) -> bool:
    """
    Decide según la estrategia de retrieval REALMENTE ejecutada, no según la
    clasificación original del router. Esto es lo que permite que un
    request_pmc_id explícito fuerce una respuesta RAG incluso cuando el
    router clasificó la pregunta aislada como 'none'.
    """
    retrieval: RetrievalResult = input_dict["source_context"]
    return retrieval.retrieval_strategy != "none"


answer_chain   = rag_prompt          | llm_langchain | StrOutputParser()
no_scope_chain = out_of_scope_prompt | llm_langchain | StrOutputParser()


# -------------- PIPELINE COMPLETO --------------

rag_chain = (
    # Paso 1: clasificar la pregunta (siempre sobre el texto original, sin modificar)
    RunnablePassthrough.assign(
        source=(itemgetter("question") | classifier_chain)
    )
    # Paso 2: recuperar contexto (async) + formatearlo
    | RunnablePassthrough.assign(
        source_context=RunnableLambda(func=lambda x: None, afunc=get_context)
    )
    | RunnablePassthrough.assign(
        context=RunnableLambda(format_docs)
    )
    # Paso 3: generar respuesta según la estrategia de retrieval efectiva
    | RunnableBranch(
        (is_in_scope, RunnablePassthrough.assign(answer=answer_chain)),
        RunnablePassthrough.assign(answer=no_scope_chain),
    )
)
