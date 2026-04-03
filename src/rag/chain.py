"""
chain.py
--------
Pipeline RAG biomédico usando LangChain LCEL.

Flujo:
  1. classifier_chain  → LLM clasifica la pregunta (SourceSelection)
  2. get_context        → búsqueda semántica en Qdrant según el tipo detectado:
       - thematic_summary   → source=pubmed (abstracts)
       - specific_query     → source=pmc + pmc_id (si se extrajo del texto)
       - transversal_query  → source=pmc (texto completo, todos los artículos)
  3. format_docs        → formatea los documentos recuperados como contexto para el LLM
  4. RunnableBranch     → respuesta RAG si está en scope, mensaje de fuera de dominio si no

Arquitectura LCEL:
  RunnablePassthrough.assign(source=classifier_chain)
  → .assign(source_context=get_context)
  → .assign(context=format_docs)
  → RunnableBranch(in_scope → answer_chain, out_of_scope → no_scope_chain)

Nota sobre filtros Qdrant:
  Todos los campos del payload están bajo el dict anidado "metadata".
  Los filtros usan dot notation: metadata.source, metadata.pmc_id, etc.
  doc.metadata (LangChain) ya ES ese dict, así que doc.metadata.get("pmc_id") funciona directo.
"""

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableBranch
from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.config import RETRIEVAL_K_THEMATIC, RETRIEVAL_K_SPECIFIC, RETRIEVAL_K_TRANSVERSAL
from src.rag.prompts import classifier_prompt, rag_prompt, out_of_scope_prompt
from src.rag.structures import SourceSelection
from src.services.vector_store import qdrant_store
from src.services.llms import llm_langchain


# -------------- CLASSIFIER --------------

classifier_chain = classifier_prompt | llm_langchain.with_structured_output(SourceSelection)


# -------------- RETRIEVAL --------------

async def _resolve_pmc_id(question: str, source_sel) -> str | None:
    """
    Intenta identificar el pmc_id del artículo cuando el usuario no lo proporcionó
    directamente. Se ejecuta solo para specific_query sin pmc_id explícito.

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


async def get_context(input_dict) -> list:
    """
    Búsqueda semántica en Qdrant según el tipo de consulta detectado.

      thematic_summary   → source=pubmed (abstracts)
      transversal_query  → source=pmc (texto completo, todos los artículos)
      specific_query     → two-stage:
                            1. Resolver pmc_id si no viene explícito (autor / semántica PubMed)
                            2. Búsqueda semántica source=pmc filtrada por pmc_id
    """
    question = input_dict["question"]
    source_sel = input_dict["source"]
    query_type = source_sel.query_type

    if query_type == "none":
        return []

    pmc_id = source_sel.pmc_id
    effective_query_type = query_type

    if query_type == "thematic_summary":
        query_filter = Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value="pubmed"))]
        )

    elif query_type == "specific_query":
        # Etapa 1: resolver pmc_id si no lo tenemos ya
        if not pmc_id:
            pmc_id = await _resolve_pmc_id(question, source_sel)

        # Si no logramos resolverlo, degradamos explícitamente a transversal
        if not pmc_id:
            effective_query_type = "transversal_query"
            query_filter = Filter(
                must=[FieldCondition(key="metadata.source", match=MatchValue(value="pmc"))]
            )
        else:
            query_filter = Filter(
                must=[
                    FieldCondition(key="metadata.source", match=MatchValue(value="pmc")),
                    FieldCondition(key="metadata.pmc_id", match=MatchValue(value=pmc_id)),
                ]
            )

    else:  # transversal_query
        query_filter = Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value="pmc"))]
        )

    k_map = {
        "thematic_summary": RETRIEVAL_K_THEMATIC,
        "specific_query": RETRIEVAL_K_SPECIFIC,
        "transversal_query": RETRIEVAL_K_TRANSVERSAL,
    }

    k = k_map.get(effective_query_type, RETRIEVAL_K_SPECIFIC)

    if effective_query_type == "transversal_query":
        docs = await qdrant_store.amax_marginal_relevance_search(
            question, k=k, fetch_k=k * 3, filter=query_filter
        )
    else:
        docs = await qdrant_store.asimilarity_search(question, k=k, filter=query_filter)
    
    # print(f"[DEBUG] query_type={effective_query_type}, k={k}, docs={len(docs)}")
    return docs 


def format_docs(input_dict) -> str:
    docs = input_dict["source_context"]
    if not docs:
        return "No se encontraron documentos relevantes en la base de datos."

    parts = []
    for i, doc in enumerate(docs, 1):
        meta    = doc.metadata
        title   = meta.get("title", "Artículo sin título")
        section = meta.get("section", "")
        header  = f"[{i}] [{title} — {section}]" if section else f"[{i}] [{title}]"

        # page_content tiene formato "título | sección | texto" (PMC)
        # o "título | abstract" (PubMed) — quitamos el prefijo redundante
        content = doc.page_content
        if " | " in content:
            # Eliminar hasta el último " | " para quedarnos solo con el texto útil
            content = content.split(" | ", 2)[-1]

        parts.append(f"{header}\n{content}")

    return "\n\n---\n\n".join(parts)


# -------------- BRANCHING --------------

def is_in_scope(input_dict) -> bool:
    return input_dict["source"].query_type != "none"


answer_chain   = rag_prompt          | llm_langchain | StrOutputParser()
no_scope_chain = out_of_scope_prompt | llm_langchain | StrOutputParser()


# -------------- PIPELINE COMPLETO --------------

rag_chain = (
    # Paso 1: clasificar la pregunta
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
    # Paso 3: generar respuesta según si está en scope o no
    | RunnableBranch(
        (is_in_scope, RunnablePassthrough.assign(answer=answer_chain)),
        RunnablePassthrough.assign(answer=no_scope_chain),
    )
)
