"""
test_routing_retrieval.py
--------------------------
Tests unitarios de get_context() / is_in_scope() / clean_document_content()
(src/rag/chain.py) y de la validación de QueryRequest.pmc_id (src/api/schema.py).

No requieren Qdrant, LLM ni Docker: mockean qdrant_store (búsqueda semántica y
MMR) y el cliente Qdrant crudo (usado dentro de _resolve_pmc_id para el filtro
por autor) con unittest.mock (stdlib). El clasificador NO se invoca — cada test
construye a mano el SourceSelection que el router habría producido, ya que
get_context() lo recibe como dato de entrada y no llama al LLM él mismo.

Uso:
    python test/test_routing_retrieval.py
"""

import asyncio
import sys
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.documents import Document
from pydantic import ValidationError

from src.rag.chain import get_context, is_in_scope, clean_document_content
from src.rag.structures import SourceSelection, RetrievalResult
from src.api.schema import QueryRequest

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"[PASS] {name}")
    else:
        FAILED += 1
        print(f"[FAIL] {name}  {detail}")


def pmc_doc(pmc_id: str, title: str = "Título", section: str = "Methods", text: str = "texto recuperado") -> Document:
    return Document(
        page_content=f"{title} | {section} | {text}",
        metadata={"source": "pmc", "pmc_id": pmc_id, "pm_id": None, "title": title, "section": section},
    )


def pubmed_doc(pm_id: str = "111", pmc_id=None, title: str = "Título", abstract: str = "resumen") -> Document:
    # build_pubmed_article_dict() une título y abstract con un espacio, no " | "
    # (ver src/ingest/pubmed.py) — replicamos el formato real, no el asumido.
    return Document(
        page_content=f"{title} {abstract}",
        metadata={"source": "pubmed", "pm_id": pm_id, "pmc_id": pmc_id, "title": title},
    )


def source_sel(query_type="specific_query", pmc_id=None, author_last_name=None,
               author_name=None, paper_title=None, reason="test") -> SourceSelection:
    return SourceSelection(
        query_type=query_type, pmc_id=pmc_id, author_last_name=author_last_name,
        author_name=author_name, paper_title=paper_title, reason=reason,
    )


async def _run_get_context(input_dict, qdrant_store_mock, qdrant_client_scroll_result=([], None)):
    with patch("src.rag.chain.qdrant_store", qdrant_store_mock), \
         patch("src.ingest.indexer.client") as mock_indexer_client:
        mock_indexer_client.scroll.return_value = qdrant_client_scroll_result
        return await get_context(input_dict)


def new_store_mock(similarity_docs=None, mmr_docs=None):
    mock = MagicMock()
    mock.asimilarity_search = AsyncMock(return_value=similarity_docs or [])
    mock.amax_marginal_relevance_search = AsyncMock(return_value=mmr_docs or [])
    return mock


# ── 1. request_pmc_id tiene prioridad absoluta, incluso sobre query_type='none' ──

async def test_request_pmc_id_forces_specific_over_none():
    store = new_store_mock(similarity_docs=[pmc_doc("PMC111")])
    input_dict = {
        "question": "¿Qué hicieron aquí?",
        "source": source_sel(query_type="none", reason="Pregunta aislada sin contexto biomédico claro"),
        "request_pmc_id": "PMC111",
    }
    result = await _run_get_context(input_dict, store)

    check("retrieval_strategy == specific_query (no 'none')", result.retrieval_strategy == "specific_query", result.retrieval_strategy)
    check("resolved_pmc_id == PMC111", result.resolved_pmc_id == "PMC111", result.resolved_pmc_id)
    check("docs recuperados", len(result.docs) == 1)
    check("retrieval_note menciona la prioridad sobre el router", "priorizó" in (result.retrieval_note or ""), result.retrieval_note)
    check("is_in_scope() == True pese a query_type='none'", is_in_scope({"source_context": result}) is True)
    store.asimilarity_search.assert_awaited_once()


async def test_request_pmc_id_no_resolve_pmc_id_called():
    """Con request_pmc_id presente, _resolve_pmc_id (scroll por autor) no debe ejecutarse."""
    store = new_store_mock(similarity_docs=[pmc_doc("PMC222")])
    input_dict = {
        "question": "test",
        "source": source_sel(query_type="specific_query", author_last_name="Smith"),
        "request_pmc_id": "PMC222",
    }
    with patch("src.rag.chain.qdrant_store", store), \
         patch("src.ingest.indexer.client") as mock_indexer_client:
        result = await get_context(input_dict)
        check("scroll (resolución por autor) NO se llamó", mock_indexer_client.scroll.call_count == 0)
    check("retrieval_strategy == specific_query", result.retrieval_strategy == "specific_query")
    check("sin retrieval_note (query_type ya era specific_query)", result.retrieval_note is None, result.retrieval_note)


# ── 2. pmc_id extraído por el clasificador del texto de la pregunta ──

async def test_classifier_extracted_pmc_id():
    store = new_store_mock(similarity_docs=[pmc_doc("PMC333")])
    input_dict = {
        "question": "What were the findings of PMC333?",
        "source": source_sel(query_type="specific_query", pmc_id="PMC333"),
        "request_pmc_id": None,
    }
    result = await _run_get_context(input_dict, store)
    check("retrieval_strategy == specific_query", result.retrieval_strategy == "specific_query")
    check("resolved_pmc_id == PMC333", result.resolved_pmc_id == "PMC333")


# ── 3. specific_query resuelto por autor ──

async def test_specific_query_resolved_by_author():
    store = new_store_mock(similarity_docs=[pmc_doc("PMC444")])
    scroll_result = (
        [MagicMock(payload={"metadata": {"pmc_id": "PMC444"}})],
        None,
    )
    input_dict = {
        "question": "What did Hosseindoust find?",
        "source": source_sel(query_type="specific_query", author_last_name="Hosseindoust"),
        "request_pmc_id": None,
    }
    result = await _run_get_context(input_dict, store, qdrant_client_scroll_result=scroll_result)
    check("retrieval_strategy == specific_query (resuelto por autor)", result.retrieval_strategy == "specific_query")
    check("resolved_pmc_id == PMC444", result.resolved_pmc_id == "PMC444")


# ── 4. specific_query sin nada resoluble -> degradación a transversal ──

async def test_specific_query_degrades_to_transversal():
    store = new_store_mock(mmr_docs=[pmc_doc("PMC555"), pmc_doc("PMC666")])
    empty_scroll = ([], None)
    input_dict = {
        "question": "What methodology was used in this study?",
        "source": source_sel(query_type="specific_query"),  # sin pmc_id, sin autor, sin título
        "request_pmc_id": None,
    }
    result = await _run_get_context(input_dict, store, qdrant_client_scroll_result=empty_scroll)
    check("retrieval_strategy == transversal_query (degradado)", result.retrieval_strategy == "transversal_query", result.retrieval_strategy)
    check("resolved_pmc_id is None", result.resolved_pmc_id is None)
    check("retrieval_note explica la degradación", "no se pudo identificar" in (result.retrieval_note or "").lower(), result.retrieval_note)
    check("docs vienen de la búsqueda transversal (MMR)", len(result.docs) == 2)
    store.amax_marginal_relevance_search.assert_awaited_once()


# ── 5. pmc_id resuelto pero sin chunks PMC indexados (pmc_id "colgante") ──

async def test_specific_query_pmc_id_without_chunks():
    store = new_store_mock(similarity_docs=[])  # el filtro pmc_id no matchea ningún chunk
    input_dict = {
        "question": "¿Qué metodología utilizó el estudio?",
        "source": source_sel(query_type="specific_query", pmc_id="PMC777"),
        "request_pmc_id": None,
    }
    result = await _run_get_context(input_dict, store)
    check("retrieval_strategy SIGUE siendo specific_query (no degrada)", result.retrieval_strategy == "specific_query", result.retrieval_strategy)
    check("resolved_pmc_id == PMC777", result.resolved_pmc_id == "PMC777")
    check("sources vacío", result.docs == [])
    check("retrieval_note indica que no está indexado", "no está indexado" in (result.retrieval_note or ""), result.retrieval_note)


# ── 6. thematic_summary ──

async def test_thematic_summary():
    store = new_store_mock(similarity_docs=[pubmed_doc("111"), pubmed_doc("222")])
    input_dict = {
        "question": "Dame los últimos papers sobre microbiota e inflamación",
        "source": source_sel(query_type="thematic_summary"),
        "request_pmc_id": None,
    }
    result = await _run_get_context(input_dict, store)
    check("retrieval_strategy == thematic_summary", result.retrieval_strategy == "thematic_summary")
    check("2 docs pubmed", len(result.docs) == 2)
    check("sin retrieval_note (hay resultados)", result.retrieval_note is None)


# ── 7. transversal_query ──

async def test_transversal_query():
    store = new_store_mock(mmr_docs=[pmc_doc("PMC1"), pmc_doc("PMC2")])
    input_dict = {
        "question": "¿Qué mecanismos relacionan la disbiosis con la inflamación?",
        "source": source_sel(query_type="transversal_query"),
        "request_pmc_id": None,
    }
    result = await _run_get_context(input_dict, store)
    check("retrieval_strategy == transversal_query", result.retrieval_strategy == "transversal_query")
    check("2 docs pmc", len(result.docs) == 2)


# ── 8. none (sin request_pmc_id) -> corto-circuito total, sin tocar Qdrant ──

async def test_none_short_circuits():
    store = new_store_mock()
    input_dict = {
        "question": "¿Cómo se forman los agujeros negros?",
        "source": source_sel(query_type="none", reason="Fuera del dominio biomédico"),
        "request_pmc_id": None,
    }
    result = await _run_get_context(input_dict, store)
    check("retrieval_strategy == none", result.retrieval_strategy == "none")
    check("docs vacío", result.docs == [])
    check("no se llamó a asimilarity_search", store.asimilarity_search.call_count == 0)
    check("no se llamó a amax_marginal_relevance_search", store.amax_marginal_relevance_search.call_count == 0)
    check("is_in_scope() == False", is_in_scope({"source_context": result}) is False)


# ── 9. clean_document_content / excerpt ──

def test_clean_document_content():
    pmc = pmc_doc("PMC1", title="T", section="Methods", text="el fragmento real")
    check("PMC: 'título | sección | texto' -> 'texto'", clean_document_content(pmc) == "el fragmento real", clean_document_content(pmc))

    pubmed = pubmed_doc(title="T", abstract="el abstract real")
    check("PubMed: 'título abstract' (sin ' | ') -> 'abstract' vía metadata.title", clean_document_content(pubmed) == "el abstract real", clean_document_content(pubmed))

    pubmed_no_meta = Document(page_content="Título Y Resto sin separador de pipe", metadata={})
    check(
        "PubMed sin metadata.title -> texto completo tal cual (fallback seguro)",
        clean_document_content(pubmed_no_meta) == "Título Y Resto sin separador de pipe",
        clean_document_content(pubmed_no_meta),
    )

    plain = Document(page_content="  sin separador  ", metadata={})
    check("sin ' | ' -> texto tal cual (trimmed)", clean_document_content(plain) == "sin separador", clean_document_content(plain))


# ── 10. QueryRequest.pmc_id — normalización y rechazo ──

def test_query_request_pmc_id_validation():
    check("None se acepta tal cual", QueryRequest(question="q", pmc_id=None).pmc_id is None)

    r1 = QueryRequest(question="q", pmc_id="12345678")
    check("'12345678' -> 'PMC12345678'", r1.pmc_id == "PMC12345678", r1.pmc_id)

    r2 = QueryRequest(question="q", pmc_id="pmc12345678")
    check("'pmc12345678' -> 'PMC12345678'", r2.pmc_id == "PMC12345678", r2.pmc_id)

    r3 = QueryRequest(question="q", pmc_id="PMC12345678")
    check("'PMC12345678' se mantiene", r3.pmc_id == "PMC12345678", r3.pmc_id)

    for invalid in ["PMC12ABC", "artículo-123", "PMC"]:
        try:
            QueryRequest(question="q", pmc_id=invalid)
            check(f"'{invalid}' debería rechazarse", False, "no lanzó ValidationError")
        except ValidationError:
            check(f"'{invalid}' rechazado con ValidationError (422 en FastAPI)", True)


def test_question_reaches_classifier_unmodified():
    """
    Regresión: request.pmc_id ya no debe inyectarse como texto "(PMC ID: ...)"
    en la pregunta. main.py construye chain_input = {"question": ..., "request_pmc_id": ...}
    sin tocar el texto — lo comprobamos aquí inspeccionando el propio código fuente,
    ya que main.py requiere un stack Docker completo para invocarse end-to-end.
    """
    import inspect
    from src.api import main as main_module
    source = inspect.getsource(main_module.query)
    check(
        "main.query() no contiene la inyección de texto '(PMC ID:'",
        "(PMC ID:" not in source,
        "todavía aparece la inyección de texto en main.py",
    )
    check(
        "main.query() pasa request_pmc_id como campo estructurado",
        "request_pmc_id" in source and "request.pmc_id" in source,
    )


def run_async(coro_fn):
    asyncio.run(coro_fn())


def main():
    print("=" * 60)
    print("  TEST — Routing / Retrieval (get_context, is_in_scope, excerpt)")
    print("=" * 60)
    print()

    run_async(test_request_pmc_id_forces_specific_over_none)
    run_async(test_request_pmc_id_no_resolve_pmc_id_called)
    run_async(test_classifier_extracted_pmc_id)
    run_async(test_specific_query_resolved_by_author)
    run_async(test_specific_query_degrades_to_transversal)
    run_async(test_specific_query_pmc_id_without_chunks)
    run_async(test_thematic_summary)
    run_async(test_transversal_query)
    run_async(test_none_short_circuits)
    test_clean_document_content()
    test_query_request_pmc_id_validation()
    test_question_reaches_classifier_unmodified()

    print()
    print("=" * 60)
    print(f"  {PASSED} pasadas, {FAILED} fallidas")
    print("=" * 60)

    if FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()
