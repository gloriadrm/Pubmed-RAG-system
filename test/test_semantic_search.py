"""
test_semantic_search.py
-----------------------
Valida los tres casos de uso del retrieval:

  Caso 1 — Resumen temático
    "Dame los últimos papers sobre microbiota e inflamación intestinal"
    → Busca entre abstracts de PubMed (source=pubmed)
    → Devuelve 1 point por artículo, ordenados por relevancia

  Caso 2 — Consulta sobre un paper concreto
    "¿Qué metodología usaron en este estudio?"
    → Busca dentro de los chunks PMC de UN artículo específico (source=pmc + pmc_id)
    → Devuelve las secciones más relevantes de ese paper

  Caso 3 — Consulta transversal entre papers
    "¿Qué evidencia hay sobre el efecto de la dieta en la permeabilidad intestinal?"
    → Busca entre todos los chunks PMC (source=pmc)
    → Puede recuperar secciones de distintos artículos

Ejecuta búsquedas reales (embeddings + Qdrant) contra la colección activa —
por eso todo el código va dentro de main(), guardado tras
if __name__ == "__main__": una simple importación (p. ej. al recolectar
tests con pytest sobre todo el directorio test/) no debe disparar consultas.

Uso:
    python test/test_semantic_search.py
"""

import sys
from pathlib import Path
from collections import Counter

# Ver test_actualizations.py: insertar al principio, no anexar al final,
# para no resolver src.* contra la copia instalada en site-packages.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

from src.ingest.indexer import search, client, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue


# ── Helper: obtener un pmc_id indexado ──────────────────────────────────────
def get_any_pmc_id() -> str | None:
    """Recupera el pmc_id de cualquier chunk PMC ya indexado."""
    results = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="source", match=MatchValue(value="pmc"))]
        ),
        limit=1,
        with_payload=True,
    )
    points = results[0]
    if points:
        return points[0].payload.get("pmc_id")
    return None


def main():
    load_dotenv()

    # ── CASO 1: Resumen temático ─────────────────────────────────────────────
    print("=" * 60)
    print("CASO 1 — Resumen temático (source=pubmed)")
    print("Query: artículos sobre microbiota y permeabilidad intestinal")
    print("=" * 60)
    search(
        query  = "papers about gut microbiota and intestinal permeability in inflammation",
        source = "pubmed",
        limit  = 5,
    )

    # ── CASO 2: Consulta sobre un paper concreto ─────────────────────────────
    print("=" * 60)
    print("CASO 2 — Paper concreto (source=pmc + pmc_id)")
    print("Query: metodología del estudio")
    print("=" * 60)

    pmc_id = get_any_pmc_id()
    if pmc_id:
        print(f"Artículo seleccionado: {pmc_id}\n")
        search(
            query  = "what methods and study design were used in this research",
            source = "pmc",
            pmc_id = pmc_id,
            limit  = 3,
        )
    else:
        print("⚠ No hay chunks PMC indexados. Ejecuta test_pmc_indexing.py primero.")

    # ── CASO 3: Consulta transversal entre papers ────────────────────────────
    print("=" * 60)
    print("CASO 3 — Transversal entre papers (source=pmc)")
    print("Query: evidencia sobre dieta y permeabilidad intestinal")
    print("=" * 60)
    search(
        query  = "evidence of diet effect on intestinal permeability and gut barrier",
        source = "pmc",
        limit  = 5,
    )

    # ── Diagnóstico: secciones más frecuentes ────────────────────────────────
    print("=" * 60)
    print("DIAGNÓSTICO — Secciones más frecuentes en chunks PMC")
    print("=" * 60)

    all_sections = []
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="source", match=MatchValue(value="pmc"))]
            ),
            limit=250,
            offset=offset,
            with_payload=["section"],
        )
        all_sections.extend(r.payload.get("section", "?") for r in results)
        if offset is None:
            break

    for section, count in Counter(all_sections).most_common(15):
        print(f"  {count:>4}x  {section}")


if __name__ == "__main__":
    main()
