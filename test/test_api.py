"""
test_api.py
-----------
Verifica el pipeline RAG completo a través de la API:
  - Router (clasificación en 4 tipos)
  - Retrieval (3 casos de búsqueda)
  - Respuesta generada por el LLM

Requiere la API corriendo:
    uvicorn src.api.main:app --reload --port 8000

Uso:
    python test/test_api.py
    python test/test_api.py --url http://localhost:8000   # URL personalizada
"""

import sys
import argparse
import json
import requests

DEFAULT_URL = "http://localhost:8000"


def check_health(base_url: str):
    r = requests.get(f"{base_url}/health", timeout=5)
    data = r.json()
    print(f"  qdrant             : {data['qdrant']}")
    print(f"  embedding_provider : {data['embedding_provider']}")
    print(f"  embedding_status   : {data['embedding_status']}")
    print(f"  llm_provider       : {data['llm_provider']}")
    print(f"  llm_status         : {data['llm_status']}")
    print(f"  status             : {data['status']}")
    if data["status"] != "ok":
        print("⚠ Algún servicio no está disponible. Continúa de todas formas...\n")
    return data["status"] == "ok"


def query(base_url: str, question: str, pmc_id: str = None) -> dict:
    payload = {"question": question}
    if pmc_id:
        payload["pmc_id"] = pmc_id
    r = requests.post(f"{base_url}/query", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def print_result(label: str, result: dict):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Query type : {result['query_type']}")
    print(f"  Razón      : {result['reason']}")
    if result.get("pmc_id"):
        print(f"  PMC ID     : {result['pmc_id']}")
    print(f"\n  Fuentes recuperadas ({len(result['sources'])}):")
    for s in result["sources"]:
        section = f" — {s['section']}" if s.get("section") else ""
        print(f"    [{s['source'].upper()}] {s['title'][:60]}{section}")
    print(f"\n  Respuesta:")
    print(f"  {result['answer'][:800]}")
    if len(result["answer"]) > 800:
        print("  [... respuesta truncada ...]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    base = args.url.rstrip("/")

    print("="*60)
    print("  TEST END-TO-END — Biomedical RAG API")
    print("="*60)

    # Health check
    print("\n[HEALTH CHECK]")
    check_health(base)

    # ── CASO 1: thematic_summary ──────────────────────────────────
    result = query(
        base,
        question="¿What is the current evidence on gut microbiota and intestinal permeability?"
    )
    print_result("CASO 1 — Resumen temático (esperado: thematic_summary → pubmed)", result)

    # ── CASO 2a: specific_query con PMC ID directo ────────────────
    # Cogemos el pmc_id del primer artículo PMC indexado
    scroll_r = requests.post(f"{base}/query", json={
        "question": "In the paper The role of gut microbiota in autoimmune thyroid diseases: nutritional determinants and diet-based modulation, what clinical recommendations are suggested?"
    }, timeout=120)
    result2a = scroll_r.json()
    print_result("CASO 2a — Paper concreto por tema (esperado: specific_query → pmc)", result2a)

    # ── CASO 2b: specific_query por autor ─────────────────────────
    result2b = query(
        base,
        question="What were the main findings of the study by Patel on dietary interventions and gut microbiome in PCOS?"
    )
    print_result("CASO 2b — Paper concreto por autor (esperado: specific_query → pmc)", result2b)

    # ── CASO 3: transversal_query ─────────────────────────────────
    result3 = query(
        base,
        question="¿What mechanisms link gut dysbiosis to intestinal permeability across the indexed studies?"
    )
    print_result("CASO 3 — Transversal (esperado: transversal_query → pmc)", result3)

    # ── OUT OF SCOPE ──────────────────────────────────────────────
    result_oos = query(base, question="What is the capital of France?")
    print_result("OUT OF SCOPE (esperado: none)", result_oos)

    print(f"\n{'='*60}")
    print("  Test completado.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
