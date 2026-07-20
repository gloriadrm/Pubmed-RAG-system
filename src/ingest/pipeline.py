"""
Pipeline principal de ingesta. Orquesta:
  PubMed (title + abstract) → OA lookup → PMC full text (si existe)

El OA lookup (PMID → PMCID → licencia) se resuelve vía
src/data_actualization/pmc_oa_client.py (bucket público de NCBI en AWS),
no vía CSV local — ver oa_updater.get_oa_df()/lookup_oa_csv() para la
interfaz que usa este módulo.

Uso:
    from src.ingest.pipeline import ingest_query
    from src.data_actualization.oa_updater import get_oa_df
    oa_df = get_oa_df()
    summary = ingest_query("gut microbiota inflammation", n=10, oa_df=oa_df)
"""

import pandas as pd

from src.ingest.pubmed import build_pubmed_article_dicts
from src.ingest.pmc import fetch_pmc_full_text, parse_pmc_xml_to_chunks, enrich_with_oa_metadata
from src.ingest.indexer import upsert_pubmed_articles, upsert_pmc_chunks
from src.data_actualization.oa_updater import lookup_oa_csv

from src.config import ALLOWED_PMC_LICENSES


def ingest_query(query: str, n: int, oa_df: pd.DataFrame) -> dict:
    """
    Pipeline completo de ingesta para una query.

    Pasos:
      1. Busca N artículos en PubMed → 1 point por artículo (title + abstract)
      2. Cruza PMIDs con el OA lookup para encontrar full texts en PMC
      3. Enriquece artículos PubMed con pmc_id y license
      4. Indexa los PubMed points
      5. Para los artículos con PMC, descarga full text y genera chunks
      6. Enriquece chunks con last_updated, article-type y license del OA lookup
      7. Indexa los PMC chunks

    Args:
        query:  término de búsqueda
        n:      número de artículos a recuperar de PubMed
        oa_df:  DataFrame de mapeo PMID→PMCID (get_oa_df())

    Returns:
        Resumen estructurado de la ejecución (usado por /ingest para
        construir IngestResponse) — cada paso además se registra en
        "log" como texto legible, para mostrar en un desplegable.
    """
    log: list[str] = []

    def _log(msg: str):
        print(msg)
        log.append(msg)

    def _summary(**overrides) -> dict:
        base = {
            "articles_found":       0,
            "pubmed_indexed":       0,
            "pmc_found":            0,
            "pmc_excluded_license": 0,
            "xml_downloaded":       0,
            "chunks_created":       0,
        }
        base.update(overrides)
        base["log"] = log
        return base

    # Paso 1: Pipeline PubMed
    # ESearch > EFetch > parse XML > construir diccionarios > list[dict]
    _log(f"[1/4] Buscando {n} artículos en PubMed: '{query}'")
    result   = build_pubmed_article_dicts(query, n)
    articles = result["articles"]

    if not articles:
        _log("  → No se encontraron artículos.")
        return _summary()

    _log(f"  → {len(articles)} artículos encontrados.")

    # Paso 2: OA lookup
    # Extrae los pm_ids que se encuentren en el catálogo OA de PMC
    _log("[2/4] Cruzando PMIDs con PMC Open Access...")
    pmids      = [a["pm_id"] for a in articles]
    oa_matches = lookup_oa_csv(pmids, oa_df)
    _log(f"  → {len(oa_matches)} artículos tienen full text en PMC.")

    # Enriquecer artículos PubMed con pmc_id, license y last_updated procedente del OA lookup
    for article in articles:
        oa = oa_matches.get(str(article["pm_id"]), {})
        article["pmc_id"]                    = oa.get("pmc_id")
        article["metadata"]["license"]           = oa.get("license")
        article["metadata"]["pmc_last_updated"]  = oa.get("last_updated")

    # Paso 3: Indexar PubMed points
    _log("[3/4] Indexando artículos PubMed...")
    upsert_pubmed_articles(articles)
    _log(f"  → {len(articles)} artículos PubMed indexados.")

    # Paso 4: PMC full text — solo licencias que permiten reproducción
    pmc_articles = [
        a for a in articles
        if a.get("pmc_id") and a["metadata"].get("license") in ALLOWED_PMC_LICENSES
    ]
    excluded = [
        a for a in articles
        if a.get("pmc_id") and a["metadata"].get("license") not in ALLOWED_PMC_LICENSES
    ]
    if excluded:
        _log(f"  ⚠ {len(excluded)} artículos excluidos por licencia restrictiva "
             f"({', '.join(set(a['metadata'].get('license','?') for a in excluded))})")

    if not pmc_articles:
        _log("[4/4] Ningún artículo con full text descargable. Pipeline completado.")
        return _summary(
            articles_found=len(articles),
            pubmed_indexed=len(articles),
            pmc_found=len(oa_matches),
            pmc_excluded_license=len(excluded),
        )

    pmc_ids = [a["pmc_id"] for a in pmc_articles]
    _log(f"[4/4] Descargando y chunking full texts de {len(pmc_ids)} artículos PMC...")
    xml_text = fetch_pmc_full_text(pmc_ids)
    chunks   = parse_pmc_xml_to_chunks(xml_text)
    _log(f"  → {len(chunks)} chunks generados.")

    # Enriquecer chunks con last_updated y license del OA lookup
    # Construye un índice por pmc_id
    oa_index = {v["pmc_id"]: v for v in oa_matches.values()}
    chunks   = enrich_with_oa_metadata(chunks, oa_index)

    upsert_pmc_chunks(chunks)
    _log(f"  → {len(chunks)} chunks PMC indexados.")
    _log("Pipeline completado.")

    return _summary(
        articles_found=len(articles),
        pubmed_indexed=len(articles),
        pmc_found=len(oa_matches),
        pmc_excluded_license=len(excluded),
        xml_downloaded=len(pmc_ids),
        chunks_created=len(chunks),
    )
