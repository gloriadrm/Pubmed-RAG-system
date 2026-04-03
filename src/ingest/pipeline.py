"""
Pipeline principal de ingesta. Orquesta:
  PubMed (title + abstract) → OA lookup → PMC full text (si existe)

Uso:
    from src.ingest.pipeline import ingest_query
    import pandas as pd
    oa_df = pd.read_csv("data/oa_file_list.csv", ...)  # o usa get_oa_df()
    ingest_query("gut microbiota inflammation", n=10, oa_df=oa_df)
"""

import pandas as pd

from src.ingest.pubmed import build_pubmed_article_dicts
from src.ingest.pmc import fetch_pmc_full_text, parse_pmc_xml_to_chunks, enrich_with_oa_metadata
from src.ingest.indexer import upsert_pubmed_articles, upsert_pmc_chunks
from src.data_actualization.oa_updater import lookup_oa_csv

from src.config import ALLOWED_PMC_LICENSES


def ingest_query(query: str, n: int, oa_df: pd.DataFrame):
    """
    Pipeline completo de ingesta para una query.

    Pasos:
      1. Busca N artículos en PubMed → 1 point por artículo (title + abstract)
      2. Cruza PMIDs con el OA CSV para encontrar full texts en PMC
      3. Enriquece artículos PubMed con pmc_id y license
      4. Indexa los PubMed points
      5. Para los artículos con PMC, descarga full text y genera chunks
      6. Enriquece chunks con last_updated, article-type y license del OA CSV
      7. Indexa los PMC chunks

    Args:
        query:  término de búsqueda
        n:      número de artículos a recuperar de PubMed
        oa_df:  DataFrame del OA file list (cargado con load_oa_csv o get_oa_df)
    """

    # Paso 1: Pipeline PubMed
    # ESearch > EFetch > parse XML > construir diccionarios > list[dict]
    print(f"\n[1/4] Buscando {n} artículos en PubMed: '{query}'")
    result   = build_pubmed_article_dicts(query, n)
    articles = result["articles"]

    if not articles:
        print("  → No se encontraron artículos.")
        return

    print(f"  → {len(articles)} artículos encontrados.")

    # Paso 2: OA lookup 
    # Extrae los pm_ids que se enceuntren en el inventario OA de PMC 
    print("[2/4] Cruzando PMIDs con OA file list...")
    pmids      = [a["pm_id"] for a in articles]
    oa_matches = lookup_oa_csv(pmids, oa_df)
    print(f"  → {len(oa_matches)} artículos tienen full text en PMC.")

    # Enriquecer artículos PubMed con pmc_id, license y last_updated procedente del OA file list de PMC
    for article in articles:
        oa = oa_matches.get(str(article["pm_id"]), {})
        article["pmc_id"]                    = oa.get("pmc_id")
        article["metadata"]["license"]           = oa.get("license")
        article["metadata"]["pmc_last_updated"]  = oa.get("last_updated")

    # Paso 3: Indexar PubMed points 
    print("[3/4] Indexando artículos PubMed...")
    upsert_pubmed_articles(articles)

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
        print(f"  ⚠ {len(excluded)} artículos excluidos por licencia restrictiva "
              f"({', '.join(set(a['metadata'].get('license','?') for a in excluded))})")

    if not pmc_articles:
        print("[4/4] Ningún artículo con full text descargable. Pipeline completado.")
        return

    pmc_ids = [a["pmc_id"] for a in pmc_articles]
    print(f"[4/4] Descargando y chunking full texts de {len(pmc_ids)} artículos PMC...")
    xml_text = fetch_pmc_full_text(pmc_ids)
    chunks   = parse_pmc_xml_to_chunks(xml_text)
    print(f"  → {len(chunks)} chunks generados.")

    # Enriquecer chunks con last_updated y license del OA CSV
    # Construye un índice por pmc_id 
    oa_index = {v["pmc_id"]: v for v in oa_matches.values()}
    chunks   = enrich_with_oa_metadata(chunks, oa_index)

    upsert_pmc_chunks(chunks)
    print("Pipeline completado.")
