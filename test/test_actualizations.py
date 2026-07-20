"""
test_actualizations.py
-----------------------
Pipeline unificado de actualización incremental.
Ejecuta en orden:

  1. Actualización PubMed via update files diarios de NLM
  2. Actualización PMC — solo para los pmc_ids ya indexados, consultando su
     metadata OA individual (bucket público de NCBI en AWS). Sin CSV, sin
     diff de catálogo global: oa_file_list.csv fue retirado por NCBI en
     2026 (ver src/data_actualization/pmc_oa_client.py).

Uso (desde el root del proyecto):
    python test/test_actualizations.py

Flags disponibles:
    --pubmed-only   Solo ejecuta la actualización de PubMed
    --pmc-only      Solo ejecuta la actualización de PMC
    --dry-run       Calcula los cambios pero no modifica Qdrant
"""

import sys
from pathlib import Path

# Invocado como "python test/test_actualizations.py" (script, no módulo),
# sys.path[0] es el directorio del script (test/), no la raíz del proyecto.
# sys.path.append('.') no basta: se añade al FINAL, después del paquete
# instalado en site-packages (build time, potencialmente obsoleto) — el
# import de src.* resolvía silenciosamente contra esa copia congelada en
# vez de contra estos mismos archivos. Insertar al principio garantiza que
# siempre se usa el código real de este checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from src.ingest.indexer import client, COLLECTION_NAME
from qdrant_client.models import Filter, FieldCondition, MatchValue


def count_points(source: str) -> int:
    return client.count(
        collection_name=COLLECTION_NAME,
        count_filter=Filter(
            must=[FieldCondition(key="metadata.source", match=MatchValue(value=source))]
        )
    ).count


def run_pubmed_update(dry_run: bool = False):
    """Actualización incremental de artículos PubMed."""
    print("\n" + "="*60)
    print("BLOQUE 1 — Actualización PubMed (update files NLM)")
    print("="*60)

    from src.data_actualization.pubmed_updater import (
        list_pubmed_update_files,
        get_last_processed_file,
        get_pending_files,
        parse_update_file,
        get_indexed_pmids,
        apply_pubmed_updates,
        save_last_processed_file,
    )

    all_files      = list_pubmed_update_files()
    last_processed = get_last_processed_file()

    print(f"Update files disponibles : {len(all_files)}")
    print(f"Último procesado         : {last_processed or 'ninguno (primera ejecución)'}")

    pending = get_pending_files(all_files, last_processed)
    print(f"Ficheros pendientes      : {len(pending)}")

    if not pending:
        print("→ Nada que actualizar en PubMed.")
        return

    indexed_pmids = get_indexed_pmids()
    print(f"PMIDs indexados en Qdrant: {len(indexed_pmids)}")

    for file_info in pending:
        filename = file_info["filename"]
        print(f"\nFichero: {filename}")
        articles, deleted_pmids = parse_update_file(file_info["url"])

        to_upsert = [a for a in articles if str(a.get("pm_id", "")) in indexed_pmids]
        to_delete = [p for p in deleted_pmids if p in indexed_pmids]

        print(f"  Artículos a actualizar : {len(to_upsert)}")
        print(f"  PMIDs a borrar         : {len(to_delete)}")

        if dry_run:
            print("  [dry-run] Cambios calculados, Qdrant no modificado.")
        else:
            # apply_pubmed_updates devuelve los PMIDs efectivamente borrados —
            # hay que restarlos de indexed_pmids aquí mismo: si no, un PMID
            # borrado en este fichero seguiría figurando como "indexado" para
            # los ficheros siguientes de esta misma ejecución, y una revisión
            # posterior del mismo PMID lo resucitaría.
            deleted = apply_pubmed_updates(articles, deleted_pmids, indexed_pmids)
            indexed_pmids -= deleted
            save_last_processed_file(filename)
            print(f"  ✓ {filename} aplicado.")


def run_pmc_update(dry_run: bool = False):
    """
    Actualización de chunks PMC — solo para los pmc_ids ya indexados,
    consultando su metadata OA individual (sin CSV, sin diff de catálogo
    global). Ver src/data_actualization/pmc_oa_client.py.
    """
    print("\n" + "="*60)
    print("BLOQUE 2 — Actualización PMC (metadata OA individual)")
    print("="*60)

    from src.data_actualization.oa_updater import (
        get_indexed_pmc_ids,
        check_indexed_pmc_articles,
        remove_pmc_fulltext,
        reingest_pmc_articles,
    )

    print("Cargando PMC IDs indexados en Qdrant...")
    indexed_pmc_ids = get_indexed_pmc_ids()
    print(f"  → {len(indexed_pmc_ids)} PMC IDs indexados")

    if not indexed_pmc_ids:
        print("→ No hay artículos PMC indexados. Nada que actualizar.")
        return

    removals, to_reingest = check_indexed_pmc_articles(indexed_pmc_ids)
    print(f"\nBajas: {len(removals)} ({', '.join(f'{k}: {v}' for k, v in removals.items())})" if removals else "\nBajas: 0")
    print(f"Cambios a reingerir: {len(to_reingest)}")

    if not removals and not to_reingest:
        print("→ Nada que actualizar en PMC.")
        return

    if dry_run:
        print("\n[dry-run] Cambios calculados, Qdrant no modificado.")
        return

    # Retirar PMC IDs que ya no cumplen requisitos: borra sus chunks Y limpia
    # pmc_id/license/pmc_last_updated del punto PubMed asociado (conservando
    # el propio documento PubMed).
    if removals:
        print(f"\nRetirando {len(removals)} artículos PMC de Qdrant...")
        remove_pmc_fulltext(list(removals.keys()))

    # Reingestar artículos cuya metadata OA cambió
    if to_reingest:
        reingest_pmc_articles(to_reingest)


def main():
    parser = argparse.ArgumentParser(description="Pipeline de actualización incremental")
    parser.add_argument("--pubmed-only", action="store_true", help="Solo actualiza PubMed")
    parser.add_argument("--pmc-only",    action="store_true", help="Solo actualiza PMC")
    parser.add_argument("--dry-run",     action="store_true", help="Calcula cambios sin modificar Qdrant")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("PIPELINE DE ACTUALIZACIÓN INCREMENTAL")
    print("="*60)

    before_pubmed = count_points("pubmed")
    before_pmc    = count_points("pmc")
    print(f"\nEstado inicial  → pubmed: {before_pubmed} | pmc: {before_pmc}")

    run_pubmed = not args.pmc_only
    run_pmc    = not args.pubmed_only

    if run_pubmed:
        run_pubmed_update(dry_run=args.dry_run)

    if run_pmc:
        run_pmc_update(dry_run=args.dry_run)

    after_pubmed = count_points("pubmed")
    after_pmc    = count_points("pmc")
    print(f"\nEstado final    → pubmed: {after_pubmed} | pmc: {after_pmc}")

    delta_pubmed = after_pubmed - before_pubmed
    delta_pmc    = after_pmc    - before_pmc
    print(f"Diferencia      → pubmed: {delta_pubmed:+d} | pmc: {delta_pmc:+d}")

    if not args.dry_run:
        print("\n✓ Pipeline de actualización completado.")


if __name__ == "__main__":
    main()
