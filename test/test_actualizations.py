"""
test_actualizations.py
-----------------------
Pipeline unificado de actualización incremental.
Ejecuta en orden:

  1. Actualización PubMed via update files diarios de NLM
  2. Actualización PMC via diff del OA file list

Uso (desde el root del proyecto):
    python test/test_actualizations.py

Flags disponibles:
    --pubmed-only   Solo ejecuta la actualización de PubMed
    --pmc-only      Solo ejecuta la actualización de PMC
    --dry-run       Calcula los cambios pero no modifica Qdrant
"""

import sys
sys.path.append('.')

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


def get_indexed_pmc_ids() -> set[str]:
    """Devuelve el conjunto de pmc_ids indexados en Qdrant (source=pmc)."""
    indexed = set()
    offset  = None
    while True:
        results, next_offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="metadata.source", match=MatchValue(value="pmc"))]
            ),
            limit=1000,
            offset=offset,
            with_payload=["metadata"],
            with_vectors=False,
        )
        for point in results:
            pmc_id = point.payload.get("metadata", {}).get("pmc_id")
            if pmc_id:
                indexed.add(str(pmc_id))
        if next_offset is None:
            break
        offset = next_offset
    return indexed


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
            apply_pubmed_updates(articles, deleted_pmids, indexed_pmids)
            save_last_processed_file(filename)
            print(f"  ✓ {filename} aplicado.")


def run_pmc_update(dry_run: bool = False):
    """Actualización incremental de chunks PMC via diff del OA file list."""
    print("\n" + "="*60)
    print("BLOQUE 2 — Actualización PMC (OA file list diff)")
    print("="*60)

    from src.data_actualization.oa_updater import (
        get_oa_df,
        download_oa_csv,
        load_oa_csv,
        diff_oa_csv,
        detect_pmc_removals,
        reingest_changed_pmc,
        LOCAL_OA,
        DATA_DIR,
    )
    import shutil
    from pathlib import Path

    # Cargar OA CSV 
    # (Primera ejecución: guardado en disco sin hacer diff al no haber versión anterior)
    if not LOCAL_OA.exists():
        print("→ No hay OA CSV local previo. Descargando por primera vez...")
        get_oa_df(force_download=True)
        print("→ OA CSV descargado. Nada más que actualizar en esta ejecución.")
        return

    print("Cargando OA CSV anterior...")
    old_df = load_oa_csv(LOCAL_OA)
    print(f"  → {len(old_df):,} entradas en OA CSV anterior")

    # Descargar versión nueva a un temporal
    tmp_path = LOCAL_OA.with_suffix(".new.csv")
    print("Descargando OA file list actualizado (~100 MB)...")
    from src.data_actualization.oa_updater import download_oa_csv, load_oa_csv
    new_df = download_oa_csv(tmp_path)
    print(f"  → {len(new_df):,} entradas en OA CSV nuevo")

    # Cruce con Qdrant: solo nos interesan los pmc_ids que ya tenemos indexados
    print("Cargando PMC IDs indexados en Qdrant...")
    indexed_pmc_ids = get_indexed_pmc_ids()
    print(f"  → {len(indexed_pmc_ids)} PMC IDs indexados")

    # --- Bajas: PMC IDs desaparecidos del catálogo o con licencia no permitida ---
    removals_global = detect_pmc_removals(old_df, new_df)
    removals = [p for p in removals_global if p in indexed_pmc_ids]
    print(f"Bajas en OA CSV (global)    : {len(removals_global)}")
    print(f"Bajas en colección  : {len(removals)}")

    # --- Actualizaciones: last_updated cambiado ---
    changed_df = diff_oa_csv(old_df, new_df)
    changed_df = changed_df[changed_df["pmc_id"].isin(indexed_pmc_ids)].reset_index(drop=True)
    print(f"Actualizaciones en colección: {len(changed_df)}")

    if not removals and changed_df.empty:
        print("→ Nada que actualizar en PMC.")
        tmp_path.unlink(missing_ok=True)
        return

    if dry_run:
        print("\n[dry-run] Cambios calculados, Qdrant no modificado.")
        tmp_path.unlink(missing_ok=True)
        return

    # Borrar PMC IDs que ya no cumplen requisitos
    if removals:
        print(f"\nBorrando {len(removals)} artículos PMC de Qdrant...")
        for pmc_id in removals:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="metadata.source", match=MatchValue(value="pmc")),
                        FieldCondition(key="metadata.pmc_id", match=MatchValue(value=pmc_id)),
                    ]
                ),
            )
            print(f"  ✓ Borrado {pmc_id}")

    # Reingestar artículos con last_updated cambiado
    if not changed_df.empty:
        reingest_changed_pmc(changed_df)

    # Reemplazar CSV local con el nuevo
    shutil.move(str(tmp_path), str(LOCAL_OA))
    print("\n✓ OA CSV actualizado en disco.")


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
