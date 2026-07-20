"""
oa_updater.py
-------------
Mantiene actualizado el texto completo PMC ya indexado:
  - Expone lookup_oa_csv() para que /ingest cruce PMIDs con PMCIDs
    (usado por src/ingest/pipeline.py, interfaz sin cambios)
  - Expone run_daily_update() para revisar el corpus ya indexado y aplicar
    bajas (retractados / licencia ya no permitida / ya no disponibles) o
    reingestas (contenido cambiado)

La obtención real de datos (mapeo PMID→PMCID, metadata de licencia y
retractación) vive en src/data_actualization/pmc_oa_client.py — sustituye a
oa_file_list.csv, retirado por NCBI en 2026. Este módulo no descarga ni
diferencia ningún catálogo global: solo consulta, artículo a artículo, los
pmc_ids que este sistema ya tiene indexados. El corpus nunca se amplía
desde aquí — eso es responsabilidad exclusiva de /ingest.

Puede ejecutarse manualmente o programarse con cron:
    0 3 * * * /path/to/.venv/bin/python -m src.data_actualization.oa_updater
"""

from dotenv import load_dotenv

load_dotenv()

import pandas as pd

from src.config import ALLOWED_PMC_LICENSES
from src.data_actualization.pmc_oa_client import (
    get_pmc_ids_df,
    resolve_pmcids,
    fetch_oa_metadata,
    OaMetadata,
    OaMetadataError,
)


# -------------- LOOKUP (usado por /ingest vía src/ingest/pipeline.py) --------------

def get_oa_df(force_download: bool = False) -> pd.DataFrame:
    """
    Mantiene el nombre y contrato históricos (DataFrame consumido por
    lookup_oa_csv) — internamente ya no es "todo el catálogo OA con
    licencia incluida" sino el mapeo bulk PMID→PMCID (PMC-ids.csv.gz, no
    deprecado). La licencia y el estado de retractación se resuelven por
    artículo, bajo demanda, dentro de lookup_oa_csv().
    """
    return get_pmc_ids_df(force_download=force_download)


def lookup_oa_csv(pmids: list[str], df: pd.DataFrame, licenses: list[str] = None) -> dict:
    """
    Cruza una lista de PMIDs con PMC: resuelve PMID→PMCID vía el mapeo bulk
    (df) y, para cada match, consulta su metadata OA individual (licencia,
    retractación).

    Args:
        pmids:    lista de PMIDs a cruzar
        df:       DataFrame de mapeo PMID→PMCID (get_oa_df())
        licenses: lista de licencias permitidas (None = todas).
                  Ej: ["CC BY", "CC BY-SA", "CC BY-NC"]

    Returns:
        {pm_id: {"pmc_id": "PMC13900", "last_updated": "<Last-Modified HTTP>", "license": "CC BY"}}
        Solo incluye PMIDs con PMCID resuelto, no retractados, y que pasan
        el filtro de licencia (si se especifica). Un fallo de red o una
        respuesta incompleta al consultar un artículo concreto lo excluye
        del resultado (se trata como "sin match" para esta ingesta, no
        como error fatal de toda la llamada).
    """
    resolved = resolve_pmcids(pmids, df)
    if not resolved:
        return {}

    result = {}
    for pm_id, pmc_id in resolved.items():
        try:
            meta = fetch_oa_metadata(pmc_id)
        except OaMetadataError:
            continue
        if meta is None or meta.is_retracted:
            continue
        if licenses is not None and meta.license_code not in licenses:
            continue
        result[pm_id] = {
            "pmc_id":       meta.pmcid,
            "last_updated": meta.last_modified,
            "license":      meta.license_code,
        }
    return result


# -------------- MANTENIMIENTO DEL CORPUS YA INDEXADO --------------

def get_indexed_pmc_ids() -> set[str]:
    """
    Devuelve el conjunto de pmc_ids actualmente indexados en Qdrant
    (source=pmc). El OA updater solo debe mantener/depurar este conjunto —
    nunca ampliarlo (eso es responsabilidad exclusiva de /ingest).
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from src.ingest.indexer import COLLECTION_NAME, QDRANT_URL

    client = QdrantClient(url=QDRANT_URL)
    indexed: set[str] = set()
    offset = None
    while True:
        results, offset = client.scroll(
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
        if offset is None:
            break
    return indexed


def _get_stored_pmc_last_updated(pmc_id: str) -> str | None:
    """pmc_last_updated tal como está guardado ahora mismo para este pmc_id
    (leído de cualquiera de sus chunks — todos comparten el mismo valor)."""
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from src.ingest.indexer import COLLECTION_NAME, QDRANT_URL

    client = QdrantClient(url=QDRANT_URL)
    existing, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="metadata.source", match=MatchValue(value="pmc")),
                FieldCondition(key="metadata.pmc_id", match=MatchValue(value=pmc_id)),
            ]
        ),
        limit=1,
        with_payload=["metadata"],
        with_vectors=False,
    )
    if not existing:
        return None
    return existing[0].payload.get("metadata", {}).get("pmc_last_updated")


def check_indexed_pmc_articles(indexed_pmc_ids: set[str]) -> tuple[dict[str, str], dict[str, OaMetadata]]:
    """
    Para cada pmc_id ya indexado, consulta su metadata OA actual y decide:
      - baja (removals): el artículo ya no está en el PMC Open Access
        Subset (404 confirmado), ha sido retractado, o su licencia ya no
        es una de ALLOWED_PMC_LICENSES.
      - reingesta (to_reingest): la licencia sigue siendo válida pero
        Last-Modified difiere de lo que tenemos almacenado — algo cambió
        en el artículo desde la última vez.

    Un fallo de red o una respuesta incompleta para un pmc_id concreto se
    registra y se salta — nunca se trata como baja: solo un 404 limpio
    (OaMetadataError NO se lanza en ese caso, fetch_oa_metadata devuelve
    None) cuenta como confirmación real de ausencia.

    No hay diff de catálogo global: cada pmc_id se consulta individualmente,
    solo para los que ya están indexados — el corpus nunca se amplía aquí.
    """
    removals: dict[str, str] = {}   # pmc_id -> "gone" | "retracted" | "license"
    to_reingest: dict[str, OaMetadata] = {}

    for pmc_id in sorted(indexed_pmc_ids):
        try:
            meta = fetch_oa_metadata(pmc_id)
        except OaMetadataError as e:
            print(f"  ⚠ No se pudo comprobar {pmc_id}: {e}. Se deja como está.")
            continue

        if meta is None:
            print(f"  → {pmc_id} ya no está en el PMC Open Access Subset (404).")
            removals[pmc_id] = "gone"
            continue
        if meta.is_retracted:
            print(f"  → {pmc_id} retractado.")
            removals[pmc_id] = "retracted"
            continue
        if meta.license_code not in ALLOWED_PMC_LICENSES:
            print(f"  → {pmc_id} con licencia ya no permitida ({meta.license_code}).")
            removals[pmc_id] = "license"
            continue

        stored_last_updated = _get_stored_pmc_last_updated(pmc_id)
        if stored_last_updated != meta.last_modified:
            to_reingest[pmc_id] = meta

    return removals, to_reingest


def remove_pmc_fulltext(pmc_ids: list[str]):
    """
    Retira el texto completo de los pmc_ids dados: se ejecuta cuando
    check_indexed_pmc_articles() detecta que un artículo ya indexado
    desapareció del PMC Open Access Subset, fue retractado, o pasó a una
    licencia no permitida.

    Borra los chunks PMC (texto completo) pero conserva el documento PubMed
    (abstract) asociado — solo se limpia su enriquecimiento (pmc_id, license,
    pmc_last_updated), porque ese texto completo ya no está disponible/
    permitido y no debemos seguir afirmando que sí lo está.
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from src.ingest.indexer import COLLECTION_NAME, QDRANT_URL

    client = QdrantClient(url=QDRANT_URL)

    for pmc_id in pmc_ids:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(key="metadata.source", match=MatchValue(value="pmc")),
                    FieldCondition(key="metadata.pmc_id", match=MatchValue(value=pmc_id)),
                ]
            ),
        )

        pubmed_filter = Filter(
            must=[
                FieldCondition(key="metadata.source", match=MatchValue(value="pubmed")),
                FieldCondition(key="metadata.pmc_id", match=MatchValue(value=pmc_id)),
            ]
        )
        existing, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=pubmed_filter,
            limit=1,
            with_payload=["metadata"],
            with_vectors=False,
        )
        if existing:
            updated_meta = existing[0].payload.get("metadata", {})
            updated_meta["pmc_id"]           = None
            updated_meta["license"]          = None
            updated_meta["pmc_last_updated"] = None
            client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"metadata": updated_meta},
                points=pubmed_filter,
            )

        print(f"  ✓ Retirado texto completo de {pmc_id} (PubMed conservado)")


def reingest_pmc_articles(to_reingest: dict[str, OaMetadata]) -> int:
    """
    Reingesta completa para cada pmc_id cuya metadata OA cambió:
      1. Descarga el XML actualizado de PMC y regenera los chunks (sin
         tocar Qdrant todavía — si esto falla, los chunks existentes de ese
         pmc_id quedan intactos).
      2. Sube los nuevos chunks (mismos chunk_id que los actuales para las
         secciones que se mantienen → sustitución in-place, sin ventana sin
         datos para esas secciones).
      3. Solo entonces borra los chunks antiguos que hayan quedado
         obsoletos (p. ej. el artículo se acortó y ya no hay tantas
         secciones/chunks como antes).
      4. Actualiza pmc_last_updated y license en el point PubMed asociado.

    Un fallo en la descarga, el parseo o el upsert de un pmc_id concreto se
    registra y se salta ese artículo (continue) — nunca deja el corpus sin
    su texto completo anterior, y no aborta el resto del lote.

    Returns:
        Número de artículos reingeridos con éxito (puede ser menor que
        len(to_reingest) si alguno falló y se saltó).
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
    from src.ingest.indexer import COLLECTION_NAME, QDRANT_URL, upsert_pmc_chunks
    from src.ingest.pmc import fetch_pmc_full_text, parse_pmc_xml_to_chunks, enrich_with_oa_metadata

    client = QdrantClient(url=QDRANT_URL)
    succeeded = 0

    total = len(to_reingest)
    for i, (pmc_id, meta) in enumerate(to_reingest.items(), 1):
        print(f"\nReingiriendo artículo [{i}/{total}] {pmc_id}...")

        # 1. Descargar + parsear + chunking ANTES de tocar Qdrant.
        try:
            xml_text = fetch_pmc_full_text([pmc_id])
            chunks   = parse_pmc_xml_to_chunks(xml_text)
        except Exception as e:
            print(f"  ✗ Error descargando/parseando {pmc_id}: {e}. Se conservan los chunks existentes.")
            continue

        if not chunks:
            print(f"  → Sin chunks para {pmc_id}, saltando (se conservan los existentes).")
            continue

        oa_index = {pmc_id: {"last_updated": meta.last_modified, "license": meta.license_code}}
        chunks   = enrich_with_oa_metadata(chunks, oa_index)

        # IDs de los chunks ya indexados para este pmc_id, capturados antes
        # de tocar nada — nos sirven después para saber cuáles han quedado
        # obsoletos tras la nueva versión.
        old_filter = Filter(
            must=[
                FieldCondition(key="metadata.source", match=MatchValue(value="pmc")),
                FieldCondition(key="metadata.pmc_id", match=MatchValue(value=pmc_id)),
            ]
        )
        old_points, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=old_filter,
            limit=10000,
            with_payload=["metadata"],
            with_vectors=False,
        )
        old_chunk_ids = {p.payload.get("metadata", {}).get("document_id") for p in old_points}
        old_chunk_ids.discard(None)

        # 2. Upsert de los nuevos chunks primero.
        try:
            upsert_pmc_chunks(chunks)
        except Exception as e:
            print(f"  ✗ Error insertando los nuevos chunks de {pmc_id}: {e}. Los chunks anteriores no se han tocado.")
            continue

        # 3. Solo ahora: borrar los chunks antiguos que ya no correspondan a
        # ninguna sección de la versión actual.
        new_chunk_ids = {c["chunk_id"] for c in chunks}
        stale_ids = old_chunk_ids - new_chunk_ids
        if stale_ids:
            client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(key="metadata.source", match=MatchValue(value="pmc")),
                        FieldCondition(key="metadata.pmc_id", match=MatchValue(value=pmc_id)),
                        FieldCondition(key="metadata.document_id", match=MatchAny(any=list(stale_ids))),
                    ]
                ),
            )
            print(f"  → {len(stale_ids)} chunks obsoletos eliminados.")

        # 4. Actualizar pmc_last_updated y license en el point PubMed asociado (si existe)
        # set_payload solo funciona a nivel top-level, así que hacemos scroll + merge + overwrite
        pubmed_filter = Filter(
            must=[
                FieldCondition(key="metadata.source", match=MatchValue(value="pubmed")),
                FieldCondition(key="metadata.pmc_id", match=MatchValue(value=pmc_id)),
            ]
        )
        existing, _ = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=pubmed_filter,
            limit=1,
            with_payload=["metadata"],
            with_vectors=False,
        )
        if existing:
            updated_meta = existing[0].payload.get("metadata", {})
            updated_meta["pmc_last_updated"] = meta.last_modified
            updated_meta["license"]          = meta.license_code
            client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"metadata": updated_meta},
                points=pubmed_filter,
            )

        succeeded += 1
        print(f"  → {len(chunks)} chunks actualizados.")

    return succeeded


# -------------- FLUJO DE MANTENIMIENTO --------------

def run_daily_update() -> dict:
    """
    Flujo de mantenimiento del corpus PMC ya indexado:
      1. Lista los pmc_ids actualmente indexados.
      2. Para cada uno, consulta su metadata OA actual (bucket público,
         sin credenciales) y decide baja o reingesta.
      3. Aplica bajas (borra chunks PMC + limpia enriquecimiento PubMed) y
         reingesta los cambios.

    Sin CSV, sin diff de catálogo global, sin fichero que reemplazar al
    final — cada pmc_id se consulta y se actualiza de forma independiente.
    Nunca amplía el corpus: eso es responsabilidad exclusiva de /ingest.

    Returns:
        Resumen: pmc_checked, reingested (con éxito), retracted,
        removed_license, removed_gone — usado por /corpus/update para
        construir UpdateResult.
    """
    print("Cargando PMC IDs indexados en Qdrant...")
    indexed_pmc_ids = get_indexed_pmc_ids()
    print(f"  → {len(indexed_pmc_ids)} PMC IDs indexados")

    if not indexed_pmc_ids:
        print("→ No hay artículos PMC indexados. Nada que actualizar.")
        return {"pmc_checked": 0, "reingested": 0, "retracted": 0, "removed_license": 0, "removed_gone": 0}

    removals, to_reingest = check_indexed_pmc_articles(indexed_pmc_ids)
    print(f"\nBajas: {len(removals)} | Cambios a reingerir: {len(to_reingest)}")

    if removals:
        print(f"\nRetirando {len(removals)} artículos PMC de Qdrant...")
        remove_pmc_fulltext(list(removals.keys()))

    reingested = 0
    if to_reingest:
        reingested = reingest_pmc_articles(to_reingest)

    print("\nActualización PMC completada.")

    reasons = removals.values()
    return {
        "pmc_checked":      len(indexed_pmc_ids),
        "reingested":       reingested,
        "retracted":        sum(1 for r in reasons if r == "retracted"),
        "removed_license":  sum(1 for r in removals.values() if r == "license"),
        "removed_gone":     sum(1 for r in removals.values() if r == "gone"),
    }


if __name__ == "__main__":
    run_daily_update()
