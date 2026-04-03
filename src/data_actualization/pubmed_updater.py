"""
pubmed_updater.py
-----------------
Actualización incremental de artículos PubMed usando los update files diarios de NLM.

Estrategia:
  - Listamos los update files del FTP de NLM (pubmedYYnXXXX.xml.gz).
  - Calculamos cuáles no hemos procesado desde la última ejecución.
  - Para cada fichero pendiente:
      · Descargamos y parseamos el XML comprimido.
      · Filtramos artículos con status=MEDLINE (registros completos y estables).
      · Cruzamos con los PMIDs ya indexados en Qdrant:
          - PMID en Qdrant + en update file        → upsert (re-embed con datos actualizados)
          - PMID en DeleteCitation + en Qdrant      → borrado selectivo
          - PMID no en Qdrant                       → ignorar (la ingesta temática decide si entra)
  - Guardamos el nombre del último fichero procesado para no repetir en la próxima ejecución.

Uso:
    python -m src.data_actualization.pubmed_updater
"""

import gzip
import re
import requests
from pathlib import Path
from xml.etree import ElementTree as ET

from qdrant_client.models import Filter, FieldCondition, MatchValue

from src.ingest.pubmed import parse_pubmed_xml_to_dicts
from src.ingest.indexer import (
    COLLECTION_NAME,
    client as qdrant_client,
    upsert_pubmed_articles,
)

# -------------- CONSTANTES --------------

from src.config import PUBMED_UPDATE_BASE, DATA_DIR, PUBMED_LAST_UPDATE as LAST_PROCESSED_FILE


# -------------- TRACKING DEL ÚLTIMO FICHERO --------------

def get_last_processed_file() -> str | None:
    """
    Devuelve el nombre del último update file procesado (p.ej. 'pubmed26n1392.xml.gz'),
    o None si es la primera ejecución.
    """
    if not LAST_PROCESSED_FILE.exists():
        return None
    content = LAST_PROCESSED_FILE.read_text().strip()
    return content if content else None


def save_last_processed_file(filename: str):
    """Persiste el nombre del último update file procesado."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LAST_PROCESSED_FILE.write_text(filename)


# -------------- LISTADO DE FICHEROS --------------

def list_pubmed_update_files(base_url: str = PUBMED_UPDATE_BASE) -> list[dict]:
    """
    Lista todos los update files disponibles en el FTP de NLM,
    ordenados por número secuencial.

    Returns:
        [{"filename": "pubmed26n1392.xml.gz", "url": "https://..."}]
    """
    response = requests.get(base_url, timeout=30)
    response.raise_for_status()

    pattern = r'pubmed\d+n\d+\.xml\.gz'
    files   = list(set(re.findall(pattern, response.text)))

    def seq_number(filename: str) -> int:
        m = re.search(r'n(\d+)', filename)
        return int(m.group(1)) if m else 0

    files_sorted = sorted(files, key=seq_number)
    return [{"filename": f, "url": PUBMED_UPDATE_BASE + f} for f in files_sorted]


def get_pending_files(all_files: list[dict], last_processed: str | None) -> list[dict]:
    """
    Devuelve los update files no procesados todavía.

    - Si last_processed es None (primera ejecución): devuelve solo el más reciente
      para no descargar cientos de ficheros la primera vez.
    - Si last_processed no está en la lista actual: aviso + procesa solo el último.
    - En caso normal: devuelve todos los posteriores al último procesado.
    """
    if not all_files:
        return []

    if last_processed is None:
        print("Primera ejecución: procesando solo el fichero más reciente.")
        return [all_files[-1]]

    filenames = [f["filename"] for f in all_files]

    if last_processed not in filenames:
        print(f"  ⚠ '{last_processed}' no encontrado en la lista actual. Procesando solo el más reciente.")
        return [all_files[-1]]

    idx = filenames.index(last_processed)
    return all_files[idx + 1:]


# -------------- PARSEO DEL UPDATE FILE --------------

def parse_update_file(url: str) -> tuple[list[dict], list[str]]:
    """
    Descarga y parsea un update file de PubMed (.xml.gz).

    Reutiliza parse_pubmed_xml_to_dicts() de src/ingest/pubmed.py ya que los
    PubmedArticle dentro del update file tienen la misma estructura que los de EFetch.
    Solo procesa artículos con status=MEDLINE (completos y estables).

    Returns:
        articles      — lista de artículos parseados (dicts estándar del sistema)
        deleted_pmids — lista de PMIDs marcados para borrado en DeleteCitation
    """
    print(f"  Descargando {url.split('/')[-1]}...")
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    xml_bytes = gzip.decompress(response.content)
    print(f"  → {len(xml_bytes):,} bytes descomprimidos")

    root = ET.fromstring(xml_bytes)

    # --- Artículos MEDLINE ---
    # Filtramos por Status=MEDLINE: registros completamente procesados y curados.
    # Descartamos Publisher, In-Process, etc. (incompletos o aún en revisión).
    medline_elems = [
        elem for elem in root.findall("./PubmedArticle")
        if (elem.find("./MedlineCitation") is not None
            and elem.find("./MedlineCitation").attrib.get("Status") == "MEDLINE")
    ]

    # Envolvemos en <PubmedArticleSet> para reutilizar el parser existente
    wrapper = ET.Element("PubmedArticleSet")
    for elem in medline_elems:
        wrapper.append(elem)

    articles = parse_pubmed_xml_to_dicts(ET.tostring(wrapper, encoding="unicode"))

    # --- PMIDs a borrar ---
    deleted_pmids = []
    for delete_node in root.findall("./DeleteCitation"):
        for pmid_node in delete_node.findall("./PMID"):
            if pmid_node.text:
                deleted_pmids.append(pmid_node.text.strip())

    print(f"  → {len(articles)} artículos MEDLINE en el fichero | {len(deleted_pmids)} PMIDs borrados en NLM (global)")
    return articles, deleted_pmids


# -------------- CRUCE CON QDRANT --------------

def get_indexed_pmids() -> set[str]:
    """
    Devuelve el conjunto de pm_ids actualmente indexados en Qdrant (source=pubmed).
    Hace scroll completo de la colección para no perder ningún punto.
    """
    indexed = set()
    offset  = None

    while True:
        results, next_offset = qdrant_client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="metadata.source", match=MatchValue(value="pubmed"))]
            ),
            limit=1000,
            offset=offset,
            with_payload=["metadata"],
            with_vectors=False,
        )
        for point in results:
            pm_id = point.payload.get("metadata", {}).get("pm_id")
            if pm_id:
                indexed.add(str(pm_id))

        if next_offset is None:
            break
        offset = next_offset

    return indexed


def apply_pubmed_updates(
    articles:       list[dict],
    deleted_pmids:  list[str],
    indexed_pmids:  set[str],
):
    """
    Aplica los cambios sobre la colección de Qdrant:
      - Artículos revisados ya indexados  → upsert (regenera embedding con datos actualizados)
      - PMIDs borrados ya indexados       → delete del point PubMed
    """
    # Upsert de revisados
    to_upsert = [a for a in articles if str(a.get("pm_id", "")) in indexed_pmids]
    print(f"  → {len(to_upsert)} artículos a actualizar (upsert)")

    if to_upsert:
        upsert_pubmed_articles(to_upsert)

    # Borrado selectivo
    to_delete = [pmid for pmid in deleted_pmids if pmid in indexed_pmids]
    print(f"  → {len(to_delete)} artículos a borrar")

    for pmid in to_delete:
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(key="metadata.source", match=MatchValue(value="pubmed")),
                    FieldCondition(key="metadata.pm_id",  match=MatchValue(value=pmid)),
                ]
            ),
        )
        print(f"    ✓ Borrado PMID {pmid}")


# -------------- FLUJO PRINCIPAL --------------

def run_pubmed_update():
    """
    Flujo completo de actualización incremental de PubMed.

    1. Lista los update files disponibles en el FTP.
    2. Calcula cuáles no se han procesado desde la última ejecución.
    3. Para cada fichero pendiente: parsea + aplica cambios en Qdrant.
    4. Guarda el nombre del último fichero procesado.
    """
    print("=== Actualización incremental PubMed ===")

    all_files      = list_pubmed_update_files()
    last_processed = get_last_processed_file()

    print(f"Update files disponibles : {len(all_files)}")
    print(f"Último procesado         : {last_processed or 'ninguno (primera ejecución)'}")

    pending = get_pending_files(all_files, last_processed)
    print(f"Ficheros pendientes      : {len(pending)}")

    if not pending:
        print("No hay ficheros nuevos. Nada que actualizar.")
        return

    # Cargamos los PMIDs indexados una sola vez (scroll completo)
    print("\nCargando PMIDs indexados en Qdrant...")
    indexed_pmids = get_indexed_pmids()
    print(f"  → {len(indexed_pmids)} PMIDs indexados")

    for file_info in pending:
        filename = file_info["filename"]
        print(f"\nProcesando {filename}...")
        articles, deleted_pmids = parse_update_file(file_info["url"])
        apply_pubmed_updates(articles, deleted_pmids, indexed_pmids)
        save_last_processed_file(filename)
        print(f"  ✓ {filename} procesado.")

    print("\n=== Actualización PubMed completada ===")


if __name__ == "__main__":
    run_pubmed_update()
