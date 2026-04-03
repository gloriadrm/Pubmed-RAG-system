"""
oa_updater.py
-------------
Gestiona el OA file list de PMC (Open Access):
  - Descarga el CSV desde NCBI
  - Compara con la versión anterior para detectar cambios
  - Actualiza solo los points afectados en Qdrant
  - Expone lookup_oa_csv() para cruzar PMIDs con PMCIDs

Puede ejecutarse manualmente o programarse con cron:
    0 3 * * * /path/to/.venv/bin/python -m src.data_actualization.oa_updater
"""

import os
import shutil
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

from src.config import OA_CSV_URL, DATA_DIR, LOCAL_OA_CSV as LOCAL_OA


# -------------- CARGA / DESCARGA --------------

def download_oa_csv(save_path: Path = LOCAL_OA, max_retries: int = 3) -> pd.DataFrame:
    """
    Descarga el OA file list de NCBI FTP y lo guarda localmente.
    Reintenta hasta max_retries veces ante errores de red.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando OA file list desde NCBI FTP (~100 MB)...")

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(OA_CSV_URL, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)
            print(f"  → guardado en {save_path}")
            return load_oa_csv(save_path)
        except Exception as e:
            last_error = e
            save_path.unlink(missing_ok=True)   # limpiar descarga parcial
            if attempt < max_retries:
                print(f"  ⚠ Error en intento {attempt}/{max_retries}: {e}. Reintentando...")
            else:
                print(f"  ✗ Fallaron {max_retries} intentos de descarga.")

    raise last_error


def load_oa_csv(path: Path = LOCAL_OA) -> pd.DataFrame:
    """
    Carga el OA file list con columnas normalizadas.

    Columnas originales del CSV (comma-separado):
        File, Article Citation, Accession ID, Last Updated (YYYY-MM-DD HH:MM:SS), PMID, License
    """
    df = pd.read_csv(path, sep=",", dtype=str, header=0)

    # Mapear por nombre real → nombre interno (robusto ante cambios de orden)
    col_map = {
        "File":                                  "file",
        "Article Citation":                      "citation",
        "Accession ID":                          "pmc_id",
        "Last Updated (YYYY-MM-DD HH:MM:SS)":   "last_updated",
        "PMID":                                  "pm_id",
        "License":                               "license",
    }
    df = df.rename(columns=col_map)
    df = df[["pmc_id", "pm_id", "file", "last_updated", "license"]].copy()
    df.dropna(subset=["pmc_id"], inplace=True)
    df["pm_id"]  = df["pm_id"].str.strip()
    df["pmc_id"] = df["pmc_id"].str.strip()
    return df


def get_oa_df(force_download: bool = False) -> pd.DataFrame:
    """Devuelve el DataFrame del OA CSV (descarga si no existe o se fuerza)."""
    if force_download or not LOCAL_OA.exists():
        return download_oa_csv()
    return load_oa_csv()


# -------------- LOOKUP --------------

def lookup_oa_csv(pmids: list[str], df: pd.DataFrame, licenses: list[str] = None) -> dict:
    """
    Cruza una lista de PMIDs con el OA file list.

    Args:
        pmids:    lista de PMIDs a cruzar
        df:       DataFrame cargado con load_oa_csv()
        licenses: lista de licencias permitidas (None = todas).
                  Ej: ["CC BY", "CC BY-SA", "CC BY-NC"]

    Returns:
        {pm_id: {"pmc_id": "PMC13900", "last_updated": "2025-06-04 10:25:31", "license": "CC BY"}}
        Solo incluye PMIDs que tienen entrada en PMC Open Access.
    """
    pmid_set = set(str(p).strip() for p in pmids)
    matches  = df[df["pm_id"].isin(pmid_set)]

    if licenses is not None:
        matches = matches[matches["license"].isin(licenses)]

    result = {}
    for _, row in matches.iterrows():
        result[str(row["pm_id"])] = {
            "pmc_id":       row["pmc_id"],
            "last_updated": row["last_updated"],
            "license":      row["license"],
        }
    return result


# Licencias permitidas para explotación del texto completo (igual que pipeline.py)
from src.config import ALLOWED_PMC_LICENSES


# -------------- DIFF + UPDATE QDRANT --------------

def detect_pmc_removals(old_df: pd.DataFrame, new_df: pd.DataFrame) -> list[str]:
    """
    Devuelve los pmc_ids que deben borrarse de Qdrant porque:
      1. Estaban en el OA CSV anterior pero han desaparecido del nuevo
         (ya no son open access).
      2. Su licencia ha cambiado a una no permitida (ND o NO-CC CODE).

    Returns:
        lista de pmc_ids a borrar
    """
    new_index = new_df.set_index("pmc_id")["license"].to_dict()
    to_remove = []

    for _, row in old_df.iterrows():
        pmc_id = row["pmc_id"]
        if pmc_id not in new_index:
            # Desapareció del catálogo OA
            to_remove.append(pmc_id)
        elif new_index[pmc_id] not in ALLOWED_PMC_LICENSES:
            # Licencia cambiada a no permitida
            to_remove.append(pmc_id)

    return to_remove


def diff_oa_csv(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve las filas de new_df que ya existían en old_df pero con last_updated distinto.
    Las entradas nuevas se ignoran: se ingestarán por el pipeline de PubMed.
    """
    old_index = old_df.set_index("pmc_id")["last_updated"].to_dict()

    changed = []
    for _, row in new_df.iterrows():
        pmc_id = row["pmc_id"]
        if pmc_id in old_index and old_index[pmc_id] != row["last_updated"]:
            changed.append(row)

    if not changed:
        return pd.DataFrame(columns=new_df.columns)
    return pd.DataFrame(changed).reset_index(drop=True)


def reingest_changed_pmc(changed_df: pd.DataFrame):
    """
    Reingesta completa para cada pmc_id cuyo last_updated ha cambiado:
      1. Descarga el XML actualizado de PMC
      2. Parsea y regenera chunks
      3. Borra los chunks antiguos de ese pmc_id en Qdrant
      4. Inserta los nuevos chunks
      5. Actualiza last_updated y license en el point PubMed asociado
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchValue
    from src.ingest.indexer import COLLECTION_NAME, QDRANT_URL, upsert_pmc_chunks
    from src.ingest.pmc import fetch_pmc_full_text, parse_pmc_xml_to_chunks, enrich_with_oa_metadata

    client = QdrantClient(url=QDRANT_URL)

    total = len(changed_df)
    for i, (_, row) in enumerate(changed_df.iterrows(), 1):
        pmc_id = row["pmc_id"]
        print(f"\nReingiriendo artículo [{i}/{total}] {pmc_id}...")

        # 1. Descargar XML actualizado
        xml_text = fetch_pmc_full_text([pmc_id])

        # 2. Parsear + chunking adaptativo
        chunks = parse_pmc_xml_to_chunks(xml_text)
        if not chunks:
            print(f"  → Sin chunks para {pmc_id}, saltando.")
            continue

        # 3. Enriquecer con OA metadata
        oa_index = {pmc_id: {"last_updated": row["last_updated"], "license": row["license"]}}
        chunks   = enrich_with_oa_metadata(chunks, oa_index)

        # 4. Borrar chunks PMC antiguos de este pmc_id
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(key="metadata.source", match=MatchValue(value="pmc")),
                    FieldCondition(key="metadata.pmc_id", match=MatchValue(value=pmc_id)),
                ]
            ),
        )

        # 5. Insertar nuevos chunks
        upsert_pmc_chunks(chunks)

        # 6. Actualizar pmc_last_updated y license en el point PubMed asociado (si existe)
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
            updated_meta["pmc_last_updated"] = row["last_updated"]
            updated_meta["license"]          = row["license"]
            client.set_payload(
                collection_name=COLLECTION_NAME,
                payload={"metadata": updated_meta},
                points=pubmed_filter,
            )

        print(f"  → {len(chunks)} chunks actualizados.")


# -------------- FLUJO DIARIO --------------

def run_daily_update():
    """
    Flujo completo de actualización diaria:
      1. Carga el OA CSV local anterior (si existe)
      2. Descarga el nuevo desde NCBI
      3. Calcula el diff (solo artículos ya indexados que han cambiado)
      4. Reingesta completa para los pmc_ids afectados
      5. Reemplaza el CSV local con el nuevo
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Cargar versión anterior
    if LOCAL_OA.exists():
        print("Cargando OA CSV anterior...")
        old_df = load_oa_csv(LOCAL_OA)
    else:
        print("Primera ejecución — no hay OA CSV previo.")
        old_df = pd.DataFrame(columns=["pmc_id", "pm_id", "file", "last_updated", "license"])

    # Descargar nuevo a un temporal
    tmp_path = LOCAL_OA.with_suffix(".new.csv")
    new_df   = download_oa_csv(tmp_path)

    # Diff: solo artículos ya indexados con last_updated cambiado
    changed_df = diff_oa_csv(old_df, new_df)
    print(f"Artículos PMC modificados: {len(changed_df)}")

    if not changed_df.empty:
        reingest_changed_pmc(changed_df)

    # Reemplazar CSV local
    shutil.move(str(tmp_path), str(LOCAL_OA))
    print("OA CSV actualizado correctamente.")


if __name__ == "__main__":
    run_daily_update()
