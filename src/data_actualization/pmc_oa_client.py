"""
pmc_oa_client.py
-----------------
Cliente para el PMC Cloud Service en AWS — sustituye a oa_file_list.csv,
retirado por NCBI en 2026 (ver
https://ncbiinsights.ncbi.nlm.nih.gov/2026/02/12/pmc-article-dataset-distribution-services/).
El fichero sigue disponible temporalmente en
https://ftp.ncbi.nlm.nih.gov/pub/pmc/deprecated/oa_file_list.csv, pero NCBI
ha anunciado su retirada definitiva en agosto de 2026 — no se usa aquí.

Dos responsabilidades independientes (deliberadamente separadas: cada una
puede fallar o cachearse de forma distinta):

  1. Resolución PMID → PMCID
     Vía PMC-ids.csv.gz — bulk, sigue vigente (no deprecado), se descarga
     y cachea igual que antes se hacía con oa_file_list.csv.

  2. Metadata OA por artículo (licencia, retractado, última modificación)
     Vía el JSON individual público en el bucket S3 "pmc-oa-opendata"
     (https://pmc-oa-opendata.s3.amazonaws.com/metadata/<PMCID>.<version>.json),
     accesible sin credenciales por HTTPS. Se consulta artículo a artículo
     — no hay un catálogo global que descargar ni diferenciar, lo cual
     encaja mejor con un corpus pequeño y curado que el modelo anterior
     (pensado para espejar todo PMC OA).

La descarga del texto completo (E-Utilities eFetch) no se ve afectada por
esta migración y sigue viviendo, sin cambios, en src/ingest/pmc.py.
"""

import gzip
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

from src.config import DATA_DIR, PMC_IDS_URL, LOCAL_PMC_IDS_CSV, PMC_OA_METADATA_BASE_URL


class OaMetadataError(Exception):
    """
    Fallo al consultar la metadata OA de un artículo: red caída, respuesta
    HTTP inesperada, o JSON incompleto/corrupto.

    Deliberadamente distinto de "None" (que fetch_oa_metadata reserva para
    un 404 limpio, es decir, "confirmado ausente del PMC Open Access
    Subset"). Quien llama debe tratar esta excepción como "no se pudo
    determinar" — dejar el artículo como está — y nunca como "retirado":
    tratar un fallo de red como si fuera un 404 borraría contenido válido
    por una incidencia transitoria.
    """


# -------------- 1. Resolución PMID → PMCID --------------

def download_pmc_ids_csv(save_path: Path = LOCAL_PMC_IDS_CSV, max_retries: int = 3) -> pd.DataFrame:
    """Descarga PMC-ids.csv.gz de NCBI, lo descomprime y lo guarda localmente."""
    save_path.parent.mkdir(parents=True, exist_ok=True)

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            with requests.get(PMC_IDS_URL, stream=True, timeout=300) as r:
                r.raise_for_status()
                with gzip.GzipFile(fileobj=r.raw) as gz, open(save_path, "wb") as f:
                    shutil.copyfileobj(gz, f)
            return load_pmc_ids_csv(save_path)
        except Exception as e:
            last_error = e
            save_path.unlink(missing_ok=True)
            if attempt >= max_retries:
                raise

    raise last_error  # pragma: no cover


def load_pmc_ids_csv(path: Path = LOCAL_PMC_IDS_CSV) -> pd.DataFrame:
    """
    Carga el mapeo PMID→PMCID desde el CSV local ya descomprimido.

    Columnas originales del CSV:
        Journal Title,ISSN,eISSN,Year,Volume,Issue,Page,DOI,PMCID,PMID,Manuscript Id,Release Date
    """
    df = pd.read_csv(path, sep=",", dtype=str, header=0)
    df = df.rename(columns={"PMCID": "pmc_id", "PMID": "pm_id"})
    df = df[["pmc_id", "pm_id"]].copy()
    df.dropna(subset=["pmc_id", "pm_id"], inplace=True)
    df["pm_id"]  = df["pm_id"].str.strip()
    df["pmc_id"] = df["pmc_id"].str.strip()
    return df


def get_pmc_ids_df(force_download: bool = False) -> pd.DataFrame:
    """Devuelve el DataFrame de mapeo PMID→PMCID (descarga si no existe o se fuerza)."""
    if force_download or not LOCAL_PMC_IDS_CSV.exists():
        return download_pmc_ids_csv()
    return load_pmc_ids_csv()


def resolve_pmcids(pmids: list[str], df: pd.DataFrame) -> dict[str, str]:
    """pm_id -> pmc_id, solo para los PMIDs que tienen un PMCID asociado."""
    pmid_set = set(str(p).strip() for p in pmids)
    matches = df[df["pm_id"].isin(pmid_set)]
    return dict(zip(matches["pm_id"], matches["pmc_id"]))


# -------------- 2. Metadata OA por artículo --------------

@dataclass
class OaMetadata:
    pmcid:         str
    pmid:          Optional[str]
    license_code:  Optional[str]
    is_retracted:  bool
    last_modified: Optional[str]   # cabecera HTTP Last-Modified del objeto JSON


_REQUIRED_KEYS = {"pmcid", "license_code", "is_retracted"}


def fetch_oa_metadata(pmcid: str, version: int = 1) -> Optional[OaMetadata]:
    """
    Descarga la metadata OA individual de un artículo (JSON público, sin
    credenciales, en el bucket S3 pmc-oa-opendata).

    Devuelve:
      - OaMetadata si el artículo existe en el PMC Open Access Subset.
      - None SOLO ante un 404 limpio — "confirmado ausente".

    Lanza OaMetadataError ante cualquier otro problema (fallo de red,
    status HTTP inesperado, JSON no parseable, o JSON incompleto —
    faltan claves esperadas). Ver OaMetadataError: nunca tratar esto como
    equivalente a None.
    """
    url = f"{PMC_OA_METADATA_BASE_URL}/{pmcid}.{version}.json"

    try:
        response = requests.get(url, timeout=15)
    except requests.exceptions.RequestException as e:
        raise OaMetadataError(f"Fallo de red consultando metadata de {pmcid}: {e}") from e

    if response.status_code == 404:
        return None
    if response.status_code != 200:
        raise OaMetadataError(
            f"Respuesta HTTP inesperada ({response.status_code}) consultando metadata de {pmcid}"
        )

    try:
        data = response.json()
    except ValueError as e:
        raise OaMetadataError(f"Respuesta no parseable como JSON para {pmcid}: {e}") from e

    missing = _REQUIRED_KEYS - data.keys()
    if missing:
        raise OaMetadataError(f"Respuesta incompleta para {pmcid}: faltan claves {sorted(missing)}")

    return OaMetadata(
        pmcid=data["pmcid"],
        pmid=str(data["pmid"]) if data.get("pmid") is not None else None,
        license_code=data["license_code"],
        is_retracted=bool(data["is_retracted"]),
        last_modified=response.headers.get("Last-Modified"),
    )


def fetch_oa_metadata_bulk(pmcids: list[str]) -> dict[str, OaMetadata]:
    """
    fetch_oa_metadata() para varios pmc_ids. Los que resuelven a None (404)
    o lanzan OaMetadataError se omiten del resultado — no del todo lo mismo:
    quien necesite distinguir "ausente" de "no se pudo comprobar" debe
    llamar a fetch_oa_metadata() directamente artículo a artículo.
    """
    result = {}
    for pmcid in pmcids:
        try:
            meta = fetch_oa_metadata(pmcid)
        except OaMetadataError:
            continue
        if meta is not None:
            result[pmcid] = meta
    return result
