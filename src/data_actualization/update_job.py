"""
update_job.py
--------------
Ejecución de la actualización del corpus (PubMed + PMC) fuera del proceso
que atiende FastAPI.

El parseo de los update files de PubMed usa xml.etree.ElementTree sobre
ficheros de ~100-200MB — CPU-bound en Python puro, no libera el GIL.
Ejecutarlo dentro del propio proceso de la API (aunque fuera en un thread,
vía asyncio.to_thread) bloquearía igualmente /health, /query,
/corpus/status y el resto de peticiones mientras dura. Por eso el trabajo
se lanza en un proceso Python independiente (ProcessPoolExecutor,
ver src/api/main.py) y el progreso se coordina a través de un fichero de
estado en disco — no hay memoria compartida entre procesos, así que el
fichero es el único punto de verdad tanto para el proceso worker como para
el proceso de la API que atiende GET /corpus/update/status.

Estados: idle → running → completed | failed
"""

import json
from datetime import datetime, timezone
from typing import Optional

from src.config import UPDATE_JOB_STATUS_FILE


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_job_status() -> dict:
    """Estado actual del job — {"state": "idle"} si nunca se ha ejecutado
    ninguno o el fichero está corrupto/ilegible."""
    if not UPDATE_JOB_STATUS_FILE.exists():
        return {"state": "idle"}
    try:
        return json.loads(UPDATE_JOB_STATUS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {"state": "idle"}


def write_job_status(status: dict):
    UPDATE_JOB_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
    UPDATE_JOB_STATUS_FILE.write_text(json.dumps(status))


def start_job_status(job_id: str) -> dict:
    """
    Escribe el estado inicial "running" — se llama de forma SÍNCRONA en el
    proceso de la API, antes de enviar el trabajo al ProcessPoolExecutor
    (nunca dentro del proceso worker): así, una segunda petición que llegue
    mientras el primer job todavía no ha arrancado ya ve "running" en vez
    de una carrera en la que ambas lean "idle" y arranquen dos jobs a la
    vez. Ver el lock en src/api/main.py alrededor de esta llamada.
    """
    status = {
        "job_id":      job_id,
        "state":       "running",
        "phase":       "pubmed",
        "started_at":  _now_iso(),
        "finished_at": None,
        "result":      None,
        "error":       None,
        "log":         ["Actualización iniciada."],
    }
    write_job_status(status)
    return status


def mark_interrupted_if_running():
    """
    Se llama al arrancar el proceso de la API (lifespan startup). Si el
    fichero de estado dice "running", el proceso worker de la ejecución
    anterior murió sin terminar (p. ej. se reinició el contenedor) — lo
    marcamos "failed" para que el frontend no se quede creyendo
    indefinidamente que sigue en curso.
    """
    status = read_job_status()
    if status.get("state") == "running":
        status["state"] = "failed"
        status["error"] = "Interrumpida: el proceso se reinició mientras se ejecutaba."
        status["finished_at"] = _now_iso()
        status.setdefault("log", []).append("Actualización interrumpida por reinicio del servicio.")
        write_job_status(status)


def run_update_job(job_id: str):
    """
    Cuerpo del job — ejecutado en un proceso worker independiente
    (ProcessPoolExecutor, ver src/api/main.py). Función a nivel de módulo
    para que sea picklable (requisito de ProcessPoolExecutor).

    No amplía el corpus (eso es /ingest): revisa PubMed (update files de
    NLM pendientes) y PMC (metadata OA de los artículos ya indexados) y
    aplica los cambios correspondientes. El progreso fichero a fichero de
    PubMed ya se persiste dentro de run_pubmed_update() — si este proceso
    muere a mitad, la siguiente ejecución retoma desde ahí.
    """
    status = read_job_status()
    log = status.get("log", [])

    try:
        from src.data_actualization.pubmed_updater import run_pubmed_update
        from src.data_actualization.oa_updater import run_daily_update
        from src.config import LAST_CORPUS_UPDATE_FILE

        log.append("Revisando PubMed (update files de NLM)...")
        write_job_status({**status, "log": log})

        pubmed_summary = run_pubmed_update()
        log.append(
            f"PubMed: {pubmed_summary['files_processed']} ficheros procesados, "
            f"{pubmed_summary['pmids_checked']} PMIDs revisados, "
            f"{pubmed_summary['updated']} modificados, {pubmed_summary['deleted']} borrados."
        )
        write_job_status({**status, "phase": "pmc", "log": log})

        log.append("Revisando PMC (metadata OA por artículo)...")
        pmc_summary = run_daily_update()
        log.append(
            f"PMC: {pmc_summary['pmc_checked']} revisados, "
            f"{pmc_summary['reingested']} reindexados, {pmc_summary['retracted']} retractados, "
            f"{pmc_summary['removed_license'] + pmc_summary['removed_gone']} retirados."
        )

        finished_at = _now_iso()
        LAST_CORPUS_UPDATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        LAST_CORPUS_UPDATE_FILE.write_text(finished_at)

        result = {
            "pubmed_checked":     pubmed_summary["pmids_checked"],
            "pubmed_updated":     pubmed_summary["updated"],
            "pubmed_deleted":     pubmed_summary["deleted"],
            "pmc_checked":        pmc_summary["pmc_checked"],
            "pmc_reingested":     pmc_summary["reingested"],
            "pmc_retracted":      pmc_summary["retracted"],
            "pmc_removed_license": pmc_summary["removed_license"],
            "pmc_removed_gone":   pmc_summary["removed_gone"],
        }

        write_job_status({
            "job_id":      job_id,
            "state":       "completed",
            "phase":       None,
            "started_at":  status.get("started_at"),
            "finished_at": finished_at,
            "result":      result,
            "error":       None,
            "log":         log,
        })
    except Exception as e:
        log.append(f"Error: {e}")
        write_job_status({
            "job_id":      job_id,
            "state":       "failed",
            "phase":       None,
            "started_at":  status.get("started_at"),
            "finished_at": _now_iso(),
            "result":      None,
            "error":       str(e),
            "log":         log,
        })
