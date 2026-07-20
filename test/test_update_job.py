"""
test_update_job.py
-------------------
Tests pytest del job de actualización asíncrono (src/data_actualization/update_job.py
y los endpoints POST /corpus/update, GET /corpus/update/status en src/api/main.py).

Tres niveles:
  1. Lógica pura del job (lectura/escritura de estado, transiciones) — sin
     multiprocessing, sin red.
  2. Endpoints vía FastAPI TestClient — arranque, rechazo de concurrencia,
     cooldown, consulta de estado.
  3. Prueba real de que la API sigue respondiendo mientras el job corre en
     un proceso aparte (ProcessPoolExecutor real, con un bucle CPU-bound
     puro para simular el parseo XML — sin red, rápido mecánicamente).

Uso:
    pytest test/test_update_job.py -v
"""

import json
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta, timezone
from multiprocessing import get_context

import pytest
from fastapi.testclient import TestClient

import src.data_actualization.update_job as uj
import src.api.main as main_module


# ============================================================
# 1. Lógica pura del job
# ============================================================

@pytest.fixture
def status_file(monkeypatch, tmp_path):
    path = tmp_path / "update_job_status.json"
    monkeypatch.setattr(uj, "UPDATE_JOB_STATUS_FILE", path)
    return path


class TestJobStatusIO:

    def test_read_status_idle_when_no_file(self, status_file):
        assert uj.read_job_status() == {"state": "idle"}

    def test_read_status_idle_when_corrupt_file(self, status_file):
        status_file.parent.mkdir(parents=True, exist_ok=True)
        status_file.write_text("{not json")
        assert uj.read_job_status() == {"state": "idle"}

    def test_start_job_status_writes_running(self, status_file):
        status = uj.start_job_status("job-1")
        assert status["state"] == "running"
        assert status["job_id"] == "job-1"
        assert status["phase"] == "pubmed"
        assert status["started_at"] is not None
        assert uj.read_job_status()["state"] == "running"

    def test_mark_interrupted_if_running_flips_to_failed(self, status_file):
        uj.start_job_status("job-2")
        uj.mark_interrupted_if_running()

        status = uj.read_job_status()
        assert status["state"] == "failed"
        assert "reinició" in status["error"]
        assert status["finished_at"] is not None

    def test_mark_interrupted_if_running_noop_when_idle(self, status_file):
        uj.mark_interrupted_if_running()
        assert uj.read_job_status() == {"state": "idle"}

    def test_mark_interrupted_if_running_noop_when_completed(self, status_file):
        uj.write_job_status({"state": "completed", "finished_at": "2026-01-01T00:00:00+00:00"})
        uj.mark_interrupted_if_running()
        assert uj.read_job_status()["state"] == "completed"


class TestRunUpdateJob:
    """run_update_job() invocado directamente (sin ProcessPoolExecutor) para
    testear la máquina de estados sin el coste/latencia de multiprocessing —
    la lógica es idéntica se ejecute en el proceso actual o en uno aparte."""

    def test_completes_successfully(self, status_file, monkeypatch):
        uj.start_job_status("job-3")

        monkeypatch.setattr(
            "src.data_actualization.pubmed_updater.run_pubmed_update",
            lambda: {"pmids_checked": 5, "files_processed": 1, "updated": 2, "deleted": 0},
        )
        monkeypatch.setattr(
            "src.data_actualization.oa_updater.run_daily_update",
            lambda: {"pmc_checked": 3, "reingested": 1, "retracted": 0, "removed_license": 0, "removed_gone": 0},
        )
        monkeypatch.setattr("src.config.LAST_CORPUS_UPDATE_FILE", status_file.parent / "last_corpus_update.txt")

        uj.run_update_job("job-3")

        status = uj.read_job_status()
        assert status["state"] == "completed"
        assert status["job_id"] == "job-3"
        assert status["phase"] is None
        assert status["error"] is None
        assert status["result"]["pubmed_checked"] == 5
        assert status["result"]["pubmed_updated"] == 2
        assert status["result"]["pmc_reingested"] == 1
        assert any("PubMed" in line for line in status["log"])
        assert any("PMC" in line for line in status["log"])

    def test_failure_is_recorded_not_raised(self, status_file, monkeypatch):
        uj.start_job_status("job-4")

        def failing_pubmed_update():
            raise ConnectionError("NLM no disponible")

        monkeypatch.setattr(
            "src.data_actualization.pubmed_updater.run_pubmed_update", failing_pubmed_update
        )

        uj.run_update_job("job-4")   # no debe propagar la excepción

        status = uj.read_job_status()
        assert status["state"] == "failed"
        assert "NLM no disponible" in status["error"]
        assert status["result"] is None
        assert status["finished_at"] is not None


# ============================================================
# 2. Endpoints (FastAPI TestClient)
# ============================================================

@pytest.fixture
def client(status_file, monkeypatch):
    # main.py importó read_job_status/start_job_status por nombre desde
    # update_job — parchear uj.UPDATE_JOB_STATUS_FILE (fixture status_file)
    # ya es suficiente porque esas funciones leen/escriben el path en tiempo
    # de llamada, no lo capturan al importar.
    monkeypatch.setattr(main_module, "_update_executor", _NoopExecutor())
    return TestClient(main_module.app)


class _NoopExecutor:
    """Sustituye al ProcessPoolExecutor real en los tests de endpoint: no
    necesitamos que el trabajo se ejecute de verdad para probar el
    arranque/rechazo del endpoint, solo que se somete correctamente."""
    def __init__(self):
        self.submitted = []

    def submit(self, fn, *args):
        self.submitted.append((fn, args))

    def shutdown(self, wait=True, cancel_futures=False):
        pass   # por si TestClient dispara el shutdown del lifespan en algún flujo


class TestEndpoints:

    def test_start_returns_job_id_and_running(self, client):
        response = client.post("/corpus/update")
        assert response.status_code == 200
        body = response.json()
        assert body["state"] == "running"
        assert body["job_id"]

    def test_second_concurrent_start_rejected(self, client, status_file):
        uj.start_job_status("already-running")
        response = client.post("/corpus/update")
        assert response.status_code == 409

    def test_cooldown_rejects_recent_finish(self, client, status_file):
        recent = (datetime.now(timezone.utc) - timedelta(seconds=10)).isoformat()
        uj.write_job_status({"state": "completed", "finished_at": recent, "result": {}, "error": None, "log": []})
        response = client.post("/corpus/update")
        assert response.status_code == 429

    def test_start_allowed_after_cooldown_expires(self, client, status_file):
        old = (datetime.now(timezone.utc) - timedelta(seconds=999)).isoformat()
        uj.write_job_status({"state": "completed", "finished_at": old, "result": {}, "error": None, "log": []})
        response = client.post("/corpus/update")
        assert response.status_code == 200

    def test_status_endpoint_reflects_current_state(self, client, status_file):
        uj.start_job_status("job-5")
        response = client.get("/corpus/update/status")
        assert response.status_code == 200
        body = response.json()
        assert body["state"] == "running"
        assert body["job_id"] == "job-5"

    def test_status_endpoint_idle_by_default(self, client, status_file):
        response = client.get("/corpus/update/status")
        assert response.json()["state"] == "idle"


# ============================================================
# 3. La API no se bloquea mientras el job corre (proceso real)
# ============================================================

def _cpu_bound_busy_loop():
    """Simula el coste real del parseo XML: CPU-bound, sin I/O ni sleep, no
    libera el GIL si corriera en un thread del proceso actual."""
    total = 0
    for i in range(150_000_000):
        total += i
    return {"pmids_checked": 0, "files_processed": 1, "updated": 0, "deleted": 0}


def _fast_noop():
    return {"pmc_checked": 0, "reingested": 0, "retracted": 0, "removed_license": 0, "removed_gone": 0}


class TestApiAvailabilityDuringJob:

    def test_other_work_not_blocked_by_running_job(self, status_file, monkeypatch, tmp_path):
        """
        Verificación explícita pedida: mientras el job corre, el resto del
        sistema no debe congelarse. Se somete un trabajo CPU-bound real a un
        ProcessPoolExecutor real (mismo mecanismo que usa la API) y, durante
        su ejecución, se comprueba que un bucle CPU-bound equivalente en el
        proceso ACTUAL tarda aproximadamente lo mismo que si el job no
        estuviera corriendo — si compartieran GIL (p. ej. con un thread en
        vez de un proceso), este bucle se vería notablemente ralentizado.
        """
        monkeypatch.setattr(
            "src.data_actualization.pubmed_updater.run_pubmed_update", _cpu_bound_busy_loop
        )
        monkeypatch.setattr(
            "src.data_actualization.oa_updater.run_daily_update", _fast_noop
        )
        monkeypatch.setattr("src.config.LAST_CORPUS_UPDATE_FILE", tmp_path / "last_corpus_update.txt")

        # Línea base: cuánto tarda el bucle CPU-bound en solitario.
        baseline_start = time.time()
        _cpu_bound_busy_loop()
        baseline_duration = time.time() - baseline_start

        uj.start_job_status("job-concurrent")
        executor = ProcessPoolExecutor(max_workers=1, mp_context=get_context("fork"))
        try:
            executor.submit(uj.run_update_job, "job-concurrent")

            # Dar tiempo a que el proceso hijo arranque y entre en el bucle.
            time.sleep(0.3)
            assert uj.read_job_status()["state"] == "running"

            concurrent_start = time.time()
            _cpu_bound_busy_loop()
            concurrent_duration = time.time() - concurrent_start

            # Con un proceso aparte, el bucle en curso no debería tardar
            # sustancialmente más que en solitario (margen generoso: 2x,
            # para no ser un test frágil en máquinas con poca CPU disponible).
            assert concurrent_duration < baseline_duration * 2, (
                f"El bucle CPU-bound del proceso actual tardó {concurrent_duration:.2f}s "
                f"con el job corriendo vs {baseline_duration:.2f}s en solitario — "
                f"sugiere que el job SÍ está bloqueando este proceso."
            )

            deadline = time.time() + 15
            while uj.read_job_status().get("state") == "running" and time.time() < deadline:
                time.sleep(0.2)

            final = uj.read_job_status()
            assert final["state"] == "completed"
        finally:
            executor.shutdown(wait=True)
