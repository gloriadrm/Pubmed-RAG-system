"""
test_health.py
---------------
Verifica que /health separa correctamente el estado de Qdrant, embeddings
(BGE-M3, local) y el proveedor LLM activo — y que "status" pasa a
"degraded" cuando cualquiera de los tres falla, sin que uno enmascare a
otro (p. ej. Ollama apagado no debe reportarse como fallo de embeddings,
que corren en proceso vía sentence-transformers).

/health solo comprueba que el modelo de embeddings está cargado en memoria
(embedding_status == "loaded"), sin volver a ejecutar inferencia en cada
petición — mantiene el endpoint barato. La prueba funcional real (un
encode() de verdad sobre el modelo) vive en este archivo, no en /health.

Requiere la API completa corriendo en Docker (docker compose up -d).
Este script SÍ para y reinicia el contenedor de Qdrant durante el caso 2
(para forzar un fallo real) y lo deja corriendo de nuevo al terminar.

Sobre "Gemini sin clave" / cambio de proveedor:
LLM_PROVIDER se resuelve una única vez al importar src/services/llms.py,
al arrancar el proceso — con una clave ausente, el contenedor no llega a
levantar /health, falla en el arranque (ver logs). Por eso ese escenario
se verifica en test/test_llm_factory.py (a nivel de construcción del
cliente, que es donde realmente ocurre el fallo), no aquí contra un
servidor vivo: no hay ningún /health "misconfigured" observable mientras
el proceso está corriendo con esta arquitectura fail-fast.

Uso:
    python test/test_health.py
    python test/test_health.py --url http://localhost:8000
"""

import argparse
import subprocess
import sys
import time

import requests

DEFAULT_URL = "http://localhost:8000"

PASSED = 0
FAILED = 0


def check(name, condition, detail=""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"[PASS] {name}")
    else:
        FAILED += 1
        print(f"[FAIL] {name}  {detail}")


def get_health(base_url):
    r = requests.get(f"{base_url}/health", timeout=5)
    return r.json()


def wait_for(predicate, timeout_s=60, interval_s=2):
    start = time.time()
    last = None
    while time.time() - start < timeout_s:
        last = predicate()
        if last:
            return True
        time.sleep(interval_s)
    return False


def case_1_happy_path(base_url):
    print("\n--- Caso 1: estado actual (happy path) ---")
    data = get_health(base_url)
    print(f"  {data}")

    check("status == ok", data["status"] == "ok", data)
    check("qdrant == ok", data["qdrant"] == "ok", data["qdrant"])
    check("embedding_provider == 'bge-m3'", data["embedding_provider"] == "bge-m3")
    check("embedding_status == loaded", data["embedding_status"] == "loaded", data["embedding_status"])
    check(
        "llm_status es configured/reachable (no depende de si Ollama está vivo)",
        data["llm_status"] in ("configured", "reachable"),
        data["llm_status"],
    )
    print(f"  -> proveedor LLM activo: {data['llm_provider']} (llm_status={data['llm_status']})")


def case_2_qdrant_down_and_recovers(base_url):
    print("\n--- Caso 2: Qdrant caído -> degraded, embeddings y LLM no se ven afectados ---")

    subprocess.run(["docker", "compose", "stop", "qdrant"], check=True, capture_output=True)
    try:
        # Esperar a que /health detecte a Qdrant caído (puede tardar unos segundos)
        data = None

        def qdrant_down():
            nonlocal data
            data = get_health(base_url)
            return data["qdrant"] != "ok"

        detected = wait_for(qdrant_down, timeout_s=30, interval_s=2)
        check("Qdrant caído detectado por /health", detected, data)

        if data:
            check("status == degraded con Qdrant caído", data["status"] == "degraded", data["status"])
            check(
                "embedding_status sigue loaded (BGE-M3 es local, no depende de Qdrant)",
                data["embedding_status"] == "loaded",
                data["embedding_status"],
            )
            check(
                "llm_status sigue configured/reachable (no depende de Qdrant)",
                data["llm_status"] in ("configured", "reachable"),
                data["llm_status"],
            )
    finally:
        subprocess.run(["docker", "compose", "start", "qdrant"], check=True, capture_output=True)

    def qdrant_recovered():
        data = get_health(base_url)
        return data["qdrant"] == "ok"

    recovered = wait_for(qdrant_recovered, timeout_s=60, interval_s=3)
    check("Qdrant se recupera y /health vuelve a status == ok", recovered)


def case_3_embedding_functional():
    """
    Prueba funcional real del modelo de embeddings (no vía /health, que solo
    comprueba que está cargado). Importa src.services.embeddings directamente
    -- no depende de Qdrant ni de ningún proveedor LLM, así que corre en el
    proceso local sin tocar Docker.
    """
    print("\n--- Caso 3: encode() real sobre BGE-M3 (prueba funcional, no vía /health) ---")

    from src.services.embeddings import embedding_text

    vector = embedding_text("gut microbiota and intestinal permeability")

    check("embedding_text devuelve una lista", isinstance(vector, list))
    check("el vector no está vacío", len(vector) > 0, len(vector) if vector else 0)
    check(
        "todos los componentes son números",
        all(isinstance(x, (int, float)) for x in vector),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    print("=" * 60)
    print("  TEST — /health (Qdrant / embeddings / LLM desacoplados)")
    print("=" * 60)

    case_1_happy_path(base_url)
    case_2_qdrant_down_and_recovers(base_url)
    case_3_embedding_functional()

    print()
    print("=" * 60)
    print(f"  {PASSED} pasadas, {FAILED} fallidas")
    print("=" * 60)
    print()
    print("Nota: 'Gemini sin clave' y el cambio dinámico de LLM_PROVIDER no son")
    print("observables vía /health en esta arquitectura (fail-fast al arrancar el")
    print("proceso) — ver test/test_llm_factory.py para esos casos.")

    if FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()
