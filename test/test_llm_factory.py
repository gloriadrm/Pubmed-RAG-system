"""
test_llm_factory.py
--------------------
Verifica la selección de proveedor LLM (LLM_PROVIDER) en src/services/llms.py
SIN hacer llamadas reales a ninguna API: construir un ChatOpenAI, un
ChatGoogleGenerativeAI o un ChatOllama es una operación local (no dispara
red); solo `.invoke()` lo haría, y este script nunca lo llama.

Cada caso se ejecuta en un subproceso con un entorno controlado y mínimo
(no hereda el .env real del proyecto) para poder probar de forma aislada
qué pasa cuando faltan o sobran variables, sin que una credencial real ya
cargada en este proceso enmascare un caso de "falta la clave".

No requiere ninguna credencial real para pasar — usa valores dummy, ya que
solo se comprueba construcción del cliente y validación de variables, no
generación real.

Uso:
    python test/test_llm_factory.py
"""

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

PASSED = 0
FAILED = 0


def run_case(name, env_extra, expect_ok, expect_in_output=None):
    global PASSED, FAILED

    env = {"PATH": "/usr/bin:/bin"}
    env.update(env_extra)

    proc = subprocess.run(
        [PYTHON, "-c", "from src.services.llms import llm_langchain; print(type(llm_langchain).__name__)"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    ok = (proc.returncode == 0)
    output = proc.stdout + proc.stderr

    status_ok = (ok == expect_ok)
    text_ok = (expect_in_output is None) or (expect_in_output in output)

    if status_ok and text_ok:
        PASSED += 1
        print(f"[PASS] {name}")
    else:
        FAILED += 1
        print(f"[FAIL] {name}")
        print(f"       esperado: ok={expect_ok}, contiene={expect_in_output!r}")
        print(f"       obtenido: returncode={proc.returncode}")
        print(f"       output:\n{output.strip()[-800:]}")


def main():
    print("=" * 60)
    print("  TEST — Factoría de proveedores LLM (get_llm)")
    print("=" * 60)
    print()

    # --- Caso 1: openai seleccionado, clave presente ---
    run_case(
        "openai seleccionado con OPENAI_API_KEY -> ChatOpenAI",
        {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-dummy-for-construction-only"},
        expect_ok=True,
        expect_in_output="ChatOpenAI",
    )

    # --- Caso 2: gemini seleccionado, GEMINI_API_KEY presente ---
    run_case(
        "gemini seleccionado con GEMINI_API_KEY -> ChatGoogleGenerativeAI",
        {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "dummy-gemini-key"},
        expect_ok=True,
        expect_in_output="ChatGoogleGenerativeAI",
    )

    # --- Caso 3: gemini seleccionado, solo el alias heredado GOOGLE_API_KEY ---
    run_case(
        "gemini seleccionado con solo GOOGLE_API_KEY (alias heredado) -> ChatGoogleGenerativeAI",
        {"LLM_PROVIDER": "gemini", "GOOGLE_API_KEY": "dummy-google-key"},
        expect_ok=True,
        expect_in_output="ChatGoogleGenerativeAI",
    )

    # --- Caso 4: ollama seleccionado, sin ninguna API key -> ChatOllama ---
    run_case(
        "ollama seleccionado sin API key -> ChatOllama",
        {"LLM_PROVIDER": "ollama"},
        expect_ok=True,
        expect_in_output="ChatOllama",
    )

    # --- Caso 5: proveedor no soportado -> error claro ---
    run_case(
        "proveedor no soportado ('bogus') -> ValueError listando los válidos",
        {"LLM_PROVIDER": "bogus"},
        expect_ok=False,
        expect_in_output="gemini, ollama, openai",  # sorted() en get_llm()
    )

    # --- Caso 6: falta la variable requerida (openai sin OPENAI_API_KEY) ---
    run_case(
        "openai seleccionado SIN OPENAI_API_KEY -> error nombrando la variable",
        {"LLM_PROVIDER": "openai"},
        expect_ok=False,
        expect_in_output="OPENAI_API_KEY",
    )

    # --- Caso 7: falta la variable requerida (gemini sin ninguna clave) ---
    run_case(
        "gemini seleccionado SIN GEMINI_API_KEY ni GOOGLE_API_KEY -> error nombrando ambas",
        {"LLM_PROVIDER": "gemini"},
        expect_ok=False,
        expect_in_output="GEMINI_API_KEY o GOOGLE_API_KEY",
    )

    # --- Caso 8: independencia entre proveedores — openai no exige clave de gemini ---
    run_case(
        "openai seleccionado con clave de gemini presente pero SIN clave propia -> igual falla pidiendo OPENAI_API_KEY",
        {"LLM_PROVIDER": "openai", "GEMINI_API_KEY": "dummy-gemini-key"},
        expect_ok=False,
        expect_in_output="OPENAI_API_KEY",
    )

    # --- Caso 9: independencia entre proveedores — gemini no exige clave de openai ---
    run_case(
        "gemini seleccionado con OPENAI_API_KEY presente pero sin clave propia de gemini -> igual falla",
        {"LLM_PROVIDER": "gemini", "OPENAI_API_KEY": "sk-dummy"},
        expect_ok=False,
        expect_in_output="GEMINI_API_KEY o GOOGLE_API_KEY",
    )

    # --- Caso 10: openai funciona con clave propia aunque NO haya clave de gemini ---
    run_case(
        "openai seleccionado, funciona sin ninguna clave de gemini presente",
        {"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-dummy"},
        expect_ok=True,
        expect_in_output="ChatOpenAI",
    )

    print()
    print("=" * 60)
    print(f"  {PASSED} pasadas, {FAILED} fallidas")
    print("=" * 60)

    if FAILED:
        sys.exit(1)


if __name__ == "__main__":
    main()
