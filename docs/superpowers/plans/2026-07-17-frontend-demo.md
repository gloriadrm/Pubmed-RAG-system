# Frontend de demo — Biomedical RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Añadir una interfaz web estática (HTML/CSS/JS vanilla) servida por el propio FastAPI para poder hacer demos en vivo del sistema RAG biomédico sin depender de `curl`/`/docs`.

**Architecture:** Página estática sin build step, montada con `StaticFiles` en `/` después de las rutas de la API. Dos cambios de contrato en el backend (`/health.llm` en vez de `.ollama`, `SourceDoc.pm_id` nuevo) para que la UI pueda mostrar el estado real del LLM activo y enlazar cada fuente citada a PubMed/PMC.

**Tech Stack:** FastAPI (`StaticFiles`), HTML/CSS/JS vanilla con ES modules (sin bundler, sin framework), Docker/docker-compose existentes.

## Global Constraints

- No añadir dependencias nuevas (ni de Python ni de JS) — todo con lo que ya está en `pyproject.toml` y navegador estándar.
- No añadir contenedores ni servicios nuevos a `docker-compose.yml`.
- El proyecto no tiene suite pytest ni tests unitarios para la API — los tests existentes (`test/test_api.py`, `test/test_indexing.py`, etc.) son scripts manuales que se ejecutan contra el stack Docker en marcha y se verifican leyendo su output. Este plan sigue esa misma convención: cada tarea se verifica con comandos reales (`curl`, el propio navegador) contra el stack corriendo, no con pytest.
- Todo el spec de diseño está en `docs/superpowers/specs/2026-07-17-frontend-demo-design.md` — cualquier duda sobre comportamiento esperado se resuelve ahí.
- Mensajes de UI en español, consistente con el resto de la documentación del proyecto.

---

## Prerrequisito único (una sola vez, antes de la Tarea 1)

El stack debe estar levantado y con datos indexados para poder verificar `/query` con resultados reales durante las tareas siguientes.

```bash
cd "/Users/gloriadelriomarquez/Documents/Career/Master Pontia/10. BD vectoriales/Entrega/project"
docker compose up --build -d
docker logs biomedical_rag_api -f
```

Espera a ver `Application startup complete.` (Ctrl+C para salir de los logs, los contenedores siguen corriendo). Después, siembra el corpus de prueba (recrea la colección desde cero, indexa 10 artículos sobre microbiota intestinal):

```bash
python test/test_indexing.py
```

Esto tarda unos minutos la primera vez. No hace falta repetirlo entre tareas — los datos persisten en `./qdrant_data`.

---

### Task 1: Backend — `/health` reporta el LLM activo, `/query` expone `pm_id`

**Files:**
- Modify: `src/api/schema.py` (líneas 18-46)
- Modify: `src/api/main.py` (líneas 15-56, 90-99)
- Test: verificación manual con `curl` contra el stack Docker

**Interfaces:**
- Produces: `HealthResponse.llm: str` (antes `.ollama`), `SourceDoc.pm_id: Optional[str]` — usados por el frontend en las Tareas 4 y 5.

- [ ] **Step 1: Modificar `HealthResponse` y `SourceDoc` en el schema**

En `src/api/schema.py`, reemplaza el bloque de `SourceDoc` (líneas 18-22):

```python
class SourceDoc(BaseModel):
    title:   str
    section: Optional[str] = None
    pmc_id:  Optional[str] = None
    pm_id:   Optional[str] = None   # PMID de PubMed (siempre presente en docs source=pubmed)
    source:  str            # "pubmed" | "pmc"
```

Y el bloque de `HealthResponse` (líneas 43-46):

```python
class HealthResponse(BaseModel):
    status: str
    qdrant: str
    llm:    str
```

- [ ] **Step 2: Reescribir `health()` para reportar el proveedor LLM activo en vez de Ollama**

En `src/api/main.py`, añade `import os` junto a los imports existentes (línea 15):

```python
import os
import requests as http_requests
from fastapi import FastAPI, HTTPException
```

Reemplaza la función `health()` completa (líneas 35-55):

```python
@app.get("/health", response_model=HealthResponse)
def health():
    """Comprueba que Qdrant está accesible y que el proveedor LLM activo está configurado."""

    # Qdrant
    try:
        r = http_requests.get(f"{QDRANT_URL}/healthz", timeout=3)
        qdrant_status = "ok" if r.status_code == 200 else f"error {r.status_code}"
    except Exception as e:
        qdrant_status = f"unreachable ({e})"

    # LLM activo — actualmente Gemini (ver src/services/llms.py). Ollama está
    # comentado y no es el proveedor en uso, por eso no se comprueba aquí.
    if os.environ.get("GOOGLE_API_KEY"):
        llm_status = "gemini"
    else:
        llm_status = "gemini (missing GOOGLE_API_KEY)"

    overall = "ok" if qdrant_status == "ok" and llm_status == "gemini" else "degraded"

    return HealthResponse(status=overall, qdrant=qdrant_status, llm=llm_status)
```

- [ ] **Step 3: Poblar `pm_id` en la respuesta de `/query`**

En `src/api/main.py`, dentro de `query()`, localiza el bucle que construye las fuentes (líneas 91-99):

```python
    sources = []
    for doc in result.get("source_context") or []:
        meta = doc.metadata
        sources.append(SourceDoc(
            title=meta.get("title", ""),
            section=meta.get("section"),
            pmc_id=meta.get("pmc_id"),
            source=meta.get("source", ""),
        ))
```

Reemplázalo por:

```python
    sources = []
    for doc in result.get("source_context") or []:
        meta = doc.metadata
        sources.append(SourceDoc(
            title=meta.get("title", ""),
            section=meta.get("section"),
            pmc_id=meta.get("pmc_id"),
            pm_id=meta.get("pm_id"),
            source=meta.get("source", ""),
        ))
```

- [ ] **Step 4: Reconstruir el contenedor y verificar**

```bash
docker compose up --build -d api
curl -s http://localhost:8000/health | python3 -m json.tool
```

Expected (con `GOOGLE_API_KEY` configurada en `.env`):
```json
{
    "status": "ok",
    "qdrant": "ok",
    "llm": "gemini"
}
```

```bash
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the current evidence on gut microbiota and intestinal permeability?"}' \
  | python3 -m json.tool
```

Expected: `query_type: "thematic_summary"`, y cada objeto en `sources` incluye ahora la clave `"pm_id"` (con valor string, no `null`, para fuentes `"source": "pubmed"`).

- [ ] **Step 5: Commit**

```bash
git add src/api/schema.py src/api/main.py
git commit -m "feat: report active LLM provider in /health, expose pm_id in /query sources"
```

---

### Task 2: Backend — montar `static/` como frontend servido por FastAPI

**Files:**
- Create: `static/index.html` (placeholder mínimo, se reemplaza en la Tarea 3)
- Modify: `src/api/main.py` (añadir mount al final del archivo)
- Modify: `Dockerfile` (línea 13)
- Modify: `docker-compose.yml` (bloque `volumes` del servicio `api`, líneas 26-30)

**Interfaces:**
- Produces: `http://localhost:8000/` sirve `static/index.html`. Cambios posteriores en `static/*` se reflejan sin rebuild (volumen montado), solo hace falta recargar el navegador.

- [ ] **Step 1: Crear placeholder**

Crea `static/index.html`:

```html
<!DOCTYPE html>
<html lang="es">
<head><meta charset="UTF-8"><title>Biomedical RAG — Demo</title></head>
<body><p>placeholder</p></body>
</html>
```

- [ ] **Step 2: Montar `StaticFiles` en `main.py`**

En `src/api/main.py`, añade el import junto a los demás (tras `from fastapi import FastAPI, HTTPException`):

```python
from fastapi.staticfiles import StaticFiles
```

Al final del archivo (después de la función `ingest()`, línea 130), añade:

```python

# -------------- FRONTEND ESTÁTICO --------------
# Debe montarse el último: como catch-all en "/", cualquier ruta de API
# registrada arriba (/health, /query, /ingest, /docs) tiene prioridad.
app.mount("/", StaticFiles(directory="static", html=True), name="static")
```

- [ ] **Step 3: Copiar `static/` en la imagen Docker**

En `Dockerfile`, tras la línea `COPY src/ ./src/` (línea 13), añade:

```dockerfile
COPY static/ ./static/
```

- [ ] **Step 4: Montar `static/` como volumen para iterar sin rebuild**

En `docker-compose.yml`, en el bloque `volumes` del servicio `api` (líneas 26-30), añade una línea:

```yaml
    volumes:
      # Datos persistentes (OA CSV, last_update tracker)
      - ./data:/app/data
      # Frontend estático — montado en modo lectura para poder iterar sin rebuild
      - ./static:/app/static:ro
      # Caché del modelo bge-m3 — evita re-descargarlo en cada rebuild
      - huggingface_cache:/root/.cache/huggingface
```

- [ ] **Step 5: Reconstruir y verificar**

```bash
docker compose up --build -d api
curl -s http://localhost:8000/ | grep -o "<title>.*</title>"
```

Expected: `<title>Biomedical RAG — Demo</title>`

```bash
curl -s http://localhost:8000/health | python3 -m json.tool
```

Expected: sigue devolviendo `status/qdrant/llm` con normalidad (confirma que el mount en `/` no rompe las rutas de la API).

- [ ] **Step 6: Commit**

```bash
git add static/index.html src/api/main.py Dockerfile docker-compose.yml
git commit -m "feat: serve static frontend from FastAPI at /"
```

---

### Task 3: Frontend — estructura HTML, estilos y pestañas

**Files:**
- Modify: `static/index.html` (reemplaza el placeholder completo)
- Create: `static/style.css`
- Create: `static/app.js` (solo lógica de pestañas por ahora)

**Interfaces:**
- Produces: elementos DOM con estos IDs, que las Tareas 4-6 usan: `#health-refresh`, `#health-api .dot`, `#health-qdrant .dot`, `#health-llm .dot`, `#query-form`, `#question`, `#query-submit`, `#query-result`, `#query-error`, `#ingest-form`, `#ingest-query`, `#ingest-n`, `#ingest-submit`, `#ingest-status`, `#ingest-error`, `#ingest-loading`. Clase CSS `.chip` con atributo `data-example` para los chips de ejemplo.

- [ ] **Step 1: Escribir `static/index.html` completo**

```html
<!DOCTYPE html>
<html lang="es">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Biomedical RAG — Demo</title>
  <link rel="stylesheet" href="style.css">
</head>
<body>
  <header class="site-header">
    <h1>Biomedical RAG</h1>
    <div class="health" id="health">
      <span class="health-item" id="health-api">API <span class="dot" data-state="unknown"></span></span>
      <span class="health-item" id="health-qdrant">Qdrant <span class="dot" data-state="unknown"></span></span>
      <span class="health-item" id="health-llm">LLM <span class="dot" data-state="unknown"></span></span>
      <button class="refresh-btn" id="health-refresh" type="button" title="Actualizar estado">↻</button>
    </div>
  </header>

  <main>
    <nav class="tabs" role="tablist">
      <button class="tab-btn active" data-tab="query" role="tab" aria-selected="true">Consultar</button>
      <button class="tab-btn" data-tab="ingest" role="tab" aria-selected="false">Ingestar</button>
    </nav>

    <section id="tab-query" class="tab-panel" role="tabpanel">
      <form id="query-form">
        <label for="question">Pregunta</label>
        <textarea id="question" name="question" rows="3"
                  placeholder="Escribe una pregunta sobre el corpus biomédico indexado…" required></textarea>

        <div class="examples">
          <span class="examples-label">Ejemplos:</span>
          <button type="button" class="chip"
                  data-example="Resumen general sobre microbiota y permeabilidad intestinal.">
            Resumen general sobre microbiota y permeabilidad intestinal
          </button>
          <button type="button" class="chip"
                  data-example="¿Qué metodología utiliza el estudio de [autor/título]?">
            ¿Qué metodología utiliza el estudio de [autor/título]?
          </button>
          <button type="button" class="chip"
                  data-example="¿Qué mecanismos inflamatorios aparecen en varios estudios?">
            ¿Qué mecanismos inflamatorios aparecen en varios estudios?
          </button>
        </div>

        <button type="submit" id="query-submit">Preguntar</button>
      </form>

      <div id="query-result" class="result" hidden></div>
      <div id="query-error" class="error-box" hidden></div>
    </section>

    <section id="tab-ingest" class="tab-panel" role="tabpanel" hidden>
      <p class="warning">
        ⚠ La ingesta puede tardar varios minutos (búsqueda en PubMed, cruce con PMC,
        chunking e indexación). No cierres esta pestaña mientras esté en curso.
      </p>
      <form id="ingest-form">
        <label for="ingest-query">Query temática</label>
        <input type="text" id="ingest-query" name="query"
               placeholder="p. ej. gut microbiota intestinal permeability" required>

        <label for="ingest-n">Número de artículos</label>
        <input type="number" id="ingest-n" name="n" value="10" min="1" max="100">

        <button type="submit" id="ingest-submit">Ingestar</button>
      </form>

      <div id="ingest-loading" class="loading" hidden>
        <span class="spinner" aria-hidden="true"></span>
        Ingestando… puede tardar varios minutos, no cierres esta pestaña.
      </div>
      <div id="ingest-status" class="status" hidden></div>
      <div id="ingest-error" class="error-box" hidden></div>
    </section>
  </main>

  <script type="module" src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Escribir `static/style.css`**

```css
:root {
  --fg: #111;
  --bg: #fff;
  --muted: #666;
  --border: #e5e5e5;
  --chip-bg: #f2f2f2;
  --accent: #111;
  --accent-fg: #fff;
  --ok: #1a7f37;
  --error: #c1121f;
  --unknown: #999;
}

* { box-sizing: border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--fg);
  background: var(--bg);
  margin: 0;
  padding: 0 16px 48px;
  max-width: 720px;
  margin-inline: auto;
}

.site-header {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  flex-wrap: wrap;
  gap: 8px;
  padding: 24px 0 12px;
}

.site-header h1 { font-size: 20px; margin: 0; }

.health {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  color: var(--muted);
}

.health-item { display: flex; align-items: center; gap: 4px; }

.dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--unknown);
}
.dot[data-state="ok"] { background: var(--ok); }
.dot[data-state="error"] { background: var(--error); }
.dot[data-state="unknown"] { background: var(--unknown); }

.refresh-btn {
  border: none;
  background: none;
  cursor: pointer;
  font-size: 14px;
  color: var(--muted);
  padding: 2px 4px;
}
.refresh-btn:hover { color: var(--fg); }

.tabs {
  display: flex;
  gap: 4px;
  border-bottom: 1px solid var(--border);
  margin-bottom: 20px;
}

.tab-btn {
  border: none;
  background: none;
  padding: 10px 14px;
  font-size: 14px;
  cursor: pointer;
  color: var(--muted);
  border-bottom: 2px solid transparent;
}
.tab-btn.active {
  color: var(--fg);
  font-weight: 600;
  border-bottom-color: var(--fg);
}

.tab-panel[hidden] { display: none; }

label {
  display: block;
  font-size: 13px;
  font-weight: 600;
  margin: 14px 0 6px;
}

textarea, input[type="text"], input[type="number"] {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border);
  border-radius: 10px;
  font-size: 14px;
  font-family: inherit;
  background: var(--bg);
  color: var(--fg);
}
textarea { resize: vertical; }

.examples {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 10px;
  align-items: center;
}
.examples-label { font-size: 12px; color: var(--muted); margin-right: 4px; }

.chip {
  border: 1px solid var(--border);
  background: var(--chip-bg);
  color: var(--fg);
  border-radius: 16px;
  padding: 6px 12px;
  font-size: 12px;
  cursor: pointer;
}
.chip:hover { background: #e8e8e8; }

button[type="submit"] {
  margin-top: 16px;
  border: none;
  background: var(--accent);
  color: var(--accent-fg);
  padding: 10px 18px;
  border-radius: 20px;
  font-size: 14px;
  font-weight: 600;
  cursor: pointer;
}
button[type="submit"]:disabled { opacity: 0.6; cursor: default; }

.warning {
  background: #fff8e6;
  border: 1px solid #f0dca0;
  border-radius: 10px;
  padding: 10px 14px;
  font-size: 13px;
  color: #6b5200;
}

.loading {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-top: 16px;
  font-size: 13px;
  color: var(--muted);
}

.spinner {
  width: 14px;
  height: 14px;
  border: 2px solid var(--border);
  border-top-color: var(--fg);
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  flex-shrink: 0;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.result, .status {
  margin-top: 20px;
  padding: 16px;
  border: 1px solid var(--border);
  border-radius: 10px;
}

.badge {
  display: inline-block;
  font-size: 11px;
  padding: 3px 10px;
  border-radius: 10px;
  background: var(--chip-bg);
  color: var(--fg);
  font-weight: 600;
}

.answer { margin-top: 10px; line-height: 1.5; white-space: pre-wrap; }

.sources { list-style: none; padding: 0; margin: 8px 0 0; }
.sources li { padding: 8px 0; border-top: 1px solid var(--border); font-size: 13px; }
.sources a { color: var(--fg); text-decoration: underline; }
.source-meta { color: var(--muted); }

.error-box {
  margin-top: 20px;
  padding: 14px 16px;
  border: 1px solid #f3b8b8;
  background: #fff5f5;
  border-radius: 10px;
  color: #7a1212;
  font-size: 13px;
}
.error-box pre { white-space: pre-wrap; font-size: 12px; margin-top: 8px; }
```

- [ ] **Step 3: Escribir `static/app.js` (solo pestañas por ahora)**

```javascript
function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;

      buttons.forEach((b) => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-selected", b === btn ? "true" : "false");
      });

      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.hidden = panel.id !== `tab-${target}`;
      });
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
});
```

- [ ] **Step 4: Verificar en el navegador**

Los archivos están montados como volumen (Tarea 2), así que no hace falta rebuild — basta recargar.

Abre `http://localhost:8000/` en el navegador. Verifica:
- Se ven la cabecera, los 3 puntos de estado (grises, `data-state="unknown"`) y el botón `↻`.
- La pestaña "Consultar" está activa por defecto, con el textarea, los 3 chips de ejemplo y el botón "Preguntar".
- Al hacer clic en "Ingestar", cambia el panel visible (aviso + formulario de ingesta) y el botón activo.
- Estilo: fondo blanco, acentos grises, botones con esquinas redondeadas (paleta minimalista).

- [ ] **Step 5: Commit**

```bash
git add static/index.html static/style.css static/app.js
git commit -m "feat: build demo UI markup, styles and tab switching"
```

---

### Task 4: Frontend — cliente API y badge de salud

**Files:**
- Create: `static/api.js`
- Create: `static/health.js`
- Modify: `static/app.js`

**Interfaces:**
- Consumes: endpoints `GET /health`, `POST /query`, `POST /ingest` (Task 1); IDs `#health-refresh`, `#health-api`, `#health-qdrant`, `#health-llm` (Task 3).
- Produces: `api.js` exporta `getHealth()`, `postQuery(question)`, `postIngest(query, n)` — usados por `query.js` (Task 5) e `ingest.js` (Task 6). `health.js` exporta `initHealth()`.

- [ ] **Step 1: Escribir `static/api.js`**

```javascript
async function parseErrorDetail(response) {
  try {
    const data = await response.json();
    return data.detail || JSON.stringify(data);
  } catch {
    return response.statusText;
  }
}

export async function getHealth() {
  const response = await fetch("/health");
  if (!response.ok) throw new Error(await parseErrorDetail(response));
  return response.json();
}

export async function postQuery(question) {
  const response = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  if (!response.ok) throw new Error(await parseErrorDetail(response));
  return response.json();
}

export async function postIngest(query, n) {
  const response = await fetch("/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, n }),
  });
  if (!response.ok) throw new Error(await parseErrorDetail(response));
  return response.json();
}
```

- [ ] **Step 2: Escribir `static/health.js`**

```javascript
import { getHealth } from "./api.js";

function setDot(containerId, state) {
  document.querySelector(`#${containerId} .dot`).dataset.state = state;
}

async function refreshHealth() {
  setDot("health-api", "unknown");
  setDot("health-qdrant", "unknown");
  setDot("health-llm", "unknown");

  try {
    const data = await getHealth();
    setDot("health-api", "ok");
    setDot("health-qdrant", data.qdrant === "ok" ? "ok" : "error");
    setDot("health-llm", data.llm === "gemini" ? "ok" : "error");
  } catch {
    setDot("health-api", "error");
    setDot("health-qdrant", "error");
    setDot("health-llm", "error");
  }
}

export function initHealth() {
  document.getElementById("health-refresh").addEventListener("click", refreshHealth);
  refreshHealth();
}
```

- [ ] **Step 3: Conectar en `app.js`**

Reemplaza `static/app.js` completo:

```javascript
import { initHealth } from "./health.js";

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;

      buttons.forEach((b) => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-selected", b === btn ? "true" : "false");
      });

      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.hidden = panel.id !== `tab-${target}`;
      });
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initHealth();
});
```

- [ ] **Step 4: Verificar en el navegador**

Recarga `http://localhost:8000/`. Verifica:
- Los 3 puntos pasan de gris a verde poco después de cargar (con el stack sano tras el Prerrequisito).
- Al pulsar `↻`, los puntos vuelven a gris un instante y luego a verde de nuevo.
- Para confirmar el camino de error: `docker compose stop qdrant`, recarga la página → el punto de Qdrant debe ponerse rojo y el `status` global sería `degraded`. Recuerda volver a levantarlo: `docker compose start qdrant`.

- [ ] **Step 5: Commit**

```bash
git add static/api.js static/health.js static/app.js
git commit -m "feat: add API client and health badge"
```

---

### Task 5: Frontend — pestaña Consultar

**Files:**
- Create: `static/query.js`
- Modify: `static/app.js`

**Interfaces:**
- Consumes: `postQuery(question)` de `api.js` (Task 4); IDs `#query-form`, `#question`, `#query-submit`, `#query-result`, `#query-error`, `.chip[data-example]` (Task 3).

- [ ] **Step 1: Escribir `static/query.js`**

```javascript
import { postQuery } from "./api.js";

function pmcUrl(pmcId) {
  return `https://www.ncbi.nlm.nih.gov/pmc/articles/${pmcId}/`;
}

function pubmedUrl(pmId) {
  return `https://pubmed.ncbi.nlm.nih.gov/${pmId}/`;
}

function renderSource(source) {
  const li = document.createElement("li");

  let url = null;
  let idLabel = null;
  if (source.source === "pmc" && source.pmc_id) {
    url = pmcUrl(source.pmc_id);
    idLabel = source.pmc_id;
  } else if (source.source === "pubmed" && source.pm_id) {
    url = pubmedUrl(source.pm_id);
    idLabel = source.pm_id;
  }

  const titleEl = document.createElement(url ? "a" : "span");
  titleEl.textContent = source.title;
  if (url) {
    titleEl.href = url;
    titleEl.target = "_blank";
    titleEl.rel = "noopener noreferrer";
  }
  li.appendChild(titleEl);

  const metaParts = [];
  if (idLabel) metaParts.push(idLabel);
  if (source.section) metaParts.push(source.section);
  metaParts.push(source.source);

  const meta = document.createElement("span");
  meta.className = "source-meta";
  meta.textContent = " — " + metaParts.join(" · ");
  li.appendChild(meta);

  return li;
}

function renderResult(result) {
  const container = document.getElementById("query-result");
  container.innerHTML = "";
  container.hidden = false;

  const badge = document.createElement("span");
  badge.className = "badge";
  badge.textContent = result.query_type;
  container.appendChild(badge);

  const answer = document.createElement("p");
  answer.className = "answer";
  answer.textContent = result.answer;
  container.appendChild(answer);

  if (result.sources.length > 0) {
    const title = document.createElement("h3");
    title.textContent = "Fuentes";
    container.appendChild(title);

    const list = document.createElement("ul");
    list.className = "sources";
    result.sources.forEach((s) => list.appendChild(renderSource(s)));
    container.appendChild(list);
  }
}

function renderError(message) {
  const box = document.getElementById("query-error");
  box.hidden = false;
  box.innerHTML = "";

  const p = document.createElement("p");
  p.textContent = "Ha ocurrido un error al procesar la consulta.";
  box.appendChild(p);

  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Ver detalle técnico";
  details.appendChild(summary);
  const pre = document.createElement("pre");
  pre.textContent = message;
  details.appendChild(pre);
  box.appendChild(details);
}

function clearMessages() {
  document.getElementById("query-result").hidden = true;
  document.getElementById("query-error").hidden = true;
}

export function initQuery() {
  const form = document.getElementById("query-form");
  const textarea = document.getElementById("question");
  const submitBtn = document.getElementById("query-submit");

  document.querySelectorAll("#tab-query .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      textarea.value = chip.dataset.example;
      textarea.focus();
    });
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearMessages();

    submitBtn.disabled = true;
    submitBtn.textContent = "Consultando…";

    try {
      const result = await postQuery(textarea.value.trim());
      renderResult(result);
    } catch (e) {
      renderError(e.message);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = "Preguntar";
    }
  });
}
```

- [ ] **Step 2: Conectar en `app.js`**

En `static/app.js`, añade el import junto a `initHealth` y la llamada dentro de `DOMContentLoaded`:

```javascript
import { initHealth } from "./health.js";
import { initQuery } from "./query.js";

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;

      buttons.forEach((b) => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-selected", b === btn ? "true" : "false");
      });

      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.hidden = panel.id !== `tab-${target}`;
      });
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initHealth();
  initQuery();
});
```

- [ ] **Step 3: Verificar en el navegador**

Recarga `http://localhost:8000/`. Con el corpus ya sembrado (Prerrequisito):

1. Clic en el primer chip de ejemplo → el textarea se rellena. Clic en "Preguntar". El botón debe decir "Consultando…" y deshabilitarse; al recibir respuesta, debe verse el badge `thematic_summary`, el texto de la respuesta y una lista de fuentes con `source: pubmed`, cada una como enlace a `pubmed.ncbi.nlm.nih.gov`.
2. Clic en el tercer chip ("mecanismos inflamatorios…") → debe dar `transversal_query` con fuentes `pmc` enlazando a `ncbi.nlm.nih.gov/pmc/articles/`.
3. Escribe una pregunta fuera de dominio, p. ej. "What is the capital of France?" → debe dar `query_type: "none"` y una respuesta indicando que está fuera de alcance, sin fuentes.
4. Para probar el camino de error: `docker compose stop qdrant`, lanza cualquier pregunta → debe aparecer el mensaje "Ha ocurrido un error al procesar la consulta." con un desplegable "Ver detalle técnico". Vuelve a levantar Qdrant: `docker compose start qdrant`.

- [ ] **Step 4: Commit**

```bash
git add static/query.js static/app.js
git commit -m "feat: implement Consultar tab with example chips and linked sources"
```

---

### Task 6: Frontend — pestaña Ingestar

**Files:**
- Create: `static/ingest.js`
- Modify: `static/app.js`

**Interfaces:**
- Consumes: `postIngest(query, n)` de `api.js` (Task 4); IDs `#ingest-form`, `#ingest-query`, `#ingest-n`, `#ingest-submit`, `#ingest-status`, `#ingest-error` (Task 3).

- [ ] **Step 1: Escribir `static/ingest.js`**

```javascript
import { postIngest } from "./api.js";

function clearMessages() {
  document.getElementById("ingest-status").hidden = true;
  document.getElementById("ingest-error").hidden = true;
}

function setLoading(isLoading) {
  document.getElementById("ingest-loading").hidden = !isLoading;
}

function renderSuccess(message) {
  const box = document.getElementById("ingest-status");
  box.hidden = false;
  box.textContent = message;
}

function renderError(message) {
  const box = document.getElementById("ingest-error");
  box.hidden = false;
  box.innerHTML = "";

  const p = document.createElement("p");
  p.textContent = "Ha ocurrido un error al lanzar la ingesta.";
  box.appendChild(p);

  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = "Ver detalle técnico";
  details.appendChild(summary);
  const pre = document.createElement("pre");
  pre.textContent = message;
  details.appendChild(pre);
  box.appendChild(details);
}

export function initIngest() {
  const form = document.getElementById("ingest-form");
  const submitBtn = document.getElementById("ingest-submit");
  const queryInput = document.getElementById("ingest-query");
  const nInput = document.getElementById("ingest-n");

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (submitBtn.disabled) return; // evita envíos duplicados
    clearMessages();

    submitBtn.disabled = true;
    setLoading(true);

    try {
      const result = await postIngest(queryInput.value.trim(), Number(nInput.value));
      renderSuccess(result.message);
    } catch (e) {
      renderError(e.message);
    } finally {
      submitBtn.disabled = false;
      setLoading(false);
    }
  });
}
```

- [ ] **Step 2: Conectar en `app.js`**

En `static/app.js`, añade el import y la llamada:

```javascript
import { initHealth } from "./health.js";
import { initQuery } from "./query.js";
import { initIngest } from "./ingest.js";

function initTabs() {
  const buttons = document.querySelectorAll(".tab-btn");

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const target = btn.dataset.tab;

      buttons.forEach((b) => {
        b.classList.toggle("active", b === btn);
        b.setAttribute("aria-selected", b === btn ? "true" : "false");
      });

      document.querySelectorAll(".tab-panel").forEach((panel) => {
        panel.hidden = panel.id !== `tab-${target}`;
      });
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initHealth();
  initQuery();
  initIngest();
});
```

- [ ] **Step 3: Verificar en el navegador**

Recarga `http://localhost:8000/` y ve a la pestaña "Ingestar".

1. Deja el campo query vacío y pulsa "Ingestar" → el navegador debe bloquear el envío (`required` del input HTML).
2. Escribe `probiotics colorectal cancer`, deja `n=10` (o baja a 2-3 para que la verificación sea rápida) y pulsa "Ingestar" → el botón debe deshabilitarse y debe aparecer el bloque con el spinner girando y el texto "Ingestando… puede tardar varios minutos, no cierres esta pestaña."; haz clic varias veces sobre el botón mientras está deshabilitado para confirmar que no se disparan envíos duplicados. Al terminar, el spinner desaparece, debe aparecer el mensaje de éxito con el texto devuelto por la API y el botón vuelve a estar habilitado.
3. Para el camino de error: `docker compose stop qdrant`, intenta ingestar → debe aparecer "Ha ocurrido un error al lanzar la ingesta." con detalle colapsado. Vuelve a levantar Qdrant: `docker compose start qdrant`.

- [ ] **Step 4: Commit**

```bash
git add static/ingest.js static/app.js
git commit -m "feat: implement Ingestar tab with duplicate-submit guard"
```

---

### Task 7: Documentación y verificación end-to-end

**Files:**
- Modify: `test/test_api.py` (líneas 25-33)
- Modify: `README.md` (líneas 93-96 y nueva sección tras "Uso de la API")

**Interfaces:** Ninguna nueva — cierre del plan.

- [ ] **Step 1: Actualizar `check_health()` en `test/test_api.py`**

Reemplaza (líneas 25-33):

```python
def check_health(base_url: str):
    r = requests.get(f"{base_url}/health", timeout=5)
    data = r.json()
    print(f"  qdrant : {data['qdrant']}")
    print(f"  ollama : {data['ollama']}")
    print(f"  status : {data['status']}")
    if data["status"] != "ok":
        print("⚠ Algún servicio no está disponible. Continúa de todas formas...\n")
    return data["status"] == "ok"
```

por:

```python
def check_health(base_url: str):
    r = requests.get(f"{base_url}/health", timeout=5)
    data = r.json()
    print(f"  qdrant : {data['qdrant']}")
    print(f"  llm    : {data['llm']}")
    print(f"  status : {data['status']}")
    if data["status"] != "ok":
        print("⚠ Algún servicio no está disponible. Continúa de todas formas...\n")
    return data["status"] == "ok"
```

- [ ] **Step 2: Actualizar el ejemplo de respuesta de `/health` en el README**

En `README.md`, reemplaza (líneas 93-96):

```markdown
Respuesta esperada:
```json
{"status": "ok", "qdrant": "ok", "ollama": "ok"}
```
```

por:

```markdown
Respuesta esperada:
```json
{"status": "ok", "qdrant": "ok", "llm": "gemini"}
```
```

- [ ] **Step 3: Añadir sección "Interfaz web" al README**

Inserta una nueva sección justo después de la sección "Uso de la API" (después de la línea 159, antes de `---` / `## Tests` en la línea 161), con este contenido:

```markdown
## Interfaz web

Además de la API, hay una interfaz de demo servida en la raíz:

http://localhost:8000/

Permite lanzar preguntas contra `/query` (con ejemplos precargados para los tres tipos de
routing), lanzar ingestas contra `/ingest` y ver el estado de salud del sistema (`/health`).

> **Limitación conocida:** `/ingest` es síncrono y bloqueante — la petición HTTP permanece
> abierta durante todo el proceso de ingesta (puede tardar varios minutos). Una versión de
> producción debería lanzarlo como job asíncrono con `job_id` y un endpoint de estado en
> lugar de depender de una conexión HTTP larga.

---
```

- [ ] **Step 4: Verificación end-to-end completa**

```bash
docker compose up --build -d
python test/test_api.py
```

Expected: el script corre sin excepciones, imprime `llm : gemini` en el health check y completa los 5 casos de prueba.

Recorre manualmente en el navegador, en `http://localhost:8000/`:
- [ ] Badge de salud en verde tras cargar.
- [ ] Los 3 chips de ejemplo, uno por cada `query_type` (thematic_summary / specific_query / transversal_query), devuelven el tipo esperado con fuentes enlazadas correctamente (pubmed → `pubmed.ncbi.nlm.nih.gov`, pmc → `ncbi.nlm.nih.gov/pmc/articles`).
- [ ] Una pregunta fuera de dominio devuelve `none` sin fuentes.
- [ ] Pestaña Ingestar: aviso visible, envío exitoso con mensaje de confirmación, botón deshabilitado durante el proceso.
- [ ] `/docs` (Swagger UI) sigue funcionando en `http://localhost:8000/docs`.

- [ ] **Step 5: Commit**

```bash
git add test/test_api.py README.md
git commit -m "docs: document web UI, update health field name in README and test script"
```
