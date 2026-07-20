# Frontend de demo — Biomedical RAG

**Fecha:** 2026-07-17
**Estado:** Aprobado, pendiente de implementación

## Objetivo

Dar al sistema RAG biomédico (FastAPI + Qdrant + Gemini, ver `README.md`) una interfaz web mínima para poder hacer demos en vivo sin depender de `curl` o de `/docs`. La demo debe permitir: preguntar en lenguaje natural, ver la respuesta con sus fuentes citadas, comprobar la salud del sistema y lanzar una ingesta de nuevos artículos.

## Alcance

Incluye los tres endpoints existentes: `/query`, `/ingest`, `/health`. No se añaden endpoints nuevos salvo el cambio de forma descrito en "Cambios en el backend". No hay autenticación, no hay persistencia de historial de conversación entre recargas, no hay streaming de la respuesta del LLM.

## Arquitectura

Página estática sin build step ni framework (HTML + CSS + JS vanilla), servida directamente por el FastAPI existente mediante `StaticFiles`, montada en `/` **después** de registrar las rutas de la API — así `/health`, `/query`, `/ingest` y `/docs` siguen funcionando exactamente igual.

```
project/static/
├── index.html
├── style.css
└── app.js
```

No se añade ningún contenedor ni servicio nuevo a `docker-compose.yml`.

## Cambios en el backend

El único cambio de contrato es en `/health`. Actualmente comprueba Ollama, que no es el proveedor activo (está comentado en `src/services/llms.py`; el LLM activo es Gemini, hardcodeado). Mostrar "Ollama" en la UI sugeriría que el sistema está degradado cuando en realidad funciona con normalidad.

`src/api/schema.py` — `HealthResponse`:
```python
class HealthResponse(BaseModel):
    status: str
    qdrant: str
    llm:    str   # antes: ollama
```

`src/api/main.py` — `health()`: deja de hacer ping a `OLLAMA_BASE_URL`. En su lugar reporta el proveedor LLM activo comprobando si `GOOGLE_API_KEY` está presente en el entorno:
- `GOOGLE_API_KEY` presente → `"llm": "gemini"`, cuenta como `ok` para el `status` global.
- Ausente → `"llm": "gemini (missing GOOGLE_API_KEY)"`, `status` global pasa a `degraded`.

Segundo cambio: `SourceDoc` y `QueryResponse` necesitan exponer el PMID además del `pmc_id`, para poder enlazar correctamente cada fuente (ver "Fuentes citadas").

`src/api/schema.py` — `SourceDoc`:
```python
class SourceDoc(BaseModel):
    title:   str
    section: Optional[str] = None
    pmc_id:  Optional[str] = None
    pm_id:   Optional[str] = None   # nuevo — PMID de PubMed
    source:  str
```

`src/api/main.py` — en `query()`, al construir cada `SourceDoc` añadir `pm_id=meta.get("pm_id")`.

## Cómo se resuelve cada tipo de consulta (contexto para las preguntas de ejemplo)

El router (`src/rag/chain.py`) clasifica cada pregunta en uno de cuatro tipos antes de recuperar contexto de Qdrant. La demo debe poder ilustrar los tres tipos "en scope":

- **`thematic_summary`** — visión general de un tema. Búsqueda semántica (`similarity_search`) filtrada a `metadata.source = pubmed`, es decir, sobre abstracts. Recupera `k=6` (`RETRIEVAL_K_THEMATIC`) documentos, uno por artículo. Ejemplo: *"Resumen general sobre microbiota y permeabilidad intestinal"*.

- **`specific_query`** — detalle de un paper concreto. Es un proceso en dos etapas:
  1. Resolver el `pmc_id` del artículo si no vino explícito, probando en orden: filtro por apellido de autor extraído por el LLM → búsqueda semántica del título extraído contra los abstracts de PubMed → si todo falla, búsqueda semántica de la pregunta completa contra PubMed (`_resolve_pmc_id`).
  2. Búsqueda semántica filtrada a `metadata.source = pmc` **y** `metadata.pmc_id = <resuelto>`, sobre los chunks de texto completo de ese artículo. Recupera `k=10` (`RETRIEVAL_K_SPECIFIC`).
  Si no se logra resolver ningún `pmc_id`, el sistema degrada silenciosamente a un comportamiento `transversal_query` (mismo tipo de búsqueda que abajo) — la respuesta seguirá indicando `specific_query` como `query_type` aunque internamente haya usado la búsqueda transversal. Ejemplo: *"¿Qué metodología utiliza el estudio de [autor/título]?"*.

- **`transversal_query`** — mecanismos o evidencia cruzada entre varios papers. Búsqueda por relevancia marginal máxima (`max_marginal_relevance_search`, MMR) filtrada a `metadata.source = pmc` sobre **todos** los artículos indexados, con `k=6` (`RETRIEVAL_K_TRANSVERSAL`) y `fetch_k=18`. MMR prioriza diversidad entre los documentos recuperados (evita devolver 6 chunks del mismo paper), apropiado cuando se busca evidencia repartida entre distintos estudios. Ejemplo: *"¿Qué mecanismos inflamatorios aparecen en varios estudios?"*.

- **`none`** — fuera del dominio biomédico. No se recupera contexto; el LLM responde que la pregunta está fuera de alcance.

## UI — Layout y estilo

Cabecera normal (no fija/sticky, se desplaza con el scroll) con el título "Biomedical RAG" y, en texto pequeño y gris —visualmente secundario frente a la consulta—, el estado de salud junto a un botón `↻` de refresco manual:

```
API · Qdrant · LLM
```

cada uno con su propio indicador ok/degraded. Se comprueba una vez al cargar la página (`fetch('/health')`) y de nuevo solo si el usuario pulsa `↻`.

Debajo, dos pestañas: **Consultar** (activa por defecto) e **Ingestar**. Paleta minimalista: blanco/negro, acentos grises, bordes redondeados, tipografía de sistema.

## Pestaña Consultar

- Input de texto libre + botón "Preguntar". Sin campo `pmc_id` manual.
- **Preguntas de ejemplo**: 3 chips debajo del input, uno por cada `query_type` en scope (ver ejemplos arriba). Al pulsar uno, se rellena el input con ese texto (editable antes de enviar) — sirve para demostrar los tres caminos de routing sin que el visitante tenga que conocer el corpus de antemano.
- Al enviar: el botón se deshabilita y su texto cambia a "Consultando…" hasta recibir respuesta (sin spinner adicional).
- Respuesta: badge con el `query_type` detectado, el texto de la respuesta, y debajo la lista de fuentes.
- **Fuentes citadas** — cada fuente es un enlace, no solo texto:
  - `source = "pmc"` → enlaza a `https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc_id}/`
  - `source = "pubmed"` → enlaza a `https://pubmed.ncbi.nlm.nih.gov/{pm_id}/`
  - Si el identificador correspondiente faltara, se muestra como texto plano sin enlace (no debería ocurrir en la práctica: `pm_id` siempre está presente en docs `pubmed`, `pmc_id` siempre está presente en docs `pmc`).
  - Formato visible: `Título del artículo` — `{pmc_id o pm_id} · {sección si aplica} · {pubmed|pmc}`.
- Error (p. ej. 500 de `/query`): mensaje genérico prominente ("Ha ocurrido un error al procesar la consulta") con el `detail` crudo del backend disponible en un `<details>` colapsado.

## Pestaña Ingestar

- Aviso visible: el proceso puede tardar varios minutos, no cerrar la pestaña.
- Input de query temática + input numérico `n` (por defecto 10) + botón "Ingestar".
- Al enviar: botón deshabilitado (evita envíos duplicados) + spinner con mensaje persistente "Ingestando…" durante toda la espera. `fetch` síncrono, sin polling a `/health` (no representa el progreso real).
- Éxito: mensaje de confirmación con el texto devuelto por la API.
- Error: mismo patrón que en Consultar (mensaje genérico + detalle colapsado).

**Limitación conocida (documentar en el propio código o en el README, no resolver ahora):** `/ingest` es síncrono y bloqueante en el backend. Una versión de producción debería lanzarlo como job asíncrono con `job_id` y un endpoint de estado, en lugar de mantener la conexión HTTP abierta varios minutos.

## Testing

No se añaden tests automatizados de frontend (fuera de alcance para una demo). Sí se debe:
- Extender `test/test_api.py` o revisar manualmente que `/health` devuelve el nuevo campo `llm` en vez de `ollama`.
- Verificar manualmente en navegador los tres tipos de query, el caso `none`, un error simulado de `/query`, y un ciclo completo de `/ingest`.
