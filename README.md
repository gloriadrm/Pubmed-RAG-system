# Biomedical RAG System

Sistema de Recuperación Aumentada con Generación (RAG) sobre artículos biomédicos indexados desde PubMed y PubMed Central (PMC). Permite consultas en lenguaje natural sobre un corpus científico personalizado, clasificando automáticamente el tipo de pregunta y recuperando los fragmentos más relevantes antes de generar una respuesta fundamentada.

---

## Arquitectura general

```
Usuario → API (FastAPI)
             │
             ├─ Clasificador (Gemini) → tipo de query
             ├─ Retrieval (Qdrant + BGE-M3) → chunks relevantes
             └─ Generación (Gemini 2.0 Flash) → respuesta citada
```

**Componentes principales:**

| Componente | Tecnología | Dónde corre |
|---|---|---|
| Base de datos vectorial | Qdrant | Docker |
| Modelo de embeddings | BGE-M3 (`BAAI/bge-m3`) | Docker (contenedor API) |
| LLM (clasificador + generación) | Gemini 2.0 Flash | API de Google |
| API REST | FastAPI + LangChain | Docker |
| Fuentes de datos | PubMed / PMC (NCBI Entrez) | API pública |

---

## Requisitos previos

### Software local
- **Docker Desktop** — necesario para levantar Qdrant y la API
  → https://www.docker.com/products/docker-desktop
- **Python 3.11+** — solo si quieres ejecutar los tests o notebooks localmente

### Credenciales (claves API)
El sistema necesita dos claves que se configuran en un archivo `.env` en la raíz del proyecto:

| Variable | Fuente | Para qué se usa |
|---|---|---|
| `NCBI_API_KEY` | https://www.ncbi.nlm.nih.gov/account/ | Aumenta el rate limit de PubMed de 3 a 10 req/s |
| `GOOGLE_API_KEY` | https://aistudio.google.com/apikey | LLM para clasificación y generación de respuestas |

Crea el archivo `.env` en la raíz del proyecto:

```
NCBI_API_KEY=tu_clave_ncbi
GOOGLE_API_KEY=tu_clave_google
```

> La `NCBI_API_KEY` es opcional pero recomendada. Sin ella el pipeline de ingesta funciona con límites más bajos.

---

## Instalación y puesta en marcha

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd project
```

### 2. Configurar el archivo `.env`

Crea el archivo `.env` en la raíz del proyecto con las claves descritas arriba.

### 3. Levantar los servicios con Docker

```bash
docker compose up --build -d
```

Esto construye la imagen de la API (incluye la descarga del modelo BGE-M3, puede tardar varios minutos la primera vez) y levanta dos contenedores:

- `qdrant_ncbi_db` — base de datos vectorial Qdrant en el puerto `6333`
- `biomedical_rag_api` — API FastAPI en el puerto `8000`

Para verificar que todo está corriendo:

```bash
docker logs biomedical_rag_api -f
```

Cuando aparezca `Application startup complete.` el sistema está listo.

### 4. Verificar el estado

```bash
curl http://localhost:8000/health
```

Respuesta esperada:
```json
{"status": "ok", "qdrant": "ok", "llm": "gemini"}
```

---

## Ingesta de datos

La colección de Qdrant empieza vacía. Hay dos formas de ingestar artículos:

### Opción A — Endpoint `/ingest` (desde la API)

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"query": "gut microbiota intestinal permeability", "n": 20}'
```

Lanza el pipeline completo: búsqueda en PubMed → cruce con PMC → chunking → indexación en Qdrant.

### Opción B — Script de test (ingesta con query predefinida)

Ejecuta el pipeline completo con una query de ejemplo sobre microbiota intestinal e indexa 10 artículos. Recrea la colección desde cero.

```bash
python test/test_indexing.py
```

> La primera vez que se ejecuta descarga el catálogo OA de PMC (~500 MB) y lo guarda en `data/oa_file_list.csv` para usos posteriores.

---

## Uso de la API

La documentación interactiva está disponible en http://localhost:8000/docs una vez levantado el sistema.

### Endpoint principal: `POST /query`

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the current evidence on gut microbiota and intestinal permeability?"}'
```

El sistema clasifica automáticamente la pregunta en uno de estos tipos:

| Tipo | Descripción | Fuente del retrieval |
|---|---|---|
| `thematic_summary` | Visión general sobre un tema | Abstracts PubMed |
| `specific_query` | Detalles de un paper concreto | Texto completo PMC |
| `transversal_query` | Mecanismo o evidencia cruzada entre papers | Texto completo PMC (MMR) |
| `none` | Fuera del dominio biomédico | — |

**Respuesta:**
```json
{
  "question": "...",
  "answer": "...",
  "query_type": "thematic_summary",
  "reason": "...",
  "pmc_id": null,
  "sources": [
    {"title": "...", "section": null, "pmc_id": "PMC...", "source": "pubmed"}
  ]
}
```

---

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

## Tests

Instala las dependencias localmente (solo para los tests):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Todos los tests se ejecutan desde la raíz del proyecto. Requieren que Qdrant esté corriendo (basta con `docker compose up -d qdrant`), salvo `test_api.py` que necesita la API completa.

---

### `test/test_indexing.py` — Ingesta de prueba

Ejecuta el pipeline completo de ingesta con una query predefinida sobre microbiota intestinal, permeabilidad e inflamación. Recrea la colección desde cero e indexa 10 artículos (abstracts PubMed + chunks PMC cuando hay texto completo disponible). Útil para verificar que todo el stack funciona correctamente tras un cambio.

```bash
python test/test_indexing.py
```

---

### `test/test_semantic_search.py` — Búsqueda semántica directa

Valida los tres patrones de retrieval directamente sobre Qdrant, sin pasar por la API ni el LLM. Ejecuta y muestra resultados para:
- **Caso 1 — Resumen temático:** búsqueda sobre abstracts PubMed (`source=pubmed`)
- **Caso 2 — Paper concreto:** búsqueda dentro de los chunks de un artículo PMC específico (`source=pmc + pmc_id`)
- **Caso 3 — Transversal:** búsqueda sobre todos los chunks PMC (`source=pmc`)
- **Diagnóstico adicional:** distribución de secciones más frecuentes en los chunks indexados

```bash
python test/test_semantic_search.py
```

---

### `test/test_api.py` — Test end-to-end de la API

Lanza 5 casos de prueba contra la API REST y muestra para cada uno el tipo de query detectado, las fuentes recuperadas y la respuesta generada por el LLM. Requiere la API corriendo en Docker.

| Caso | Query | Tipo esperado |
|---|---|---|
| 1 | Evidencia general sobre microbiota y permeabilidad | `thematic_summary` → PubMed |
| 2a | Paper concreto identificado por título | `specific_query` → PMC |
| 2b | Paper concreto identificado por autor | `specific_query` → PMC |
| 3 | Mecanismos transversales entre papers | `transversal_query` → PMC |
| OOS | Capital de Francia | `none` |

```bash
python test/test_api.py
```

Opcionalmente se puede indicar una URL distinta:

```bash
python test/test_api.py --url http://localhost:8000
```

---

### `test/test_actualizations.py` — Pipeline de actualización incremental

Ejecuta el pipeline completo de actualización: descarga los update files diarios de NLM para PubMed y compara el catálogo OA de PMC con la copia local para detectar artículos nuevos, modificados o eliminados, aplicando los cambios en Qdrant.

```bash
# Actualización completa (PubMed + PMC)
python test/test_actualizations.py

# Solo PubMed
python test/test_actualizations.py --pubmed-only

# Solo PMC
python test/test_actualizations.py --pmc-only

# Calcular cambios sin modificar Qdrant
python test/test_actualizations.py --dry-run
```

---

## Estructura del proyecto

```
project/
├── src/
│   ├── api/              # FastAPI: endpoints /health, /query, /ingest
│   ├── ingest/           # Pipeline de ingesta: PubMed, PMC, chunking, indexación
│   ├── rag/              # Chain RAG: clasificador, retrieval, prompts, estructuras
│   ├── services/         # Embeddings (BGE-M3), LLM (Gemini), vector store (Qdrant)
│   ├── data_actualization/ # Actualizadores PubMed y PMC
│   └── config.py         # Configuración centralizada
├── test/                 # Tests end-to-end y unitarios
├── notebooks/            # Exploración: extracción, embeddings, actualizaciones
├── data/                 # Archivos de estado (oa_file_list.csv, last_update tracker)
├── qdrant_data/          # Volumen persistente de Qdrant (generado automáticamente)
├── qdrant_config/        # Configuración de Qdrant
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

---

## Variables de entorno

| Variable | Requerida | Valor por defecto | Descripción |
|---|---|---|---|
| `GOOGLE_API_KEY` | Sí | — | Clave API de Google Gemini |
| `NCBI_API_KEY` | Recomendada | — | Clave API de NCBI para PubMed |
| `QDRANT_URL` | No | `http://localhost:6333` | URL de Qdrant (en Docker: `http://qdrant:6333`) |
| `OLLAMA_BASE_URL` | No | `http://localhost:11434` | URL de Ollama (alternativa local al LLM) |

---

## Notas

- El modelo BGE-M3 (~1.1 GB) se descarga automáticamente en el primer `docker compose up --build` y se cachea en un volumen Docker (`huggingface_cache`). Los rebuilds posteriores no lo vuelven a descargar.
- Los datos de Qdrant persisten en `./qdrant_data` aunque se reinicie o reconstruya el contenedor.
- El código de integración con Ollama (llama3.1:8b) está disponible comentado en `src/services/llms.py` como alternativa local al LLM si se dispone de GPU.
