import { postQuery } from "./api.js";
import { getCachedHealth, PROVIDER_LABELS } from "./health.js";

// -------------- Etiquetas humanas (deterministas, sin depender del LLM) --------------

const QUERY_TYPE_LABELS = {
  thematic_summary: "Resumen temático",
  specific_query: "Artículo concreto",
  transversal_query: "Comparación transversal",
  none: "Fuera de dominio",
};

const QUERY_TYPE_EXPLANATIONS = {
  thematic_summary: "La consulta solicita una visión general sobre un tema biomédico.",
  specific_query: "La consulta hace referencia a un artículo científico concreto.",
  transversal_query: "La consulta requiere comparar información procedente de varios artículos.",
  none: "La consulta está fuera del ámbito o de las capacidades del sistema.",
};

function retrievalLabel(strategy, pmcId) {
  switch (strategy) {
    case "thematic_summary": return "Abstracts de PubMed";
    case "specific_query": return pmcId ? `Texto completo — ${pmcId}` : "Texto completo del artículo";
    case "transversal_query": return "Texto completo, varios artículos (PMC)";
    case "none": return "Sin retrieval";
    default: return strategy;
  }
}

function retrievalDetail(strategy, pmcId) {
  switch (strategy) {
    case "thematic_summary":
      return "Se recuperaron abstracts de PubMed mediante búsqueda semántica.";
    case "specific_query":
      return `Se recuperó el texto completo del artículo ${pmcId || ""} mediante búsqueda semántica dentro de sus fragmentos indexados.`;
    case "transversal_query":
      return "Se recuperaron fragmentos de texto completo de varios artículos PMC mediante búsqueda por relevancia marginal máxima (MMR), priorizando diversidad entre estudios.";
    case "none":
      return "No se realizó ninguna búsqueda en la base de datos.";
    default:
      return "";
  }
}

function prettyModel(model) {
  if (!model) return "";
  return model.replace(/^gpt-/i, "GPT-");
}

// -------------- Pipeline --------------

function pipelineStep(label, value, detailNodes) {
  const step = document.createElement("div");
  step.className = "pipeline-step";

  const labelEl = document.createElement("div");
  labelEl.className = "pipeline-step-label";
  labelEl.textContent = label;
  step.appendChild(labelEl);

  const valueEl = document.createElement("div");
  valueEl.className = "pipeline-step-value";
  valueEl.textContent = value;
  step.appendChild(valueEl);

  if (detailNodes) {
    const details = document.createElement("details");
    details.className = "pipeline-detail";
    const summary = document.createElement("summary");
    summary.textContent = "Detalle";
    details.appendChild(summary);
    detailNodes.forEach((node) => details.appendChild(node));
    step.appendChild(details);
  }

  return step;
}

function pipelineArrow() {
  const arrow = document.createElement("div");
  arrow.className = "pipeline-arrow";
  arrow.textContent = "→";
  arrow.setAttribute("aria-hidden", "true");
  return arrow;
}

function renderPipeline(result) {
  const wrap = document.createElement("div");
  wrap.className = "pipeline";

  // 1. Consulta
  wrap.appendChild(pipelineStep("Consulta", `"${result.question}"`));
  wrap.appendChild(pipelineArrow());

  // 2. Clasificación — el "reason" crudo del clasificador queda como detalle
  // técnico anidado, no como explicación principal: mantenemos reason como
  // dato técnico (posiblemente en inglés) y mostramos una explicación fija
  // y localizada por query_type, consistente entre proveedores LLM.
  const classificationDetail = document.createElement("p");
  classificationDetail.textContent = QUERY_TYPE_EXPLANATIONS[result.query_type] || result.query_type;

  const technicalDetails = document.createElement("details");
  technicalDetails.className = "pipeline-subdetail";
  const technicalSummary = document.createElement("summary");
  technicalSummary.textContent = "Ver detalle técnico";
  technicalDetails.appendChild(technicalSummary);
  const reasonPre = document.createElement("pre");
  reasonPre.textContent = result.reason;
  technicalDetails.appendChild(reasonPre);

  wrap.appendChild(pipelineStep(
    "Clasificación",
    QUERY_TYPE_LABELS[result.query_type] || result.query_type,
    [classificationDetail, technicalDetails],
  ));
  wrap.appendChild(pipelineArrow());

  // 3. Retrieval — puede diferir de la clasificación (degradación, pmc_id
  // forzado, etc.); se muestra igual, sin énfasis visual especial.
  const retrievalDetailP = document.createElement("p");
  retrievalDetailP.textContent = retrievalDetail(result.retrieval_strategy, result.pmc_id);
  wrap.appendChild(pipelineStep(
    "Retrieval",
    retrievalLabel(result.retrieval_strategy, result.pmc_id),
    [retrievalDetailP],
  ));
  wrap.appendChild(pipelineArrow());

  // 4. Modelo — dato de /health, cacheado (sin llamada extra por consulta)
  const health = getCachedHealth();
  let modelValue = "—";
  if (health) {
    const providerLabel = PROVIDER_LABELS[health.llm_provider] || health.llm_provider;
    const model = prettyModel(health.llm_model);
    modelValue = model ? `${providerLabel} · ${model}` : providerLabel;
  }
  wrap.appendChild(pipelineStep("Modelo", modelValue));

  return wrap;
}

// -------------- Fuentes --------------

function pmcUrl(pmcId) {
  return `https://www.ncbi.nlm.nih.gov/pmc/articles/${pmcId}/`;
}

function pubmedUrl(pmId) {
  return `https://pubmed.ncbi.nlm.nih.gov/${pmId}/`;
}

function renderSource(source) {
  const card = document.createElement("li");
  card.className = "source-card";

  let url = null;
  let idLabel = null;
  let typeLabel = null;
  if (source.source === "pmc" && source.pmc_id) {
    url = pmcUrl(source.pmc_id);
    idLabel = source.pmc_id;
    typeLabel = "Texto completo · PMC";
  } else if (source.source === "pubmed" && source.pm_id) {
    url = pubmedUrl(source.pm_id);
    idLabel = source.pm_id;
    typeLabel = "Abstract · PubMed";
  } else {
    typeLabel = source.source;
  }

  const type = document.createElement("div");
  type.className = "source-type";
  type.textContent = typeLabel;
  card.appendChild(type);

  const titleEl = document.createElement(url ? "a" : "div");
  titleEl.className = "source-title";
  titleEl.textContent = source.title;
  if (url) {
    titleEl.href = url;
    titleEl.target = "_blank";
    titleEl.rel = "noopener noreferrer";
  }
  card.appendChild(titleEl);

  const metaParts = [];
  if (idLabel) metaParts.push(idLabel);
  if (source.section) metaParts.push(source.section);
  if (metaParts.length) {
    const meta = document.createElement("div");
    meta.className = "source-meta";
    meta.textContent = metaParts.join(" · ");
    card.appendChild(meta);
  }

  if (source.excerpt) {
    const excerptLabel = source.source === "pubmed" ? "abstract" : "fragmento";

    const toggle = document.createElement("details");
    toggle.className = "source-excerpt-toggle";

    const summary = document.createElement("summary");
    summary.textContent = `Mostrar ${excerptLabel}`;
    toggle.addEventListener("toggle", () => {
      summary.textContent = toggle.open ? `Ocultar ${excerptLabel}` : `Mostrar ${excerptLabel}`;
    });
    toggle.appendChild(summary);

    const excerpt = document.createElement("p");
    excerpt.className = "source-excerpt";
    excerpt.textContent = source.excerpt;
    toggle.appendChild(excerpt);

    card.appendChild(toggle);
  }

  return card;
}

// -------------- Resultado completo --------------

function sectionHeader(text) {
  const h3 = document.createElement("h3");
  h3.className = "section-header";
  h3.textContent = text;
  return h3;
}

function renderResult(result) {
  const container = document.getElementById("query-result");
  container.innerHTML = "";
  container.hidden = false;

  container.appendChild(renderPipeline(result));

  if (result.retrieval_note) {
    const note = document.createElement("p");
    note.className = "retrieval-note";
    note.textContent = `ℹ️ ${result.retrieval_note}`;
    container.appendChild(note);
  }

  container.appendChild(sectionHeader("Resultado"));
  const answer = document.createElement("p");
  answer.className = "answer";
  answer.textContent = result.answer;
  container.appendChild(answer);

  if (result.sources.length > 0) {
    container.appendChild(sectionHeader(`Fuentes utilizadas (${result.sources.length})`));
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

// -------------- Campo PMC ID --------------
//
// Solo existe visualmente cuando el workflow "Artículo concreto" está activo
// (chip con data-pmc-id) — no hay disclosure manual: para el resto de
// workflows el campo permanece colapsado (max-height:0) y nunca se envía.
// Es una restricción de UX del frontend, no del backend: QueryRequest.pmc_id
// sigue siendo opcional en la API, para no romper la resolución automática
// por autor/título/semántica de otras vías (curl, /docs, etc.).

// El backend normaliza formatos flexibles (con o sin prefijo "PMC", may/min);
// aquí solo replicamos esa misma tolerancia para decidir si habilitar el
// botón — la validación estricta de verdad sigue viviendo en el backend.
const PMC_ID_PATTERN = /^(pmc)?\d+$/i;

// Única fuente de verdad para decidir si pmc_id viaja en el payload. No basta
// con que el campo esté visualmente oculto (classList "visible"): al cambiar
// de workflow el valor anterior podía sobrevivir en pmcInput.value y colarse
// en la siguiente petición, dando prioridad indebida a request.pmc_id en el
// backend. activeWorkflow se resetea explícitamente en cada clic de chip.
let activeWorkflow = null;

function isPmcIdPlausible(value) {
  return PMC_ID_PATTERN.test(value.trim());
}

function updateSubmitState() {
  const submitBtn = document.getElementById("query-submit");
  if (activeWorkflow !== "specific_query") {
    submitBtn.disabled = false;
    return;
  }
  // El campo PMC ID es una vía estructurada alternativa, no un requisito: en
  // specific_query el artículo puede identificarse igual de bien dentro de la
  // propia pregunta (autor, título...), así que el campo vacío es válido. Solo
  // bloqueamos el envío si el usuario ha escrito algo que no tiene pinta de
  // PMC ID — evita un 422 evitable, no impone rellenarlo.
  const hasQuestion = document.getElementById("question").value.trim().length > 0;
  const pmcValue = document.getElementById("pmc-id").value.trim();
  const pmcOk = pmcValue === "" || isPmcIdPlausible(pmcValue);
  submitBtn.disabled = !(hasQuestion && pmcOk);
}

function openPmcField(value) {
  const field = document.getElementById("pmc-field");
  document.getElementById("pmc-id").value = value || "";
  field.classList.add("visible");
  activeWorkflow = "specific_query";
  updateSubmitState();
}

function closePmcField(workflow) {
  const field = document.getElementById("pmc-field");
  field.classList.remove("visible");
  document.getElementById("pmc-id").value = "";
  activeWorkflow = workflow || null;
  updateSubmitState();
}

// -------------- Init --------------

export function initQuery() {
  const form = document.getElementById("query-form");
  const textarea = document.getElementById("question");
  const pmcInput = document.getElementById("pmc-id");
  const submitBtn = document.getElementById("query-submit");

  pmcInput.addEventListener("input", updateSubmitState);
  textarea.addEventListener("input", updateSubmitState);

  document.querySelectorAll("#tab-query .chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      textarea.value = chip.dataset.example;
      if (chip.dataset.workflow === "specific_query") {
        openPmcField(chip.dataset.pmcId);
      } else {
        closePmcField(chip.dataset.workflow);
      }
      textarea.focus();
    });
  });

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    clearMessages();

    submitBtn.disabled = true;
    submitBtn.textContent = "Consultando…";

    // No basta con leer pmcInput.value: solo viaja si el workflow activo es
    // realmente specific_query, aunque el input conserve un valor residual.
    const pmcId = activeWorkflow === "specific_query" ? pmcInput.value.trim() : "";

    try {
      const result = await postQuery(textarea.value.trim(), pmcId);
      renderResult(result);
    } catch (e) {
      renderError(e.message);
    } finally {
      updateSubmitState();
      submitBtn.textContent = "Preguntar";
    }
  });
}
