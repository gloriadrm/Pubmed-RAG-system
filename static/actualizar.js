import { getCorpusStatus, postUpdateCorpus, getUpdateJobStatus } from "./api.js";

const POLL_INTERVAL_MS = 3000;
let pollTimer = null;

function formatDate(iso) {
  if (!iso) return "Nunca";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString("es-ES", { dateStyle: "short", timeStyle: "short" });
}

function statRow(label, value) {
  const row = document.createElement("div");
  row.className = "corpus-stat-row";
  const l = document.createElement("span");
  l.className = "corpus-stat-label";
  l.textContent = label;
  const v = document.createElement("span");
  v.className = "corpus-stat-value";
  v.textContent = value;
  row.append(l, v);
  return row;
}

// -------------- Tooltip de licencias --------------
//
// Sin librerías externas: la visibilidad del tooltip la controla el CSS
// (:hover/:focus sobre .help-icon). Sale hacia la derecha del icono,
// centrado verticalmente — pero eso hay que recalcularlo en JS justo antes
// de mostrarlo: si no cabe a la derecha, se voltea a la izquierda
// (.tooltip-left); si se sale por arriba o por abajo, se ajusta verticalmente.

function repositionTooltip(icon, tooltip) {
  tooltip.classList.remove("tooltip-left");
  tooltip.style.transform = "translateY(-50%)";

  const margin = 8;
  let rect = tooltip.getBoundingClientRect();

  if (rect.right > window.innerWidth - margin) {
    tooltip.classList.add("tooltip-left");
    rect = tooltip.getBoundingClientRect();
  }

  let shiftY = 0;
  if (rect.top < margin) {
    shiftY = margin - rect.top;
  } else if (rect.bottom > window.innerHeight - margin) {
    shiftY = (window.innerHeight - margin) - rect.bottom;
  }
  if (shiftY !== 0) {
    tooltip.style.transform = `translateY(calc(-50% + ${shiftY}px))`;
  }
}

function licenseExample(code, text) {
  const li = document.createElement("li");
  const strong = document.createElement("strong");
  strong.textContent = code;
  li.append(strong, ` → ${text}`);
  return li;
}

function createLicenseHelpIcon() {
  const wrapper = document.createElement("span");
  wrapper.className = "help-icon";
  wrapper.textContent = "ⓘ";
  wrapper.tabIndex = 0;
  wrapper.setAttribute("aria-describedby", "license-tooltip");

  const tooltip = document.createElement("span");
  tooltip.className = "tooltip";
  tooltip.id = "license-tooltip";
  tooltip.setAttribute("role", "tooltip");

  const title = document.createElement("strong");
  title.textContent = "Licencias Open Access";
  tooltip.appendChild(title);

  const intro = document.createElement("p");
  intro.textContent = "El sistema solo indexa artículos cuyo texto completo puede reutilizarse según su licencia de PubMed Central (PMC).";
  tooltip.appendChild(intro);

  const examplesLabel = document.createElement("p");
  examplesLabel.textContent = "Ejemplos:";
  examplesLabel.className = "tooltip-examples-label";
  tooltip.appendChild(examplesLabel);

  const list = document.createElement("ul");
  list.append(
    licenseExample("CC BY", "reutilización permitida con atribución."),
    licenseExample("CC BY-NC", "reutilización no comercial."),
    licenseExample("TDM", "licencia específica para minería de texto y datos."),
  );
  tooltip.appendChild(list);

  const outro = document.createElement("p");
  outro.textContent = "Estas licencias determinan qué artículos pueden incorporarse al corpus.";
  tooltip.appendChild(outro);

  wrapper.appendChild(tooltip);

  const show = () => repositionTooltip(wrapper, tooltip);
  wrapper.addEventListener("mouseenter", show);
  wrapper.addEventListener("focus", show);

  return wrapper;
}

function renderStatus(status) {
  const stats = document.getElementById("corpus-stats");
  stats.innerHTML = "";
  stats.appendChild(statRow("PubMed", status.pubmed_count.toLocaleString("es-ES")));
  stats.appendChild(statRow("PMC", status.pmc_article_count.toLocaleString("es-ES")));
  stats.appendChild(statRow("Chunks", status.pmc_chunk_count.toLocaleString("es-ES")));

  const licensesEl = document.getElementById("corpus-licenses");
  licensesEl.innerHTML = "";
  if (status.licenses.length > 0) {
    const titleRow = document.createElement("div");
    titleRow.className = "corpus-licenses-title-row";
    const h4 = document.createElement("h4");
    h4.className = "corpus-licenses-title";
    h4.textContent = "Licencias";
    titleRow.append(h4, createLicenseHelpIcon());
    licensesEl.appendChild(titleRow);
    status.licenses.forEach((l) => {
      licensesEl.appendChild(statRow(l.license, l.count));
    });
  }

  document.getElementById("corpus-last-updated").textContent =
    `Última actualización: ${formatDate(status.last_updated)}`;
}

async function refreshCorpusStatus() {
  try {
    const status = await getCorpusStatus();
    renderStatus(status);
  } catch {
    // El estado del corpus no es crítico para el resto de la demo — /health
    // ya reporta problemas de conectividad con Qdrant; aquí simplemente no
    // se actualizan las cifras.
  }
}

function clearMessages() {
  document.getElementById("update-result").hidden = true;
  document.getElementById("update-error").hidden = true;
}

function phaseLabel(phase) {
  if (phase === "pubmed") return "Actualizando… revisando PubMed.";
  if (phase === "pmc") return "Actualizando… revisando PMC.";
  return "Actualizando…";
}

function setLoading(isLoading) {
  document.getElementById("update-loading").hidden = !isLoading;
  document.getElementById("update-loading-note").hidden = !isLoading;
}

function renderRunning(jobStatus) {
  setLoading(true);
  const label = document.getElementById("update-loading-label");
  if (label) label.textContent = phaseLabel(jobStatus.phase);
}

function renderResult(result, log) {
  const box = document.getElementById("update-result");
  box.hidden = false;
  box.innerHTML = "";

  const list = document.createElement("ul");
  list.className = "update-checklist";

  const pmcRemoved = result.pmc_removed_license + result.pmc_removed_gone;
  const items = [
    `PubMed revisados: ${result.pubmed_checked}`,
    `Artículos PubMed modificados: ${result.pubmed_updated}`,
  ];
  if (result.pubmed_deleted > 0) items.push(`Artículos PubMed borrados: ${result.pubmed_deleted}`);
  items.push(`PMC revisados: ${result.pmc_checked}`,
             `PMC reindexados: ${result.pmc_reingested}`);
  if (result.pmc_retracted > 0) items.push(`Artículos retractados: ${result.pmc_retracted}`);
  if (pmcRemoved > 0) items.push(`PMC retirados: ${pmcRemoved}`);
  items.push("Sin errores");

  items.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = `✔ ${text}`;
    list.appendChild(li);
  });
  box.appendChild(list);

  if (log && log.length) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Ver log";
    details.appendChild(summary);
    const pre = document.createElement("pre");
    pre.textContent = log.join("\n");
    details.appendChild(pre);
    box.appendChild(details);
  }
}

function renderError(message) {
  const box = document.getElementById("update-error");
  box.hidden = false;
  box.innerHTML = "";

  const p = document.createElement("p");
  p.textContent = "Ha ocurrido un error al actualizar el corpus.";
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

function stopPolling() {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
}

function setButtonRunning(isRunning) {
  document.getElementById("update-submit").disabled = isRunning;
}

async function poll() {
  let jobStatus;
  try {
    jobStatus = await getUpdateJobStatus();
  } catch (e) {
    stopPolling();
    setButtonRunning(false);
    setLoading(false);
    renderError(e.message);
    return;
  }

  if (jobStatus.state === "running") {
    renderRunning(jobStatus);
    setButtonRunning(true);
    pollTimer = setTimeout(poll, POLL_INTERVAL_MS);
    return;
  }

  // completed | failed | idle: ya no hay nada corriendo.
  stopPolling();
  setButtonRunning(false);
  setLoading(false);

  if (jobStatus.state === "completed" && jobStatus.result) {
    clearMessages();
    renderResult(jobStatus.result, jobStatus.log);
    refreshCorpusStatus();
  } else if (jobStatus.state === "failed") {
    clearMessages();
    renderError(jobStatus.error || "Error desconocido.");
  }
}

async function resumeIfRunning() {
  // Se llama al iniciar la pestaña (carga de página o cambio de pestaña):
  // si ya hay un job en curso — porque se lanzó antes de cambiar de
  // pestaña, o de recargar — retoma el polling en vez de asumir que no hay
  // nada. Si no hay ningún job activo, no se muestra automáticamente el
  // resultado de una ejecución antigua: solo aparece cuando el usuario
  // dispara una nueva desde esta sesión.
  let jobStatus;
  try {
    jobStatus = await getUpdateJobStatus();
  } catch {
    return;
  }
  if (jobStatus.state === "running") {
    setButtonRunning(true);
    renderRunning(jobStatus);
    stopPolling();
    pollTimer = setTimeout(poll, POLL_INTERVAL_MS);
  }
}

export function initActualizar() {
  const submitBtn = document.getElementById("update-submit");
  const modal = document.getElementById("update-confirm-modal");
  const cancelBtn = document.getElementById("update-confirm-cancel");
  const acceptBtn = document.getElementById("update-confirm-accept");
  const updateTabBtn = document.querySelector('.tab-btn[data-tab="update"]');

  if (updateTabBtn) {
    updateTabBtn.addEventListener("click", () => {
      refreshCorpusStatus();
      resumeIfRunning();
    });
  }
  refreshCorpusStatus();
  resumeIfRunning();

  submitBtn.addEventListener("click", () => {
    if (submitBtn.disabled) return;
    modal.hidden = false;
  });

  cancelBtn.addEventListener("click", () => { modal.hidden = true; });
  modal.addEventListener("click", (event) => {
    if (event.target === modal) modal.hidden = true;
  });

  acceptBtn.addEventListener("click", async () => {
    modal.hidden = true;
    clearMessages();
    setButtonRunning(true);
    setLoading(true);
    document.getElementById("update-loading-label").textContent = "Actualizando…";

    try {
      await postUpdateCorpus();
      stopPolling();
      pollTimer = setTimeout(poll, 500);   // primera comprobación casi inmediata
    } catch (e) {
      setButtonRunning(false);
      setLoading(false);
      renderError(e.message);
    }
  });
}
