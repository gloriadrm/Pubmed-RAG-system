import { postIngest } from "./api.js";

function clearMessages() {
  document.getElementById("ingest-status").hidden = true;
  document.getElementById("ingest-error").hidden = true;
}

function setLoading(isLoading) {
  document.getElementById("ingest-loading").hidden = !isLoading;
}

function renderSuccess(result) {
  const box = document.getElementById("ingest-status");
  box.hidden = false;
  box.innerHTML = "";

  const list = document.createElement("ul");
  list.className = "update-checklist";

  const items = [`${result.articles_found} artículos encontrados`,
                  `${result.pubmed_indexed} PubMed indexados`,
                  `${result.pmc_found} PMC encontrados`];
  if (result.pmc_excluded_license > 0) {
    items.push(`${result.pmc_excluded_license} excluidos por licencia restrictiva`);
  }
  items.push(`${result.xml_downloaded} XML descargados`,
             `${result.chunks_created} chunks creados`,
             "Qdrant actualizado");

  items.forEach((text) => {
    const li = document.createElement("li");
    li.textContent = `✔ ${text}`;
    list.appendChild(li);
  });
  box.appendChild(list);

  if (result.log && result.log.length) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = "Ver log";
    details.appendChild(summary);
    const pre = document.createElement("pre");
    pre.textContent = result.log.join("\n");
    details.appendChild(pre);
    box.appendChild(details);
  }
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
      renderSuccess(result);
    } catch (e) {
      renderError(e.message);
    } finally {
      submitBtn.disabled = false;
      setLoading(false);
    }
  });
}
