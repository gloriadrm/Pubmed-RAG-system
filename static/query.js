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
