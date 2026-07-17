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
