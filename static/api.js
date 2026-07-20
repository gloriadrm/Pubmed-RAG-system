async function parseErrorDetail(response) {
  try {
    const data = await response.json();
    // Errores 500 del backend: detail es un string plano.
    if (typeof data.detail === "string") return data.detail;
    // Errores 422 de validación de Pydantic/FastAPI: detail es una lista de
    // objetos {msg, loc, ...} — nos quedamos solo con el mensaje legible.
    if (Array.isArray(data.detail)) {
      return data.detail.map((e) => e.msg || JSON.stringify(e)).join("; ");
    }
    return JSON.stringify(data);
  } catch {
    return response.statusText;
  }
}

export async function getHealth() {
  const response = await fetch("/health");
  if (!response.ok) throw new Error(await parseErrorDetail(response));
  return response.json();
}

export async function postQuery(question, pmcId) {
  const body = { question };
  if (pmcId) body.pmc_id = pmcId;
  const response = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
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

export async function getCorpusStatus() {
  const response = await fetch("/corpus/status");
  if (!response.ok) throw new Error(await parseErrorDetail(response));
  return response.json();
}

export async function postUpdateCorpus() {
  // Responde de inmediato con {job_id, state:"running"} — no espera a que
  // termine el trabajo real. El progreso se sigue con getUpdateJobStatus().
  const response = await fetch("/corpus/update", { method: "POST" });
  if (!response.ok) throw new Error(await parseErrorDetail(response));
  return response.json();
}

export async function getUpdateJobStatus() {
  const response = await fetch("/corpus/update/status");
  if (!response.ok) throw new Error(await parseErrorDetail(response));
  return response.json();
}
