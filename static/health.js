import { getHealth } from "./api.js";

export const PROVIDER_LABELS = {
  openai: "OpenAI",
  gemini: "Gemini",
  ollama: "Ollama",
};

// Última respuesta de /health cacheada — reutilizada por query.js para mostrar
// el modelo LLM activo en el pipeline de cada consulta, sin pedirla de nuevo.
let lastHealth = null;

export function getCachedHealth() {
  return lastHealth;
}

function setDot(containerId, state, title) {
  const container = document.getElementById(containerId);
  const dot = container.querySelector(".dot");
  dot.dataset.state = state;
  if (title) {
    container.title = title;
  } else {
    container.removeAttribute("title");
  }
}

function llmStatusLabel(provider, llmStatus) {
  // OpenAI/Gemini: "configured" -> "configurado". Ollama: "reachable" -> "disponible".
  // Cualquier otro valor (misconfigured/unreachable/error ...) se muestra tal cual,
  // ya es descriptivo (p. ej. "misconfigured (missing OPENAI_API_KEY)").
  if (provider === "ollama" && llmStatus === "reachable") return "disponible";
  if ((provider === "openai" || provider === "gemini") && llmStatus === "configured") return "configurado";
  return llmStatus;
}

async function refreshHealth() {
  setDot("health-api", "unknown");
  setDot("health-qdrant", "unknown");
  setDot("health-embeddings", "unknown");
  setDot("health-llm", "unknown");
  document.getElementById("health-llm-label").textContent = "LLM";

  try {
    const data = await getHealth();
    lastHealth = data;

    setDot("health-api", "ok");
    setDot("health-qdrant", data.qdrant === "ok" ? "ok" : "error", `Qdrant: ${data.qdrant}`);
    setDot(
      "health-embeddings",
      data.embedding_status === "loaded" ? "ok" : "error",
      `${data.embedding_provider}: ${data.embedding_status}`
    );

    const providerLabel = PROVIDER_LABELS[data.llm_provider] || data.llm_provider;
    const llmOk = data.llm_status === "configured" || data.llm_status === "reachable";
    document.getElementById("health-llm-label").textContent = `LLM: ${providerLabel}`;
    setDot("health-llm", llmOk ? "ok" : "error", llmStatusLabel(data.llm_provider, data.llm_status));
  } catch {
    setDot("health-api", "error");
    setDot("health-qdrant", "error");
    setDot("health-embeddings", "error");
    setDot("health-llm", "error");
  }
}

export function initHealth() {
  document.getElementById("health-refresh").addEventListener("click", refreshHealth);
  refreshHealth();
}
