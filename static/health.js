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
