async function parseErrorDetail(response) {
  try {
    const data = await response.json();
    return data.detail || JSON.stringify(data);
  } catch {
    return response.statusText;
  }
}

export async function getHealth() {
  const response = await fetch("/health");
  if (!response.ok) throw new Error(await parseErrorDetail(response));
  return response.json();
}

export async function postQuery(question) {
  const response = await fetch("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
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
