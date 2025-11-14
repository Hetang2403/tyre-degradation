// Simple wrapper around fetch for our FastAPI backend

const BASE_URL = import.meta.env.VITE_API_BASE_URL?.replace(/\/$/, "") || "";

async function handleResponse(res) {
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    let detail = text;
    try {
      const json = JSON.parse(text);
      detail = json.detail || JSON.stringify(json);
    } catch {}
    throw new Error(`API error ${res.status}: ${detail}`);
  }
  return res.json();
}

export async function fetchMeta() {
  const res = await fetch(`${BASE_URL}/api/meta`);
  return handleResponse(res);
}

export async function simulateStrategy(payload) {
  const res = await fetch(`${BASE_URL}/api/simulate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  return handleResponse(res);
}
