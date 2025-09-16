// api.js
// Fetch data from Flask APIs with basic error handling.

export async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

export async function loadAllData() {
  const [constructs, relationships, papers] = await Promise.all([
    fetchJSON('/api/constructs?limit=10000&connected_only=true').then(r => r.items || []),
    fetchJSON('/api/relationships?limit=10000').then(r => r.items || []),
    fetchJSON('/api/papers?limit=10000').then(r => r.items || []),
  ]);
  return { constructs, relationships, papers };
}


