const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function searchCases(params: Record<string, string>) {
  const query = new URLSearchParams(params).toString();
  const res = await fetch(`${API_BASE}/api/search?${query}`);
  if (!res.ok) throw new Error("Search failed");
  return res.json();
}

export async function listCases(params: Record<string, string> = {}) {
  const query = new URLSearchParams(params).toString();
  const res = await fetch(`${API_BASE}/api/cases?${query}`);
  if (!res.ok) throw new Error("List cases failed");
  return res.json();
}

export async function getCase(caseId: string) {
  const res = await fetch(`${API_BASE}/api/cases/${caseId}`);
  if (!res.ok) throw new Error("Case not found");
  return res.json();
}

export async function getCaseSummary(caseId: string) {
  const res = await fetch(`${API_BASE}/api/cases/${caseId}/summary`, {
    method: "POST",
  });
  if (!res.ok) throw new Error("Summary generation failed");
  return res.json();
}

export async function getTrending() {
  const res = await fetch(`${API_BASE}/api/trending`);
  if (!res.ok) return { queries: [] };
  return res.json();
}

export function streamChat(
  query: string,
  mode: string,
  onStatus: (data: any) => void,
  onSources: (data: any) => void,
  onChunk: (text: string) => void,
  onComplete: (data: any) => void,
  onError: (err: Error) => void
) {
  const eventSource = new EventSource(
    `${API_BASE}/api/chat?_method=POST`
  );

  // Use fetch with SSE for POST
  fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, mode }),
  })
    .then(async (res) => {
      if (!res.ok) throw new Error("Chat failed");
      const reader = res.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (line.startsWith("event: ")) {
            const eventType = line.slice(7).trim();
            continue;
          }
          if (line.startsWith("data: ")) {
            const rawData = line.slice(6);
            try {
              const data = JSON.parse(rawData);
              // Detect event type from data content
              if (data.stage) onStatus(data);
              else if (Array.isArray(data)) onSources(data);
              else if (data.text !== undefined) onChunk(data.text);
              else if (data.citations) onComplete(data);
            } catch {}
          }
        }
      }
    })
    .catch(onError);
}

export async function quickChat(query: string, mode: string) {
  const res = await fetch(`${API_BASE}/api/chat/quick`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, mode }),
  });
  if (!res.ok) throw new Error("Chat failed");
  return res.json();
}
