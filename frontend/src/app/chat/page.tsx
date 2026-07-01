"use client";
import { useState, useRef, useEffect, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import Navbar from "@/components/Navbar";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface Message { role: "user" | "assistant"; content: string; citations?: any[]; confidence?: any; sources?: any[]; suggestions?: string[]; }

const SUGGESTIONS = [
  "What are the grounds for bail in non-bailable offenses?",
  "Explain the basic structure doctrine",
  "What is the right to privacy under Indian law?",
  "Section 304B dowry death requirements",
];

function ChatContent() {
  const searchParams = useSearchParams();
  const contextCase = searchParams.get("context") || "";
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState(contextCase ? `Tell me about ${contextCase}` : "");
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState<"formal" | "eli5">("formal");
  const [stage, setStage] = useState("");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages, stage]);

  const sendMessage = async (text?: string) => {
    const query = text || input.trim();
    if (!query || loading) return;
    setMessages((prev) => [...prev, { role: "user", content: query }]);
    setInput(""); setLoading(true); setStage("retrieving");

    try {
      const res = await fetch(`${API_BASE}/api/chat`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ query, mode }) });
      if (!res.ok) throw new Error("Chat failed");
      const reader = res.body?.getReader();
      if (!reader) throw new Error("No body");
      const decoder = new TextDecoder();
      let buffer = "", answerText = "", currentEventType = "", finalData: any = null;
      const sources: any[] = [];

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n"); buffer = lines.pop() || "";
        for (const line of lines) {
          if (line.startsWith("event: ")) { currentEventType = line.slice(7).trim(); continue; }
          if (line.startsWith("data: ")) {
            try {
              const data = JSON.parse(line.slice(6));
              if (currentEventType === "status" && data.stage) setStage(data.stage);
              else if (currentEventType === "sources" && Array.isArray(data)) sources.push(...data);
              else if (currentEventType === "answer_chunk" && data.text !== undefined) {
                answerText += data.text; setStage("answering");
                setMessages((prev) => { const u = [...prev]; const l = u.length - 1; if (u[l]?.role === "assistant") u[l] = { ...u[l], content: answerText }; else u.push({ role: "assistant", content: answerText }); return u; });
              } else if (currentEventType === "complete") finalData = data;
            } catch {}
          }
        }
      }
      setMessages((prev) => { const u = [...prev]; const l = u.length - 1; if (u[l]?.role === "assistant") u[l] = { ...u[l], content: answerText, citations: finalData?.citations || [], confidence: finalData?.confidence, sources, suggestions: finalData?.suggestions || [] }; return u; });
    } catch {
      setMessages((prev) => [...prev, { role: "assistant", content: "⚠️ Could not connect. Run: cd backend && python -m uvicorn app.main:app --reload" }]);
    }
    setLoading(false); setStage("");
  };

  return (
    <div className="min-h-screen bg-white flex flex-col">
      <Navbar />
      <main className="flex-1 flex flex-col pt-16">
        {/* Header */}
        <div className="border-b border-[var(--border-light)] px-4 py-3">
          <div className="max-w-3xl mx-auto flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-orange-400 to-orange-600 flex items-center justify-center text-white text-sm font-bold shadow-md shadow-orange-500/20">N</div>
              <div>
                <h1 className="text-sm font-semibold text-[var(--navy-900)]">AI Legal Assistant</h1>
                <p className="text-xs text-[var(--text-muted)]">3-Agent RAG Pipeline</p>
              </div>
            </div>
            <div className="flex items-center gap-1 p-1 bg-[var(--bg-subtle)] rounded-full">
              <button onClick={() => setMode("formal")} className={`px-4 py-1.5 rounded-full text-xs font-medium transition-all ${mode === "formal" ? "bg-[var(--navy-900)] text-white shadow-sm" : "text-[var(--text-secondary)] hover:text-[var(--navy-900)]"}`}>Formal</button>
              <button onClick={() => setMode("eli5")} className={`px-4 py-1.5 rounded-full text-xs font-medium transition-all ${mode === "eli5" ? "bg-[var(--orange-500)] text-white shadow-sm" : "text-[var(--text-secondary)] hover:text-[var(--navy-900)]"}`}>ELI5 🧒</button>
            </div>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto bg-[var(--bg-light)]">
          <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
            {messages.length === 0 && (
              <div className="text-center py-20 animate-fade-in-up">
                <p className="cursive-accent text-2xl mb-2">Ask anything</p>
                <h2 className="text-xl font-bold text-[var(--navy-900)] mb-2">Your Legal AI Assistant</h2>
                <p className="text-sm text-[var(--text-secondary)] mb-8 max-w-sm mx-auto">Every answer grounded in Indian case law with verifiable citations.</p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-lg mx-auto">
                  {SUGGESTIONS.map((s) => (
                    <button key={s} onClick={() => sendMessage(s)} className="text-left card p-3.5 text-sm text-[var(--text-secondary)] hover:text-[var(--navy-900)] hover:border-orange-300 transition-all">{s}</button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((msg, i) => (
              <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"} animate-fade-in-up`}>
                <div className={`max-w-[85%]`}>
                  <div className={`rounded-2xl px-4 py-3 ${msg.role === "user" ? "bg-[var(--navy-900)] text-white" : "card text-[var(--navy-700)]"}`}>
                    <div className="text-sm leading-relaxed whitespace-pre-wrap">{msg.content}</div>
                  </div>
                  {msg.role === "assistant" && msg.confidence && (
                    <div className="mt-2"><span className={`inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium badge-${msg.confidence.color}`}>{msg.confidence.label} ({msg.confidence.percentage}%)</span></div>
                  )}
                  {msg.role === "assistant" && msg.citations && msg.citations.length > 0 && (
                    <div className="mt-3 space-y-2">
                      <p className="text-xs text-[var(--text-muted)] font-medium uppercase tracking-wider">📚 Sources</p>
                      {msg.citations.map((c: any, j: number) => (
                        <div key={j} className="card p-3">
                          <p className="text-xs font-medium text-[var(--orange-500)]">{c.case_name}</p>
                          {c.paragraph_ref && <span className="text-[10px] text-[var(--text-muted)]">{c.paragraph_ref}</span>}
                          <p className="text-xs text-[var(--text-secondary)] mt-1 line-clamp-2">{c.paragraph_preview}</p>
                        </div>
                      ))}
                    </div>
                  )}
                  {msg.role === "assistant" && msg.suggestions && msg.suggestions.length > 0 && i === messages.length - 1 && (
                    <div className="mt-3 flex flex-wrap gap-2">
                      {msg.suggestions.map((s: string, j: number) => (
                        <button key={j} onClick={() => sendMessage(s)} className="text-xs px-3 py-1.5 rounded-full border border-[var(--border-light)] text-[var(--text-secondary)] hover:text-[var(--navy-900)] hover:border-orange-300 transition-all">{s}</button>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start animate-fade-in-up">
                <div className="card px-4 py-3 flex items-center gap-3">
                  <div className="flex gap-1">
                    <div className="w-2 h-2 rounded-full bg-orange-400 animate-bounce" style={{ animationDelay: "0ms" }} />
                    <div className="w-2 h-2 rounded-full bg-blue-400 animate-bounce" style={{ animationDelay: "150ms" }} />
                    <div className="w-2 h-2 rounded-full bg-emerald-400 animate-bounce" style={{ animationDelay: "300ms" }} />
                  </div>
                  <span className="text-xs text-[var(--text-secondary)]">
                    {stage === "retrieving" && "🔍 Agent 1: Searching case law..."}
                    {stage === "retrieved" && "📚 Found relevant cases"}
                    {stage === "reasoning" && "🧠 Agent 2: Analyzing..."}
                    {stage === "answering" && "✍️ Agent 3: Formatting..."}
                    {!stage && "Processing..."}
                  </span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input */}
        <div className="border-t border-[var(--border-light)] p-4 bg-white">
          <div className="max-w-3xl mx-auto">
            <form onSubmit={(e) => { e.preventDefault(); sendMessage(); }} className="flex gap-3">
              <input type="text" value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask a legal question..." disabled={loading}
                className="flex-1 px-4 py-3 input-clean disabled:opacity-50" />
              <button type="submit" disabled={loading || !input.trim()} className="btn-primary !rounded-xl disabled:opacity-40 flex items-center gap-2">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" /></svg>
                Send
              </button>
            </form>
            <p className="text-[10px] text-[var(--text-muted)] mt-2 text-center">Retrieval-first, AI-second. No hallucination by design.</p>
          </div>
        </div>
      </main>
    </div>
  );
}

export default function ChatPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-white flex items-center justify-center"><div className="w-8 h-8 border-2 border-orange-200 border-t-orange-500 rounded-full animate-spin" /></div>}>
      <ChatContent />
    </Suspense>
  );
}
