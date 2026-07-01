"use client";
import { useState, useEffect, use } from "react";
import { useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import { CaseSkeleton } from "@/components/Skeletons";
import { getCase, getCaseSummary } from "@/lib/api";

export default function CasePage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const router = useRouter();
  const [caseData, setCaseData] = useState<any>(null);
  const [summary, setSummary] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [showSummary, setShowSummary] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => { loadCase(); }, [id]);

  const loadCase = async () => {
    setLoading(true);
    try { setCaseData(await getCase(id)); } catch { setError("Case not found or backend unavailable."); }
    setLoading(false);
  };

  const generateSummary = async () => {
    setSummaryLoading(true);
    try { setSummary(await getCaseSummary(id)); setShowSummary(true); }
    catch { setSummary({ facts: "Failed to generate. Ensure Gemini API key is set.", issues: "", judgment: "", ratio_decidendi: "" }); setShowSummary(true); }
    setSummaryLoading(false);
  };

  if (loading) return <div className="min-h-screen bg-white"><Navbar /><main className="pt-20 max-w-6xl mx-auto px-4"><CaseSkeleton /></main></div>;
  if (error || !caseData) return (
    <div className="min-h-screen bg-white"><Navbar />
      <main className="pt-20 max-w-6xl mx-auto px-4 text-center py-20">
        <p className="text-[var(--text-secondary)] text-lg">⚠️ {error || "Case not found"}</p>
        <button onClick={() => router.back()} className="mt-4 text-sm text-[var(--orange-500)] hover:underline">← Back</button>
      </main>
    </div>
  );

  const paragraphs = caseData.paragraphs || [];

  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <main className="pt-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center gap-2 text-sm text-[var(--text-muted)] mb-6">
          <button onClick={() => router.push("/")} className="hover:text-[var(--orange-500)] transition-colors">Home</button><span>/</span>
          <button onClick={() => router.back()} className="hover:text-[var(--orange-500)] transition-colors">Search</button><span>/</span>
          <span className="text-[var(--text-secondary)] truncate max-w-xs">{caseData.title}</span>
        </div>

        <div className="flex gap-6">
          <aside className="hidden lg:block w-72 flex-shrink-0">
            <div className="card p-5 sticky top-24 space-y-5">
              <div>
                <h3 className="text-sm font-semibold text-[var(--navy-900)] mb-3">Case Details</h3>
                <dl className="space-y-3 text-sm">
                  <div><dt className="text-[var(--text-muted)] text-xs">Court</dt><dd className="text-[var(--navy-700)]">{caseData.court}</dd></div>
                  <div><dt className="text-[var(--text-muted)] text-xs">Date</dt><dd className="text-[var(--navy-700)]">{caseData.date || "N/A"}</dd></div>
                  <div><dt className="text-[var(--text-muted)] text-xs">Citation</dt><dd className="text-[var(--orange-500)] text-xs font-medium">{caseData.citation || "N/A"}</dd></div>
                  {caseData.judges?.length > 0 && <div><dt className="text-[var(--text-muted)] text-xs">Bench</dt><dd className="text-[var(--navy-700)] text-xs">{caseData.judges.join(", ")}</dd></div>}
                </dl>
              </div>
              <div className="space-y-2">
                <button onClick={generateSummary} disabled={summaryLoading}
                  className="w-full btn-primary !rounded-xl !py-2.5 text-sm disabled:opacity-50 flex items-center justify-center gap-2">
                  {summaryLoading ? <><div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" /> Generating...</> : <>✨ AI Summary</>}
                </button>
                <button onClick={() => router.push(`/chat?context=${encodeURIComponent(caseData.title)}`)}
                  className="w-full btn-outline !rounded-xl !py-2.5 text-sm">🤖 Ask AI</button>
              </div>
            </div>
          </aside>

          <div className="flex-1 min-w-0">
            <h1 className="text-2xl sm:text-3xl font-bold text-[var(--navy-900)] mb-3">{caseData.title}</h1>
            <div className="flex flex-wrap gap-3 mb-8 text-sm">
              <span className="px-3 py-1 rounded-full bg-orange-50 text-orange-600 border border-orange-200 font-medium">{caseData.court}</span>
              {caseData.date && <span className="px-3 py-1 rounded-full bg-[var(--bg-subtle)] text-[var(--text-secondary)]">{caseData.date}</span>}
              {caseData.case_type && <span className="px-3 py-1 rounded-full bg-blue-50 text-blue-600 border border-blue-200">{caseData.case_type}</span>}
            </div>
            <div className="space-y-4">
              {paragraphs.map((p: any, i: number) => (
                <div key={i} id={`para-${i}`} className={`group ${p.is_important ? "para-important" : ""}`}>
                  <div className="flex gap-3">
                    <span className="text-xs text-[var(--text-muted)] font-mono mt-1 min-w-[2rem]">{p.ref}</span>
                    <p className={`text-sm leading-relaxed ${p.is_citation ? "text-[var(--orange-500)] font-medium" : "text-[var(--navy-700)]"}`}>{p.text}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {showSummary && summary && (
            <aside className="hidden xl:block w-80 flex-shrink-0">
              <div className="card p-5 sticky top-24 space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="font-semibold text-[var(--navy-900)] text-sm">✨ AI Summary</h3>
                  <button onClick={() => setShowSummary(false)} className="text-[var(--text-muted)] hover:text-[var(--navy-900)] text-xs">✕</button>
                </div>
                {(["facts", "issues", "judgment", "ratio_decidendi"] as const).map((key) => {
                  const labels: Record<string, string> = { facts: "📋 Facts", issues: "❓ Issues", judgment: "⚖️ Judgment", ratio_decidendi: "📜 Ratio" };
                  if (!summary[key]) return null;
                  return (<div key={key}><h4 className="text-xs font-semibold text-[var(--orange-500)] mb-1">{labels[key]}</h4><p className="text-xs text-[var(--text-secondary)] leading-relaxed">{summary[key]}</p></div>);
                })}
              </div>
            </aside>
          )}
        </div>
      </main>
    </div>
  );
}
