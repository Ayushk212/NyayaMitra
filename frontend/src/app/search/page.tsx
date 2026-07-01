"use client";
import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Navbar from "@/components/Navbar";
import { SearchSkeleton } from "@/components/Skeletons";
import { searchCases, listCases } from "@/lib/api";

const COURTS = ["Supreme Court", "Delhi HC", "Bombay HC", "Madras HC", "Calcutta HC"];
const CASE_TYPES = ["Constitutional", "Criminal", "Civil", "Tax", "Environmental"];

function SearchContent() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const activeQuery = searchParams.get("q") || "";

  const [query, setQuery] = useState(activeQuery);
  const [results, setResults] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [total, setTotal] = useState(0);
  const [court, setCourt] = useState("");
  const [caseType, setCaseType] = useState("");
  const [sortBy, setSortBy] = useState("relevance");
  const [page, setPage] = useState(1);
  const [error, setError] = useState("");

  const performSearch = async () => {
    setLoading(true); setError("");
    try {
      const params: Record<string, string> = {};
      if (court) params.court = court;
      if (caseType) params.case_type = caseType;
      if (sortBy) params.sort_by = sortBy;
      if (page > 1) params.page = page.toString();

      let data;
      if (activeQuery) {
        params.q = activeQuery;
        data = await searchCases(params);
      } else {
        data = await listCases(params);
      }
      setResults(data.results || []); setTotal(data.total || 0);
    } catch { setError("Search service unavailable. Make sure the backend is running."); setResults([]); }
    setLoading(false);
  };

  useEffect(() => {
    performSearch();
  }, [activeQuery, court, caseType, sortBy, page]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    if (query.trim() !== activeQuery) {
      if (query.trim()) {
        router.push(`/search?q=${encodeURIComponent(query.trim())}`);
      } else {
        router.push(`/search`);
      }
    } else {
      performSearch();
    }
  };

  const getScoreColor = (score: number) => {
    const pct = score > 1 ? score : score * 100;
    if (pct >= 70) return "badge-green";
    if (pct >= 40) return "badge-yellow";
    return "badge-red";
  };

  return (
    <div className="min-h-screen bg-white">
      <Navbar />
      <main className="pt-20 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Search */}
        <form onSubmit={handleSearch} className="mb-8">
          <div className="flex gap-3">
            <div className="flex-1 relative">
              <svg className="absolute left-4 top-1/2 -translate-y-1/2 w-5 h-5 text-[var(--text-muted)]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
              <input type="text" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search cases..."
                className="w-full pl-12 pr-4 py-3.5 input-clean" />
            </div>
            <button type="submit" className="btn-primary !rounded-xl">Search</button>
          </div>
        </form>

        <div className="flex gap-8">
          {/* Sidebar */}
          <aside className="hidden lg:block w-60 flex-shrink-0">
            <div className="card p-5 sticky top-24 space-y-6">
              <h3 className="font-semibold text-[var(--navy-900)] text-sm uppercase tracking-wider">Filters</h3>
              <div>
                <label className="text-xs text-[var(--text-muted)] font-medium uppercase tracking-wider">Court</label>
                <div className="mt-2 space-y-2">
                  <label className="flex items-center gap-2.5 text-sm text-[var(--text-secondary)] cursor-pointer hover:text-[var(--navy-900)] transition-colors">
                    <input type="radio" name="court" checked={court === ""} onChange={() => { setCourt(""); setPage(1); }} className="accent-orange-500 w-4 h-4" />
                    All Courts
                  </label>
                  {COURTS.map((c) => (
                    <label key={c} className="flex items-center gap-2.5 text-sm text-[var(--text-secondary)] cursor-pointer hover:text-[var(--navy-900)] transition-colors">
                      <input type="radio" name="court" checked={court === c} onChange={() => { setCourt(c); setPage(1); }} className="accent-orange-500 w-4 h-4" />
                      {c}
                    </label>
                  ))}
                </div>
              </div>
              <div>
                <label className="text-xs text-[var(--text-muted)] font-medium uppercase tracking-wider">Case Type</label>
                <div className="mt-2 space-y-2">
                  <label className="flex items-center gap-2.5 text-sm text-[var(--text-secondary)] cursor-pointer hover:text-[var(--navy-900)] transition-colors">
                    <input type="radio" name="caseType" checked={caseType === ""} onChange={() => { setCaseType(""); setPage(1); }} className="accent-orange-500 w-4 h-4" />
                    All Case Types
                  </label>
                  {CASE_TYPES.map((ct) => (
                    <label key={ct} className="flex items-center gap-2.5 text-sm text-[var(--text-secondary)] cursor-pointer hover:text-[var(--navy-900)] transition-colors">
                      <input type="radio" name="caseType" checked={caseType === ct} onChange={() => { setCaseType(ct); setPage(1); }} className="accent-orange-500 w-4 h-4" />
                      {ct}
                    </label>
                  ))}
                </div>
              </div>
              <button type={"button"} onClick={() => { setCourt(""); setCaseType(""); setPage(1); }}
                className="w-full py-2.5 text-sm text-[var(--text-secondary)] hover:text-[var(--navy-900)] border border-[var(--border-light)] rounded-xl hover:border-orange-300 transition-all">
                Clear Filters
              </button>
            </div>
          </aside>

          {/* Results */}
          <div className="flex-1">
            <div className="flex items-center justify-between mb-5">
              <p className="text-sm text-[var(--text-secondary)]">{loading ? "Searching..." : `${total} results found`}</p>
              <select value={sortBy} onChange={(e) => { setSortBy(e.target.value); setPage(1); }}
                className="text-sm input-clean px-3 py-2 !rounded-lg">
                <option value="relevance">Sort: Relevance</option>
                <option value="date">Sort: Date</option>
              </select>
            </div>

            {error && (
              <div className="card p-6 text-center border-orange-200">
                <p className="text-orange-600 text-sm mb-2">⚠️ {error}</p>
                <p className="text-xs text-[var(--text-muted)]">Run: <code className="bg-[var(--bg-subtle)] px-2 py-0.5 rounded text-[var(--navy-700)]">cd backend && python -m uvicorn app.main:app --reload</code></p>
              </div>
            )}

            {loading ? <SearchSkeleton /> : (
              <div className="space-y-3">
                {results.map((r, i) => (
                  <button key={r.id || i} onClick={() => router.push(`/case/${r.id}`)}
                    className="w-full text-left card p-5 group animate-fade-in-up" style={{ animationDelay: `${i * 0.05}s` }}>
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex-1">
                        <h3 className="font-semibold text-[var(--navy-900)] group-hover:text-[var(--orange-500)] transition-colors">{r.title}</h3>
                        <div className="flex flex-wrap gap-2 mt-1.5 text-xs">
                          <span className="text-[var(--text-secondary)]">{r.court}</span>
                          {r.date && <span className="text-[var(--text-muted)]">• {r.date}</span>}
                          {r.citation && <span className="text-orange-500/80">• {r.citation}</span>}
                        </div>
                        <p className="text-sm text-[var(--text-secondary)] mt-2 line-clamp-2">{r.snippet}</p>
                      </div>
                      <span className={`px-2.5 py-1 rounded-full text-xs font-medium whitespace-nowrap ${getScoreColor(r.score)}`}>
                        {r.score > 1 ? r.score : Math.round(r.score * 100)}% match
                      </span>
                    </div>
                  </button>
                ))}
                {!loading && results.length === 0 && activeQuery && !error && (
                  <div className="text-center py-16">
                    <p className="text-[var(--text-secondary)] text-lg">No results found</p>
                    <p className="text-[var(--text-muted)] text-sm mt-1">Try different keywords</p>
                  </div>
                )}
                {!loading && total > 10 && (
                  <div className="flex justify-between items-center mt-8 py-4 border-t border-[var(--border-light)]">
                    <button 
                      onClick={() => { setPage(p => Math.max(1, p - 1)); window.scrollTo(0,0); }} 
                      disabled={page === 1} 
                      className="px-4 py-2 text-sm font-medium text-[var(--navy-900)] bg-white border border-[var(--border-light)] rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed">
                      Previous
                    </button>
                    <span className="text-sm text-[var(--text-secondary)]">
                      Page {page} of {Math.ceil(total / 10)}
                    </span>
                    <button 
                      onClick={() => { setPage(p => Math.min(Math.ceil(total / 10), p + 1)); window.scrollTo(0,0); }} 
                      disabled={page >= Math.ceil(total / 10)} 
                      className="px-4 py-2 text-sm font-medium text-[var(--navy-900)] bg-white border border-[var(--border-light)] rounded-lg hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed">
                      Next
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}

export default function SearchPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-white"><Navbar /><main className="pt-20 max-w-7xl mx-auto px-4"><SearchSkeleton /></main></div>}>
      <SearchContent />
    </Suspense>
  );
}
