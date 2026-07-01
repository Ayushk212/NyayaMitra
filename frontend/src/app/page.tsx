"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";
import Navbar from "@/components/Navbar";
import { motion } from "framer-motion";

const FEATURES = [
  { label: "Case Search", active: true },
  { label: "AI Analysis", active: false },
  { label: "Citations", active: false },
  { label: "Summaries", active: false },
  { label: "Legal Research", active: false },
];

const TRENDING = [
  { text: "Right to Privacy", category: "Constitutional" },
  { text: "Bail in Non-Bailable Offenses", category: "Criminal" },
  { text: "Article 21 Right to Life", category: "Fundamental Rights" },
  { text: "Dowry Death Section 304B", category: "Criminal" },
  { text: "Basic Structure Doctrine", category: "Constitutional" },
  { text: "Sexual Harassment at Workplace", category: "Constitutional" },
];

const STATS = [
  { value: "12+", label: "Landmark Cases" },
  { value: "3", label: "AI Agents" },
  { value: "100%", label: "Citation Backed" },
  { value: "0%", label: "Hallucination" },
];

const TAB_CONTENT: Record<string, { title: string; description: string; icon: string; link: string }> = {
  "Case Search": {
    title: "Semantic Case Search",
    description: "Find exactly what you need with AI that understands legal context, not just keywords.",
    icon: "🔍",
    link: "/search"
  },
  "AI Analysis": {
    title: "Deep Legal Analysis",
    description: "Get comprehensive breakdowns of complex judgments powered by Gemini 2.0.",
    icon: "🧠",
    link: "/chat"
  },
  "Citations": {
    title: "Verified Citations",
    description: "Every claim is backed by pinpoint paragraph citations to prevent AI hallucination.",
    icon: "📝",
    link: "/search"
  },
  "Summaries": {
    title: "Instant Case Summaries",
    description: "Read 100-page judgments in minutes with structured facts, issues, and ratio decidendi.",
    icon: "📄",
    link: "/search"
  },
  "Legal Research": {
    title: "Multi-hop Research",
    description: "Connect precedents across different jurisdictions to build your winning argument.",
    icon: "⚖️",
    link: "/chat"
  }
};

export default function HomePage() {
  const router = useRouter();
  const [activeTab, setActiveTab] = useState("Case Search");

  return (
    <div id="main-scroll-container" className="h-[100dvh] w-full bg-white relative overflow-x-hidden overflow-y-auto snap-y snap-mandatory scroll-smooth">
      <Navbar />

      {/* Fluid Mesh Gradient Background */}
      <div className="fixed inset-0 pointer-events-none overflow-hidden z-0 bg-white">
        <div className="mesh-blob mesh-peach w-[60vw] h-[60vw] max-w-[800px] max-h-[800px] -top-[10%] -left-[10%] animate-mesh-1" />
        <div className="mesh-blob mesh-purple w-[60vw] h-[60vw] max-w-[800px] max-h-[800px] top-[20%] -right-[10%] animate-mesh-2" />
        <div className="mesh-blob mesh-pink w-[50vw] h-[50vw] max-w-[700px] max-h-[700px] top-[10%] left-[20%] animate-mesh-3" />
      </div>

      <main className="relative z-10 w-full h-full">
        {/* SECTION 1: Hero */}
        <section className="min-h-[100dvh] w-full snap-start flex flex-col justify-center items-center px-4 pt-16 relative">
          <div className="max-w-4xl mx-auto text-center w-full">
            <motion.p 
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: false, amount: 0.3 }}
              className="cursive-accent text-xl sm:text-2xl mb-3"
            >
              India&apos;s Smartest Legal Research
            </motion.p>

            <motion.h1 
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: false, amount: 0.3 }}
              transition={{ delay: 0.1 }}
              className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight text-[var(--navy-900)] mb-6 leading-tight"
            >
              The Next Gen Of{" "}
              <span className="relative inline-block">
                Legal AI
                <svg className="absolute -bottom-2 left-0 w-full" viewBox="0 0 200 12" fill="none">
                  <path d="M2 8 C40 2, 80 2, 100 6 C120 10, 160 10, 198 4" stroke="#ff6b35" strokeWidth="3" strokeLinecap="round" fill="none" opacity="0.4"/>
                </svg>
              </span>
              .
            </motion.h1>

            <motion.p 
              initial={{ opacity: 0, y: 30 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: false, amount: 0.3 }}
              transition={{ delay: 0.2 }}
              className="text-base sm:text-lg text-[var(--text-secondary)] max-w-xl mx-auto mb-10 leading-relaxed"
            >
              Search Supreme Court & High Court judgments with AI-powered insights.
              Every answer grounded in real case law with verifiable citations.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div 
              initial={{ opacity: 0, scale: 0.9 }}
              whileInView={{ opacity: 1, scale: 1 }}
              viewport={{ once: false, amount: 0.3 }}
              transition={{ delay: 0.3 }}
              className="flex flex-wrap items-center justify-center gap-4 mb-12"
            >
              <button 
                onClick={() => router.push("/search")} 
                className="btn-outline hover:scale-105 hover:shadow-lg hover:shadow-black/5 active:scale-95 transition-all duration-200"
              >
                Browse Cases
              </button>
              <button 
                onClick={() => router.push("/chat")} 
                className="btn-primary hover:scale-105 hover:shadow-lg hover:shadow-orange-500/25 active:scale-95 transition-all duration-200"
              >
                Ask AI Assistant
              </button>
            </motion.div>

            {/* Feature tabs */}
            <motion.div 
              initial={{ opacity: 0 }}
              whileInView={{ opacity: 1 }}
              viewport={{ once: false, amount: 0.3 }}
              transition={{ delay: 0.4 }}
              className="flex flex-wrap items-center justify-center gap-2 mb-8"
            >
              {FEATURES.map((feat) => (
                <button
                  key={feat.label}
                  onClick={() => setActiveTab(feat.label)}
                  className={`tab-pill hover:bg-[var(--bg-subtle)] hover:-translate-y-0.5 active:translate-y-0 transition-all ${activeTab === feat.label ? "tab-pill-active" : ""}`}
                >
                  {feat.label}
                </button>
              ))}
            </motion.div>

            {/* Dynamic Tab Content */}
            <motion.div
              key={activeTab}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3 }}
              className="max-w-2xl mx-auto bg-white/70 backdrop-blur-md border border-[var(--border-light)] rounded-2xl p-4 sm:p-5 shadow-lg shadow-black/5 flex flex-col sm:flex-row items-center gap-4 text-left"
            >
              <div className="text-3xl sm:text-4xl bg-[var(--bg-subtle)] p-3 rounded-xl">{TAB_CONTENT[activeTab]?.icon}</div>
              <div className="flex-1 text-center sm:text-left">
                <h3 className="font-bold text-[var(--navy-900)] text-lg">{TAB_CONTENT[activeTab]?.title}</h3>
                <p className="text-sm text-[var(--text-secondary)] mt-1">{TAB_CONTENT[activeTab]?.description}</p>
              </div>
              <button 
                onClick={() => router.push(TAB_CONTENT[activeTab]?.link)}
                className="text-sm font-semibold text-orange-600 hover:text-orange-700 whitespace-nowrap bg-orange-50 px-4 py-2 rounded-lg transition-colors hover:bg-orange-100 mt-2 sm:mt-0"
              >
                Explore →
              </button>
            </motion.div>
          </div>
        </section>

        {/* SECTION 2: Preview & Stats */}
        <section className="min-h-[100dvh] w-full snap-start flex flex-col justify-center items-center px-4 relative">
          <motion.div 
            initial={{ opacity: 0, scale: 0.95, y: 40 }}
            whileInView={{ opacity: 1, scale: 1, y: 0 }}
            viewport={{ once: false, amount: 0.4 }}
            transition={{ type: "spring", stiffness: 100, damping: 20 }}
            className="max-w-3xl w-full mx-auto mb-16"
          >
            <div className="card p-1.5 shadow-xl shadow-black/5 hover:shadow-2xl transition-shadow duration-500">
              <div className="bg-[var(--bg-subtle)] rounded-[12px] p-4 sm:p-6">
                <div className="flex items-center gap-2 mb-4">
                  <div className="flex gap-1.5">
                    <div className="w-3 h-3 rounded-full bg-red-400" />
                    <div className="w-3 h-3 rounded-full bg-yellow-400" />
                    <div className="w-3 h-3 rounded-full bg-green-400" />
                  </div>
                  <div className="flex-1 mx-4">
                    <div className="bg-white rounded-full py-1.5 px-4 text-xs text-[var(--text-muted)] border border-[var(--border-light)] text-center font-mono">
                      nyayamitra.ai/search
                    </div>
                  </div>
                </div>
                <div className="bg-white rounded-xl p-4 border border-[var(--border-light)]">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-400 to-orange-500 flex items-center justify-center text-white text-xs font-bold shadow-md">N</div>
                    <div className="flex-1">
                      <div className="text-sm font-semibold text-[var(--navy-900)]">Right to Privacy</div>
                      <div className="text-xs text-[var(--text-muted)]">10 results found</div>
                    </div>
                    <span className="text-xs px-2.5 py-1 rounded-full badge-green font-medium">98% match</span>
                  </div>
                  <div className="space-y-2.5">
                    {["Justice K.S. Puttaswamy v. Union of India", "Maneka Gandhi v. Union of India", "Kesavananda Bharati v. State of Kerala"].map((name, i) => (
                      <div key={i} className="flex items-center gap-3 p-2.5 rounded-lg hover:bg-[var(--bg-subtle)] hover:translate-x-1 cursor-pointer transition-all active:scale-[0.98]" onClick={() => router.push(`/search?q=${encodeURIComponent(name)}`)}>
                        <div className="w-1.5 h-1.5 rounded-full bg-orange-400" />
                        <span className="text-sm text-[var(--navy-700)] font-medium">{name}</span>
                        <span className="ml-auto text-[10px] text-[var(--text-muted)]">Supreme Court</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          <div className="w-full max-w-4xl mx-auto pt-8 border-t border-[var(--border-light)]">
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-8">
              {STATS.map((stat, i) => (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: false }}
                  transition={{ delay: 0.1 * i + 0.3 }}
                  key={stat.label} 
                  className="text-center group"
                >
                  <div className="text-3xl sm:text-4xl font-black text-[var(--navy-900)] group-hover:text-[var(--orange-500)] transition-colors">{stat.value}</div>
                  <div className="text-sm text-[var(--text-secondary)] mt-1 font-medium">{stat.label}</div>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* SECTION 3: Trending */}
        <section className="min-h-[100dvh] w-full snap-start flex flex-col justify-center items-center px-4 relative overflow-hidden">
          {/* subtle background blob for this section */}
          <div className="absolute -z-10 w-[600px] h-[600px] bg-gradient-to-tr from-purple-100/40 to-orange-100/40 blur-[80px] rounded-full top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 opacity-60" />

          <div className="max-w-6xl w-full mx-auto z-10">
            <motion.div 
              initial={{ opacity: 0, y: -20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: false, amount: 0.5 }}
              className="text-center mb-10"
            >
              <p className="cursive-accent text-xl mb-2">Popular Searches</p>
              <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold text-[var(--navy-900)]">Trending Legal Queries</h2>
            </motion.div>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
              {TRENDING.map((item, i) => (
                <motion.button
                  initial={{ opacity: 0, scale: 0.9 }}
                  whileInView={{ opacity: 1, scale: 1 }}
                  viewport={{ once: false, amount: 0.2 }}
                  transition={{ delay: 0.05 * i }}
                  key={item.text}
                  onClick={() => router.push(`/search?q=${encodeURIComponent(item.text)}`)}
                  className="p-6 text-left group hover:-translate-y-2 transition-all duration-300 bg-white/40 hover:bg-white/60 backdrop-blur-md border border-white/60 shadow-[0_8px_30px_rgb(0,0,0,0.04)] hover:shadow-[0_8px_30px_rgb(255,107,53,0.1)] rounded-2xl relative overflow-hidden"
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-white/40 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
                  <p className="text-sm md:text-base font-semibold text-[var(--navy-900)] group-hover:text-[var(--orange-600)] transition-colors line-clamp-2 h-10 md:h-12 relative z-10">
                    {item.text}
                  </p>
                  <span className="text-xs text-[var(--text-muted)] mt-4 inline-block px-3 py-1 rounded-full bg-white/50 border border-white/80 group-hover:bg-orange-100/50 group-hover:text-orange-600 transition-colors relative z-10">
                    {item.category}
                  </span>
                </motion.button>
              ))}
            </div>
          </div>
        </section>

        {/* SECTION 4: Architecture */}
        <section className="min-h-[100dvh] w-full snap-start flex flex-col justify-center items-center px-4 relative overflow-hidden">
          {/* subtle background blob for this section */}
          <div className="absolute -z-10 w-[800px] h-[800px] bg-gradient-to-tr from-blue-100/40 to-emerald-100/40 blur-[100px] rounded-full bottom-0 right-0 opacity-50" />

          <div className="max-w-5xl w-full mx-auto z-10">
            <motion.div 
              initial={{ opacity: 0, y: -20 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: false, amount: 0.5 }}
              className="text-center mb-12 md:mb-16"
            >
              <p className="cursive-accent text-xl mb-2">How It Works</p>
              <h2 className="text-2xl sm:text-3xl md:text-4xl font-bold text-[var(--navy-900)]">3-Agent Architecture</h2>
              <p className="text-sm md:text-base text-[var(--text-secondary)] mt-4 max-w-lg mx-auto">
                Your query flows through three specialized agents. No single agent does everything — that&apos;s what prevents hallucination.
              </p>
            </motion.div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 md:gap-8">
              {[
                { num: "01", title: "Retriever", desc: "Searches case law using BM25 + vector search. Finds relevant documents — never interprets.", color: "from-orange-100/80 to-orange-50/40 border-orange-200/50 hover:shadow-orange-500/20", numBg: "bg-gradient-to-br from-orange-400 to-orange-600", delay: 0.1 },
                { num: "02", title: "Reasoner", desc: "Generates answers using ONLY retrieved passages. Cites specific paragraphs. No hallucination.", color: "from-blue-100/80 to-blue-50/40 border-blue-200/50 hover:shadow-blue-500/20", numBg: "bg-gradient-to-br from-blue-400 to-blue-600", delay: 0.3 },
                { num: "03", title: "Formatter", desc: "Structures the output with citations, confidence scores, and follow-up suggestions.", color: "from-emerald-100/80 to-emerald-50/40 border-emerald-200/50 hover:shadow-emerald-500/20", numBg: "bg-gradient-to-br from-emerald-400 to-emerald-600", delay: 0.5 },
              ].map((agent) => (
                <motion.div 
                  initial={{ opacity: 0, y: 30 }}
                  whileInView={{ opacity: 1, y: 0 }}
                  viewport={{ once: false, amount: 0.3 }}
                  transition={{ delay: agent.delay, type: "spring", stiffness: 100 }}
                  key={agent.num} 
                  className={`bg-gradient-to-br ${agent.color} backdrop-blur-xl rounded-3xl p-6 md:p-8 border transition-all duration-300 hover:-translate-y-3 shadow-xl group`}
                >
                  <div className={`w-14 h-14 rounded-2xl ${agent.numBg} text-white flex items-center justify-center font-bold text-xl mb-6 shadow-lg group-hover:scale-110 transition-transform duration-300`}>
                    {agent.num}
                  </div>
                  <h3 className="text-xl font-bold text-[var(--navy-900)] mb-3">{agent.title}</h3>
                  <p className="text-sm md:text-base text-[var(--navy-800)]/80 leading-relaxed">{agent.desc}</p>
                </motion.div>
              ))}
            </div>
          </div>
        </section>

        {/* SECTION 5: Footer CTA & Footer */}
        <section className="min-h-[100dvh] w-full snap-start flex flex-col justify-between pt-20 relative overflow-hidden bg-gradient-to-b from-white to-gray-50">
          <div className="mesh-blob mesh-peach w-[600px] h-[600px] -top-20 left-1/4 opacity-30 absolute -z-10 animate-mesh-1" />
          <div className="mesh-blob mesh-purple w-[400px] h-[400px] bottom-0 right-1/4 opacity-30 absolute -z-10 animate-mesh-2" />
          
          <div className="flex-1 flex flex-col justify-center items-center text-center relative z-10 max-w-xl mx-auto px-4 w-full">
            <motion.div
              initial={{ opacity: 0, scale: 0.9, y: 20 }}
              whileInView={{ opacity: 1, scale: 1, y: 0 }}
              viewport={{ once: false }}
              transition={{ duration: 0.5 }}
            >
              <p className="cursive-accent text-xl mb-3">Ready to Research?</p>
              <h2 className="text-3xl sm:text-4xl md:text-5xl font-bold text-[var(--navy-900)] mb-6">Start Your Legal Research</h2>
              <p className="text-base md:text-lg text-[var(--text-secondary)] mb-10 max-w-md mx-auto">Search landmark cases, get AI analysis, and never miss a citation.</p>
              <div className="flex flex-wrap justify-center gap-4">
                <button 
                  onClick={() => router.push("/search")} 
                  className="btn-outline px-8 py-3 text-base hover:scale-105 hover:bg-gray-50 active:scale-95 transition-all duration-200"
                >
                  Browse Cases
                </button>
                <button 
                  onClick={() => router.push("/chat")} 
                  className="btn-primary px-8 py-3 text-base shadow-lg shadow-orange-500/30 hover:scale-105 hover:shadow-orange-500/50 active:scale-95 transition-all duration-200"
                >
                  Ask AI →
                </button>
              </div>
            </motion.div>
          </div>

          {/* Professional Footer */}
          <footer className="w-full bg-[#0a1128] text-white/70 py-8 px-6 z-20 mt-auto border-t border-white/10 shrink-0">
            <div className="max-w-7xl mx-auto flex flex-col md:flex-row justify-between items-center gap-6 text-sm">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-orange-400 to-orange-600 flex items-center justify-center text-white font-bold text-sm shadow-md">
                  N
                </div>
                <span className="font-bold text-white text-lg tracking-wide">NyayaMitra</span>
                <span className="ml-2 text-white/50">© {new Date().getFullYear()} All rights reserved.</span>
              </div>
              <div className="flex flex-wrap justify-center gap-6 md:gap-8 font-medium">
                <button onClick={() => router.push("/")} className="hover:text-orange-400 transition-colors">Privacy Policy</button>
                <button onClick={() => router.push("/")} className="hover:text-orange-400 transition-colors">Terms of Service</button>
                <button onClick={() => router.push("/")} className="hover:text-orange-400 transition-colors">Contact</button>
              </div>
            </div>
          </footer>
        </section>
      </main>
    </div>
  );
}
