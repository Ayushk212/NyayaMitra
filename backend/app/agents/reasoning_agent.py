"""
Agent 2: Legal Reasoning Agent (Core Intelligence)
====================================================
RESPONSIBILITY: Generate correct legal answers using retrieved data ONLY.
- Reads retrieved documents from Agent 1
- Extracts relevant legal principles and arguments
- Generates structured answers WITH citations

MUST NOT: Guess, use external knowledge, or hallucinate.
If no relevant data: "I cannot find sufficient legal support for this query."
"""
import json
from typing import Optional

from app.core.config import GEMINI_API_KEY
from app.models.schemas import RetrievedChunk, ReasoningResult, Citation

# System prompts — the heart of Agent 2's behavior
FORMAL_SYSTEM_PROMPT = """You are a Legal Reasoning Agent for Indian law. You MUST follow these rules STRICTLY:

1. You are given RETRIEVED CASE LAW PASSAGES. Use ONLY these passages to answer.
2. NEVER use knowledge outside the provided passages.
3. Every claim MUST cite the specific case name and paragraph reference.
4. Format citations as: [Case Name, Para X] or [Case Name, Citation]
5. If the passages do not contain sufficient information, say:
   "I cannot find sufficient legal support for this query in the retrieved cases."
6. Structure your answer with:
   - A direct answer to the question
   - Supporting legal principles with citations
   - Key precedents referenced
7. Use formal legal language.
8. DO NOT hallucinate case names, dates, or legal principles.

RETRIEVED PASSAGES:
{context}

USER QUERY: {query}

Provide your answer with inline citations:"""

ELI5_SYSTEM_PROMPT = """You are a Legal Reasoning Agent for Indian law, but explaining to a non-lawyer. You MUST follow these rules:

1. Use ONLY the retrieved passages below to answer. NO outside knowledge.
2. Every claim must reference the specific case, but explain in simple language.
3. Use analogies and plain English. Avoid jargon.
4. If passages are insufficient, say: "I don't have enough information from the cases to answer this clearly."
5. Cite cases as: [Case Name] — explained simply.
6. Structure: Simple answer first, then supporting details.

RETRIEVED PASSAGES:
{context}

USER QUERY: {query}

Explain in simple terms with case references:"""


def build_context(chunks: list[RetrievedChunk]) -> str:
    """Build context string from retrieved chunks for the LLM prompt."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        ref = chunk.paragraph_ref or f"Chunk {i}"
        context_parts.append(
            f"[Source {i}] Case: {chunk.case_name}\n"
            f"Court: {chunk.court} | Date: {chunk.date or 'N/A'} | Citation: {chunk.citation or 'N/A'}\n"
            f"Paragraph ({ref}):\n{chunk.paragraph}\n"
            f"---"
        )
    return "\n".join(context_parts)


def extract_citations(answer: str, chunks: list[RetrievedChunk]) -> list[Citation]:
    """Extract/match citations from the answer text to source chunks."""
    citations = []
    seen = set()
    for chunk in chunks:
        name_lower = chunk.case_name.lower()
        # Check if this case is referenced in the answer
        if name_lower in answer.lower() or (chunk.citation and chunk.citation in answer):
            key = chunk.case_name
            if key not in seen:
                citations.append(Citation(
                    case_name=chunk.case_name,
                    paragraph=chunk.paragraph[:200],
                    paragraph_ref=chunk.paragraph_ref,
                ))
                seen.add(key)
    return citations


def calculate_confidence(chunks: list[RetrievedChunk], answer: str) -> float:
    """
    Calculate confidence score based on:
    - Average retrieval score of source chunks
    - Number of citations found in answer
    - Whether the answer admits uncertainty
    """
    if not chunks:
        return 0.0

    avg_score = sum(c.score for c in chunks) / len(chunks)

    # Count case references in answer
    ref_count = sum(1 for c in chunks if c.case_name.lower() in answer.lower())
    ref_ratio = ref_count / len(chunks) if chunks else 0

    # Penalize uncertain answers
    uncertainty_phrases = [
        "cannot find sufficient",
        "not enough information",
        "don't have enough",
        "no relevant",
    ]
    has_uncertainty = any(p in answer.lower() for p in uncertainty_phrases)

    if has_uncertainty:
        return round(max(0.1, avg_score * 0.3), 2)

    confidence = (avg_score * 0.5 + ref_ratio * 0.5)
    return round(min(confidence * 1.2, 0.99), 2)


async def reason(
    query: str,
    retrieved_chunks: list[RetrievedChunk],
    mode: str = "formal",
) -> ReasoningResult:
    """
    Main reasoning function — Agent 2's only public interface.

    Takes query + Agent 1's retrieval results.
    Generates a legally grounded answer with citations.
    Uses ONLY the provided retrieved data. NO external knowledge.

    Output format:
    {
        "answer": "...",
        "citations": [{"case": "...", "paragraph": "..."}],
        "confidence": 0.82
    }
    """
    if not retrieved_chunks:
        return ReasoningResult(
            answer="I cannot find sufficient legal support for this query. No relevant cases were found in the database. Please try refining your search terms.",
            citations=[],
            confidence=0.0,
            mode=mode,
        )

    context = build_context(retrieved_chunks)

    # Select prompt based on mode
    prompt_template = ELI5_SYSTEM_PROMPT if mode == "eli5" else FORMAL_SYSTEM_PROMPT
    prompt = prompt_template.format(context=context, query=query)

    # Call Gemini
    if not GEMINI_API_KEY:
        # Fallback: generate a mock answer using the context directly
        return _fallback_answer(query, retrieved_chunks, mode)

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        answer = response.text
    except Exception as e:
        return _fallback_answer(query, retrieved_chunks, mode)

    citations = extract_citations(answer, retrieved_chunks)
    confidence = calculate_confidence(retrieved_chunks, answer)

    return ReasoningResult(
        answer=answer,
        citations=citations,
        confidence=confidence,
        mode=mode,
    )


async def generate_summary(case_text: str, case_title: str) -> dict:
    """
    Generate a structured case summary.
    Extracts: Facts, Issues, Judgment, Ratio Decidendi.
    """
    prompt = f"""You are a legal case summarizer for Indian law. Analyze this judgment and extract:

1. **Facts**: Key factual background (2-3 paragraphs)
2. **Issues**: Legal questions the court addressed (bullet points)
3. **Judgment**: The court's decision and orders
4. **Ratio Decidendi**: The legal principle established

Case: {case_title}

Judgment Text (use ONLY this text):
{case_text[:8000]}

Provide a structured summary with the 4 sections above. Use formal legal language."""

    if not GEMINI_API_KEY:
        return {
            "facts": "Summary generation requires a Gemini API key. Please configure GEMINI_API_KEY.",
            "issues": "N/A",
            "judgment": "N/A",
            "ratio_decidendi": "N/A",
        }

    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(prompt)
        text = response.text

        # Parse sections
        sections = {"facts": "", "issues": "", "judgment": "", "ratio_decidendi": ""}
        current = None
        for line in text.split("\n"):
            lower = line.lower().strip()
            if "fact" in lower and ("**" in line or "#" in line):
                current = "facts"
                continue
            elif "issue" in lower and ("**" in line or "#" in line):
                current = "issues"
                continue
            elif "judgment" in lower and ("**" in line or "#" in line) and "ratio" not in lower:
                current = "judgment"
                continue
            elif "ratio" in lower and ("**" in line or "#" in line):
                current = "ratio_decidendi"
                continue
            if current:
                sections[current] += line + "\n"

        # If parsing failed, put everything in facts
        if not any(v.strip() for v in sections.values()):
            sections["facts"] = text

        return {k: v.strip() for k, v in sections.items()}
    except Exception:
        return {
            "facts": "Failed to generate summary.",
            "issues": "N/A",
            "judgment": "N/A",
            "ratio_decidendi": "N/A",
        }


def _fallback_answer(
    query: str, chunks: list[RetrievedChunk], mode: str
) -> ReasoningResult:
    """Fallback when no API key — construct answer from retrieved data directly."""
    parts = ["Based on the retrieved cases:\n"]
    citations = []

    for i, chunk in enumerate(chunks[:5], 1):
        parts.append(f"**{i}. {chunk.case_name}** ({chunk.court}, {chunk.date or 'N/A'})")
        parts.append(f"   {chunk.paragraph[:300]}...")
        parts.append("")
        citations.append(Citation(
            case_name=chunk.case_name,
            paragraph=chunk.paragraph[:200],
            paragraph_ref=chunk.paragraph_ref,
        ))

    parts.append("\n*Note: Full AI reasoning requires a Gemini API key. Showing raw retrieved passages.*")

    return ReasoningResult(
        answer="\n".join(parts),
        citations=citations,
        confidence=0.5,
        mode=mode,
    )
