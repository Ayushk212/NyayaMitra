"""
Agent 3: UI & Interaction Agent (User Layer)
=============================================
RESPONSIBILITY: Format output for the user. Make it readable and interactive.
- Formats Agent 2's response into clean, structured UI blocks
- Provides suggestions, highlights, and simplified views
- Handles streaming response formatting

MUST NOT: Perform legal reasoning or data retrieval.
"""
import re
from app.models.schemas import ReasoningResult, Citation


def format_answer_blocks(reasoning: ReasoningResult) -> dict:
    """
    Transform Agent 2's reasoning result into structured UI blocks.
    This is what Agent 3 sends to the frontend.

    Output format:
    {
        "formatted_answer": "...",
        "ui_blocks": [...],
        "citations_panel": [...],
        "confidence": { "score": 0.82, "label": "High", "color": "green" },
        "suggestions": ["related queries"],
        "mode": "formal" | "eli5"
    }
    """
    # Format the main answer with markdown
    formatted = reasoning.answer

    # Build citation panel
    citations_panel = []
    for i, cit in enumerate(reasoning.citations, 1):
        citations_panel.append({
            "index": i,
            "case_name": cit.case_name,
            "paragraph_preview": cit.paragraph[:150] + "..." if len(cit.paragraph) > 150 else cit.paragraph,
            "paragraph_ref": cit.paragraph_ref,
        })

    # Build UI blocks (paragraphs, headings, citations)
    ui_blocks = _parse_answer_to_blocks(formatted)

    # Confidence label
    conf = reasoning.confidence
    if conf >= 0.8:
        conf_label, conf_color = "High Confidence", "green"
    elif conf >= 0.5:
        conf_label, conf_color = "Moderate Confidence", "yellow"
    else:
        conf_label, conf_color = "Low Confidence", "red"

    # Generate follow-up suggestions
    suggestions = _generate_suggestions(reasoning)

    return {
        "formatted_answer": formatted,
        "ui_blocks": ui_blocks,
        "citations_panel": citations_panel,
        "confidence": {
            "score": conf,
            "label": conf_label,
            "color": conf_color,
            "percentage": int(conf * 100),
        },
        "suggestions": suggestions,
        "mode": reasoning.mode,
    }


def _parse_answer_to_blocks(text: str) -> list[dict]:
    """Parse answer text into typed UI blocks for structured rendering."""
    blocks = []
    lines = text.split("\n")
    current_para = []

    for line in lines:
        stripped = line.strip()

        # Heading
        if stripped.startswith("##"):
            if current_para:
                blocks.append({"type": "paragraph", "content": "\n".join(current_para)})
                current_para = []
            blocks.append({"type": "heading", "content": stripped.lstrip("#").strip()})

        # Bullet list
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if current_para:
                blocks.append({"type": "paragraph", "content": "\n".join(current_para)})
                current_para = []
            blocks.append({"type": "bullet", "content": stripped[2:]})

        # Numbered list
        elif re.match(r'^\d+\.', stripped):
            if current_para:
                blocks.append({"type": "paragraph", "content": "\n".join(current_para)})
                current_para = []
            blocks.append({"type": "numbered", "content": re.sub(r'^\d+\.\s*', '', stripped)})

        # Citation reference
        elif "[" in stripped and "]" in stripped and ("v." in stripped or "vs" in stripped.lower()):
            if current_para:
                blocks.append({"type": "paragraph", "content": "\n".join(current_para)})
                current_para = []
            blocks.append({"type": "citation_inline", "content": stripped})

        # Empty line = paragraph break
        elif not stripped:
            if current_para:
                blocks.append({"type": "paragraph", "content": "\n".join(current_para)})
                current_para = []

        else:
            current_para.append(line)

    if current_para:
        blocks.append({"type": "paragraph", "content": "\n".join(current_para)})

    return blocks


def _generate_suggestions(reasoning: ReasoningResult) -> list[str]:
    """Generate contextual follow-up query suggestions."""
    suggestions = []

    # Extract case names mentioned for drill-down
    for cit in reasoning.citations[:2]:
        suggestions.append(f"Tell me more about {cit.case_name}")

    # Generic legal follow-ups
    generic = [
        "What are the exceptions to this principle?",
        "Are there any recent judgments on this topic?",
        "Explain the dissenting opinions",
        "What is the current legal position?",
    ]

    # Add generics until we have 4 suggestions
    for g in generic:
        if len(suggestions) >= 4:
            break
        suggestions.append(g)

    return suggestions[:4]


def format_search_results(results: list[dict]) -> list[dict]:
    """Format search results for the results page UI."""
    formatted = []
    for r in results:
        snippet = r.get("snippet", r.get("paragraph", ""))
        # Truncate and add ellipsis
        if len(snippet) > 250:
            snippet = snippet[:250].rsplit(" ", 1)[0] + "..."

        formatted.append({
            "id": r.get("case_id", r.get("id", "")),
            "title": r.get("case_name", r.get("title", "Untitled")),
            "court": r.get("court", "Unknown Court"),
            "date": r.get("date"),
            "citation": r.get("citation"),
            "snippet": snippet,
            "score": r.get("score", 0),
            "score_label": _score_label(r.get("score", 0)),
        })
    return formatted


def _score_label(score: float) -> dict:
    """Convert a score to a display label."""
    pct = int(score * 100) if score <= 1 else int(score)
    if pct >= 80:
        return {"text": f"{pct}% match", "color": "green"}
    elif pct >= 50:
        return {"text": f"{pct}% match", "color": "yellow"}
    else:
        return {"text": f"{pct}% match", "color": "red"}


def format_case_paragraphs(full_text: str) -> list[dict]:
    """
    Split case judgment into numbered paragraphs for the case viewer.
    Handles common Indian judgment formatting.
    """
    paragraphs = []
    # Split on double newlines or numbered paragraphs
    raw_paras = re.split(r'\n\s*\n|\n(?=\d+\.\s)', full_text)

    for i, para in enumerate(raw_paras, 1):
        text = para.strip()
        if not text:
            continue

        # Detect if this is a citation
        is_citation = bool(re.search(r'\(\d{4}\)\s+\d+\s+SCC\s+\d+|\[\d{4}\]\s+\d+\s+SCR', text))

        # Detect if this is an important paragraph (contains key legal terms)
        important_terms = ["held", "ratio", "principle", "established", "observed", "directed", "ordered"]
        is_important = any(term in text.lower() for term in important_terms)

        paragraphs.append({
            "index": i,
            "ref": f"¶{i}",
            "text": text,
            "is_citation": is_citation,
            "is_important": is_important,
        })

    return paragraphs
