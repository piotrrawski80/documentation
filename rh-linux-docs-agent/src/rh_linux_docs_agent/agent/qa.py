"""
agent/qa.py — QA answer engine: retrieval + context assembly + LLM generation.

Takes a user question, retrieves relevant RHEL documentation chunks via the
Retriever pipeline, assembles them into a prompt, and sends to an LLM for
grounded answer generation with citations.

Pipeline:
  user query → Retriever(classify+hybrid+rerank+dedup) → context assembly → LLM → Answer

Key principles:
  - Answers ONLY from retrieved context (no hallucination)
  - Every claim must cite section_url + heading
  - Explicit uncertainty when evidence is insufficient
  - Version-aware (RHEL 9 focus)

Usage:
    engine = QAEngine()
    answer = engine.ask("How to configure a static IP on RHEL 9?")
    print(answer.text)
    for src in answer.sources:
        print(f"  - {src.heading}: {src.section_url}")
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from rh_linux_docs_agent.search.retriever import Retriever, classify_query
from rh_linux_docs_agent.config import settings

logger = logging.getLogger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Source:
    """A cited source from the retrieved documentation."""
    guide_title: str
    heading: str
    section_url: str
    guide_slug: str
    content_type: str
    chunk_text: str = ""
    rerank_score: float = 0.0

    @property
    def citation_label(self) -> str:
        """Consistent 'Guide Title — Heading' label used in all citation lists."""
        return f"{self.guide_title} — {self.heading}"


@dataclass
class Answer:
    """Structured QA answer with citations and metadata."""
    query: str
    text: str
    sources: list[Source] = field(default_factory=list)
    confidence: str = "high"           # "high", "medium", "low", "insufficient"
    query_type: str = "procedure"      # "procedure", "concept", "troubleshooting", "reference"
    retrieval_time_s: float = 0.0
    generation_time_s: float = 0.0
    model: str = ""
    context_chunks: int = 0
    raw_context: str = ""              # The full context sent to LLM (for debugging)


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a Red Hat Enterprise Linux (RHEL) documentation assistant.
You answer questions ONLY from the retrieved documentation chunks provided below.

## Strict grounding rules

1. **Evidence-only answers.** Every statement you make must come from the
   retrieved chunks. Do NOT introduce commands, file paths, CLI options,
   configuration directives, package names, or procedures that do not appear
   verbatim in the provided context — even if you know them to be correct.

2. **No gap-filling.** If the retrieved context discusses a topic but does not
   include the specific command, flag, or step the user is asking about, you
   must say so explicitly. Use phrasing such as:
   - "The retrieved documentation covers [topic X], but the specific
     command/procedure for [Y] is not shown in the retrieved evidence."
   - "The retrieved chunks describe [general concept]; for the exact CLI
     syntax, consult the full guide linked below."
   Never fill the gap with commands from your own training data.

3. **Mandatory citations.** After every factual claim, append [Source N]
   where N is the chunk number. At the end of your answer, list every cited
   source in this exact format:

   **Sources:**
   1. *Guide Title — Section Heading*
      URL

   Always include both the guide title and full URL for every source.

4. **Confidence header.** Begin with exactly one of:
   - **Confidence: High** — the context directly and fully answers the question
   - **Confidence: Medium** — the context is relevant but incomplete or
     requires interpretation; say what IS covered and what is NOT
   - **Confidence: Low** — the context is only tangentially related
   - **Confidence: Insufficient** — the context does not contain enough
     information to answer

5. **When confidence is Medium, Low, or Insufficient**, always include:
   "For the complete procedure, consult the full section at [URL] or the
   official Red Hat documentation at https://docs.redhat.com"

6. **Verbatim reproduction.** When the context contains a command, config
   snippet, or code block, reproduce it exactly in a fenced code block.
   Do not modify, extend, or add flags/options that are not in the source.

7. **Complementary & conflicting sources.** If multiple chunks from different
   guides address the same question, note this. If they appear to conflict,
   flag the discrepancy and cite both.

8. **Version awareness.** The context comes from RHEL 9. If the user asks
   about RHEL 8 or 10, state that the retrieved evidence is RHEL 9-specific.

9. **Format.** Markdown. Code in fenced blocks. Be concise — focus on what
   the evidence supports, not on general background.
"""


# ── Context assembly ──────────────────────────────────────────────────────────

def assemble_context(results: list[dict], max_chars: int = 12000) -> tuple[str, list[Source]]:
    """
    Build the context string and source list from retrieval results.

    Each chunk is numbered [Source 1], [Source 2], ... for citation.
    Chunk headers include guide title, heading, content type, and full URL
    so the LLM can produce consistent citations.

    Truncates total context to max_chars to stay within LLM limits.

    Returns:
        (context_string, list_of_Source_objects)
    """
    sources: list[Source] = []
    parts: list[str] = []
    total_chars = 0

    for i, r in enumerate(results, start=1):
        text = r.get("chunk_text", "")
        heading = r.get("heading", "Unknown")
        guide_title = r.get("guide_title", "Unknown")
        section_url = r.get("section_url", "")
        guide_slug = r.get("guide_slug", "")
        content_type = r.get("content_type", "")
        rerank_score = r.get("_rerank_score", r.get("_score", 0.0))

        # Truncate individual chunks if needed
        if len(text) > 3000:
            text = text[:3000] + "\n... [truncated]"

        # Consistent header with full citation metadata
        chunk_header = (
            f"[Source {i}]\n"
            f"  Guide:   {guide_title}\n"
            f"  Section: {heading}\n"
            f"  Type:    {content_type}\n"
            f"  URL:     {section_url}\n"
        )
        chunk_block = f"{chunk_header}\n{text}\n"

        if total_chars + len(chunk_block) > max_chars:
            break

        parts.append(chunk_block)
        total_chars += len(chunk_block)

        sources.append(Source(
            guide_title=guide_title,
            heading=heading,
            section_url=section_url,
            guide_slug=guide_slug,
            content_type=content_type,
            chunk_text=text[:500],
            rerank_score=float(rerank_score),
        ))

    context = "\n---\n".join(parts)
    return context, sources


def _assess_confidence(results: list[dict], query: str) -> str:
    """
    Pre-assess confidence based on retrieval scores before LLM generation.

    Uses both the reranker score and the spread between top results.
    Returns "high", "medium", "low", or "insufficient".
    """
    if not results:
        return "insufficient"

    top_score = results[0].get("_rerank_score", results[0].get("_score", 0))

    # Detect score type: cross-encoder scores are typically in range [-10, 12],
    # while RRF/vector scores are in [0, 1]. Cross-encoder always > 1 for good matches.
    is_cross_encoder = abs(top_score) > 1.5

    if is_cross_encoder:
        # Cross-encoder scores: >7 strong, 4-7 moderate, <4 weak
        if top_score > 7:
            return "high"
        elif top_score > 4:
            if len(results) >= 3:
                avg_top3 = sum(r.get("_rerank_score", r.get("_score", 0)) for r in results[:3]) / 3
                if avg_top3 > 5:
                    return "high"
            return "medium"
        elif top_score > 2:
            return "low"
        else:
            return "insufficient"
    else:
        # RRF or vector similarity score (0 to 1 range)
        if top_score > 0.5:
            return "high"
        elif top_score > 0.3:
            return "medium"
        elif top_score > 0.005:
            return "low"
        else:
            return "insufficient"


# ── Citation formatting ──────────────────────────────────────────────────────

def _format_source_list(sources: list[Source]) -> str:
    """
    Render the numbered source list used at the bottom of both LLM and
    offline answers.  Each entry shows guide title, heading, and full URL.

    Format:
        **Sources:**
        1. *Guide Title — Heading*
           https://docs.redhat.com/...
    """
    lines = ["**Sources:**\n"]
    for i, s in enumerate(sources, start=1):
        lines.append(f"{i}. *{s.citation_label}*")
        lines.append(f"   {s.section_url}")
    return "\n".join(lines)


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(system: str, user_message: str) -> str:
    """
    Call the LLM via OpenRouter (OpenAI-compatible API).

    Uses httpx for a simple synchronous POST — no pydantic-ai dependency.

    Returns:
        The LLM's response text.

    Raises:
        RuntimeError: If the API call fails or no key is configured.
    """
    api_key = settings.openrouter_api_key
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. "
            "Create a .env file with: OPENROUTER_API_KEY=your-key-here"
        )

    response = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.llm_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            "temperature": 0.1,
            "max_tokens": 2000,
        },
        timeout=60.0,
    )
    response.raise_for_status()
    data = response.json()

    return data["choices"][0]["message"]["content"]


# ── QA Engine ─────────────────────────────────────────────────────────────────

class QAEngine:
    """
    Full QA pipeline: retrieve → assemble context → generate answer.

    Args:
        use_reranker: Whether to use cross-encoder reranking (default True).
        retriever:    Optional pre-built Retriever to reuse.
    """

    def __init__(
        self,
        use_reranker: bool = True,
        retriever: Retriever | None = None,
    ) -> None:
        self.retriever = retriever or Retriever(use_reranker=use_reranker)

    def ask(
        self,
        query: str,
        *,
        major_version: str = "9",
        top_k: int = 20,
        top_n: int | None = None,
        doc_type: str | None = None,
        guide_slug: str | None = None,
    ) -> Answer:
        """
        Answer a question using retrieved RHEL documentation.

        Args:
            query:          The user's question.
            major_version:  RHEL version to search (default "9").
            top_k:          Retrieval candidates.
            top_n:          Final chunks for context (default from settings).
            doc_type:       Optional doc_type filter.
            guide_slug:     Optional guide filter.

        Returns:
            Answer object with text, sources, confidence, and timing.
        """
        n = top_n or settings.search_rerank_top_n

        # Step 0: Classify query
        query_type = classify_query(query)

        # Step 1: Retrieve (includes rerank + threshold + dedup)
        t0 = time.time()
        results = self.retriever.retrieve(
            query,
            major_version=major_version,
            top_k=top_k,
            top_n=n,
            doc_type=doc_type,
            guide_slug=guide_slug,
        )
        retrieval_time = time.time() - t0

        # Step 2: Assemble context
        context, sources = assemble_context(results)
        confidence = _assess_confidence(results, query)

        # Step 3: Build user message
        user_message = (
            f"## Question\n{query}\n\n"
            f"## Retrieved Documentation (RHEL {major_version})\n\n{context}\n\n"
            f"## Instructions\n"
            f"Answer the question using ONLY the documentation chunks above.\n"
            f"- Cite sources as [Source N].\n"
            f"- Do NOT introduce any commands, file paths, CLI flags, config "
            f"directives, or package names that do not appear verbatim in the "
            f"retrieved chunks — even if you know them to be correct.\n"
            f"- If the chunks discuss the topic but lack the specific command or "
            f"procedure, say explicitly what IS covered and what is NOT, then "
            f"link to the full guide.\n"
            f"- If the documentation is insufficient, say so clearly.\n"
            f"- End your answer with a **Sources:** list in this format:\n"
            f"  1. *Guide Title — Section Heading*\n"
            f"     Full URL"
        )

        # Step 4: Generate answer
        t1 = time.time()
        try:
            answer_text = _call_llm(SYSTEM_PROMPT, user_message)
            model_used = settings.llm_model
        except RuntimeError as e:
            # No API key — return context-only answer
            answer_text = _build_offline_answer(query, results, sources, confidence)
            model_used = "offline (no API key)"
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            answer_text = (
                f"LLM generation failed: {e}\n\n"
                + _build_offline_answer(query, results, sources, confidence)
            )
            model_used = f"error: {e}"
        generation_time = time.time() - t1

        return Answer(
            query=query,
            text=answer_text,
            sources=sources,
            confidence=confidence,
            query_type=query_type,
            retrieval_time_s=round(retrieval_time, 3),
            generation_time_s=round(generation_time, 3),
            model=model_used,
            context_chunks=len(sources),
            raw_context=context,
        )

    def retrieve_only(
        self,
        query: str,
        *,
        major_version: str = "9",
        top_k: int = 20,
        top_n: int | None = None,
    ) -> tuple[list[dict], str, list[Source]]:
        """
        Retrieve and assemble context without calling the LLM.

        Useful for debugging or when you want to inspect the context
        before sending to an LLM.

        Returns:
            (results, context_string, sources)
        """
        n = top_n or settings.search_rerank_top_n
        results = self.retriever.retrieve(
            query, major_version=major_version, top_k=top_k, top_n=n,
        )
        context, sources = assemble_context(results)
        return results, context, sources


def _build_offline_answer(
    query: str,
    results: list[dict],
    sources: list[Source],
    confidence: str,
) -> str:
    """
    Build a structured answer when no LLM API is available.

    Shows the retrieved context in a readable format with consistent citations.
    """
    if not results:
        return (
            f"**Confidence: Insufficient**\n\n"
            f"No relevant documentation found for: \"{query}\"\n"
            f"Please consult https://docs.redhat.com"
        )

    lines = [
        f"**Confidence: {confidence.title()}**\n",
        f"*Retrieved {len(results)} relevant documentation chunks for RHEL 9.*\n",
        (
            "*(LLM generation unavailable — showing retrieved context directly.*\n"
            "*Only verbatim content from the retrieved chunks is shown below."
            " No commands or procedures have been added beyond what appears"
            " in the source documents.)*\n"
        ),
    ]

    for i, (r, s) in enumerate(zip(results, sources), start=1):
        text = r.get("chunk_text", "")[:600]
        lines.append(f"### [Source {i}] {s.heading}")
        lines.append(f"*Guide: {s.guide_title}*\n")
        lines.append(text)
        lines.append(f"\n**Full section:** {s.section_url}\n")

    if confidence in ("medium", "low", "insufficient"):
        lines.append(
            "---\n**Note:** The retrieved chunks may not contain the complete "
            "procedure. For full details, consult the linked sections above or "
            "the official Red Hat documentation at https://docs.redhat.com\n"
        )

    lines.append("---\n")
    lines.append(_format_source_list(sources))

    return "\n".join(lines)
