"""
agent/qa.py — QA answer engine: retrieval + context assembly + answer synthesis.

Pipeline:
  user query -> Retriever(classify+hybrid+rerank+dedup) -> coverage assessment
             -> context assembly -> answer synthesis (offline or LLM) -> Answer

Two execution modes:
  - offline:  Extracts commands/steps directly from retrieved chunks.
              Produces a concise, structured answer with no external API.
  - online:   Sends context to an LLM (via OpenRouter) for grounded synthesis.
              Still strictly grounded — the LLM may only cite retrieved evidence.

Key principles:
  - Answers ONLY from retrieved context (no hallucination)
  - Every claim cites [Source N] with full URL
  - Explicit uncertainty when evidence is insufficient
  - Multi-part queries get per-facet coverage check
"""

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

from rh_linux_docs_agent.search.retriever import (
    Retriever, classify_query, detect_interface_intent,
)
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
        return f"{self.guide_title} — {self.heading}"


@dataclass
class Answer:
    """Structured QA answer with citations and metadata."""
    query: str
    text: str
    sources: list[Source] = field(default_factory=list)
    confidence: str = "medium"          # high / medium / low / insufficient
    answer_mode: str = "exact"          # exact / partial / insufficient
    query_type: str = "procedure"       # procedure / concept / troubleshooting / reference
    interface_intent: str = "neutral"   # cli / gui / neutral
    interface_mismatch: str = ""        # non-empty when results don't match intent
    retrieval_time_s: float = 0.0
    generation_time_s: float = 0.0
    model: str = ""
    context_chunks: int = 0
    raw_context: str = ""


# ── System prompt (LLM online mode) ──────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a Red Hat Enterprise Linux (RHEL) documentation assistant.
You answer questions ONLY from the retrieved documentation chunks provided below.

## Strict grounding rules

1. **Evidence-only answers.** Every statement must come from the retrieved
   chunks. Do NOT introduce commands, file paths, CLI options, configuration
   directives, package names, or procedures not present verbatim in the
   provided context — even if you know them to be correct.

2. **No gap-filling.** If the context discusses a topic but lacks the specific
   command or step the user asks about, say so explicitly. Never fill the gap
   with commands from your own training data.

3. **Mandatory citations.** After every factual claim, append [Source N].
   At the end list every cited source:
   **Sources:**
   1. *Guide Title — Section Heading*
      URL

4. **Confidence header.** Begin with exactly one of:
   - **Confidence: High** — context directly and fully answers the question
   - **Confidence: Medium** — context is relevant but incomplete
   - **Confidence: Low** — context is only tangentially related
   - **Confidence: Insufficient** — not enough information to answer

5. **When confidence < High**, include:
   "For the complete procedure, consult the full section at [URL] or
   https://docs.redhat.com"

6. **Verbatim reproduction.** Reproduce commands/code exactly in fenced blocks.
   Do not modify, extend, or add flags not in the source.

7. **Version awareness.** Context is from RHEL 9. If the user asks about
   another version, note this.

8. **Format.** Markdown. Code in fenced blocks. Be concise.
"""


# ── Query facet extraction ───────────────────────────────────────────────────

_FACET_PATTERNS: list[tuple[str, re.Pattern, list[str]]] = [
    ("create", re.compile(r"\b(create|creating|make|making|add|new)\b", re.I),
     ["lvcreate", "creating", "create a", "create the"]),
    ("extend", re.compile(r"\b(extend|expanding|expand|enlarge|grow|resize|increase)\b", re.I),
     ["lvextend", "extending", "extend", "resize"]),
    ("shrink", re.compile(r"\b(shrink|reducing|reduce|decrease)\b", re.I),
     ["lvreduce", "shrink", "reducing"]),
    ("remove", re.compile(r"\b(remove|removing|delete|deleting)\b", re.I),
     ["lvremove", "removing", "remove"]),
    ("snapshot", re.compile(r"\bsnapshot\b", re.I), ["snapshot"]),
    ("configure", re.compile(r"\b(configure|configuring|set\s*up|setup)\b", re.I),
     ["configur", "set up"]),
    ("enable", re.compile(r"\b(enable|enabling)\b", re.I), ["enable", "enabling"]),
    ("disable", re.compile(r"\b(disable|disabling)\b", re.I), ["disable", "disabling"]),
    ("troubleshoot", re.compile(r"\b(troubleshoot|troubleshooting|debug|diagnose|fix)\b", re.I),
     ["troubleshoot", "debug", "diagnos"]),
    ("install", re.compile(r"\b(install|installing)\b", re.I), ["install"]),
    ("start", re.compile(r"\b(start|starting|restart|restarting)\b", re.I), ["start", "restart"]),
    ("stop", re.compile(r"\b(stop|stopping)\b", re.I), ["stop"]),
    ("list", re.compile(r"\b(list|show|display)\b", re.I), ["list", "show", "display"]),
]


def _extract_facets(query: str) -> list[str]:
    """Extract 2+ distinct operational facets from a query, or [] for single-op."""
    matched = []
    for label, pattern, _kw in _FACET_PATTERNS:
        if pattern.search(query):
            matched.append(label)
    return matched if len(matched) >= 2 else []


def _check_facet_coverage(
    facets: list[str],
    results: list[dict],
    sources: list[Source],
) -> dict[str, list[int]]:
    """Map each facet to 1-based source indices that cover it."""
    coverage: dict[str, list[int]] = {f: [] for f in facets}
    for facet in facets:
        keywords = []
        for label, _pat, kw in _FACET_PATTERNS:
            if label == facet:
                keywords = kw
                break
        for i, (r, s) in enumerate(zip(results, sources), start=1):
            combined = f"{s.heading}\n{r.get('chunk_text', '')}".lower()
            if any(kw.lower() in combined for kw in keywords):
                coverage[facet].append(i)
    return coverage


# ── Confidence assessment ────────────────────────────────────────────────────

def _assess_confidence(
    results: list[dict],
    query: str,
    interface_mismatch: str,
    facet_coverage: dict[str, list[int]] | None,
) -> tuple[str, str]:
    """
    Returns (confidence, answer_mode).
    confidence: high / medium / low / insufficient
    answer_mode: exact / partial / insufficient
    """
    if not results:
        return "insufficient", "insufficient"

    scores = [r.get("_rerank_score", r.get("_score", 0)) for r in results]
    top_score = scores[0]
    avg_score = sum(scores) / len(scores)
    is_cross_encoder = abs(top_score) > 1.5

    if is_cross_encoder:
        score_quality = (
            "strong" if top_score > 7 and avg_score > 5
            else "good" if top_score > 5 and avg_score > 4
            else "moderate" if top_score > 3
            else "weak"
        )
    else:
        score_quality = (
            "strong" if top_score > 0.5
            else "good" if top_score > 0.3
            else "moderate" if top_score > 0.1
            else "weak"
        )

    has_mismatch = bool(interface_mismatch)

    if facet_coverage:
        covered_facets = sum(1 for srcs in facet_coverage.values() if srcs)
        total_facets = len(facet_coverage)
        all_covered = covered_facets == total_facets
        most_covered = covered_facets >= total_facets * 0.5
    else:
        all_covered = True
        most_covered = True

    if len(results) >= 3:
        primary_guide = results[0].get("guide_slug", "")
        on_topic = sum(
            1 for r in results
            if r.get("guide_slug") == primary_guide
            or r.get("content_type") == "procedure"
        )
        noise_ratio = 1.0 - (on_topic / len(results))
    else:
        noise_ratio = 0.0

    if has_mismatch:
        return "low", "partial"
    if score_quality == "weak":
        return "insufficient", "insufficient"
    if all_covered and score_quality in ("strong", "good") and noise_ratio < 0.4:
        return "high", "exact"
    if all_covered and score_quality == "moderate":
        return "medium", "exact"
    if all_covered and noise_ratio >= 0.4:
        return "medium", "exact"
    if most_covered:
        return "medium", "partial"
    return "low", "partial"


# ── Context assembly ──────────────────────────────────────────────────────────

def assemble_context(
    results: list[dict], max_chars: int = 12_000,
) -> tuple[str, list[Source]]:
    """Build numbered [Source N] context string and Source list."""
    sources: list[Source] = []
    parts: list[str] = []
    total = 0

    for i, r in enumerate(results, 1):
        text = r.get("chunk_text", "")
        if len(text) > 3000:
            text = text[:3000] + "\n... [truncated]"

        block = (
            f"[Source {i}]\n"
            f"  Guide:   {r.get('guide_title', 'Unknown')}\n"
            f"  Section: {r.get('heading', 'Unknown')}\n"
            f"  Type:    {r.get('content_type', '')}\n"
            f"  URL:     {r.get('section_url', '')}\n"
            f"\n{text}\n"
        )
        if total + len(block) > max_chars:
            break
        parts.append(block)
        total += len(block)

        sources.append(Source(
            guide_title=r.get("guide_title", "Unknown"),
            heading=r.get("heading", "Unknown"),
            section_url=r.get("section_url", ""),
            guide_slug=r.get("guide_slug", ""),
            content_type=r.get("content_type", ""),
            chunk_text=text[:500],
            rerank_score=float(r.get("_rerank_score", r.get("_score", 0.0))),
        ))

    return "\n---\n".join(parts), sources


def _format_source_list(sources: list[Source]) -> str:
    lines = ["**Sources:**"]
    for i, s in enumerate(sources, 1):
        lines.append(f"{i}. *{s.citation_label}*")
        lines.append(f"   {s.section_url}")
    return "\n".join(lines)


# ── Chunk text helpers ────────────────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)
_NUMBERED_STEP_RE = re.compile(r"^(\d+)\.\s+(.+)", re.MULTILINE)


def _extract_commands(text: str) -> list[str]:
    """Pull fenced code blocks from a chunk."""
    return [m.group(1).strip() for m in _CODE_BLOCK_RE.finditer(text)]


def _extract_steps(text: str) -> list[str]:
    """Pull numbered procedure steps from a chunk (first line of each)."""
    return [m.group(0).strip() for m in _NUMBERED_STEP_RE.finditer(text)]


def _first_sentence(text: str, max_len: int = 200) -> str:
    """Return the first meaningful sentence from chunk text, skipping the heading."""
    lines = text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            continue
        # Skip lines that look like headings (all-digit prefix like "4.2.1.")
        if re.match(r"^\d+(\.\d+)*\.\s", line):
            continue
        # Skip prerequisite bullet lines
        if line.startswith("- ") and len(line) < 60:
            continue
        sent = line[:max_len]
        if len(line) > max_len:
            sent = sent.rsplit(" ", 1)[0] + "..."
        return sent
    return ""


# ── LLM call ─────────────────────────────────────────────────────────────────

def _call_llm(system: str, user_message: str) -> str:
    """Call the LLM via OpenRouter. Raises RuntimeError if no API key."""
    api_key = settings.openrouter_api_key
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set")

    resp = httpx.post(
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
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Structured logging ────────────────────────────────────────────────────────

def _log_query(answer: "Answer") -> None:
    """Emit a single structured log line with all query metadata."""
    top_sources = [
        {"title": s.heading[:60], "url": s.section_url, "score": round(s.rerank_score, 2)}
        for s in answer.sources[:3]
    ]
    logger.info(
        "QA query processed | query=%r | query_type=%s | intent=%s | "
        "confidence=%s | answer_mode=%s | mismatch=%s | chunks=%d | "
        "retrieval=%.3fs | generation=%.3fs | model=%s | top_sources=%s",
        answer.query[:120],
        answer.query_type,
        answer.interface_intent,
        answer.confidence,
        answer.answer_mode,
        bool(answer.interface_mismatch),
        answer.context_chunks,
        answer.retrieval_time_s,
        answer.generation_time_s,
        answer.model,
        top_sources,
    )


# ── QA Engine ─────────────────────────────────────────────────────────────────

class QAEngine:
    """
    Full QA pipeline: retrieve -> assess coverage -> synthesize answer.

    Works in two modes:
      - offline (no API key): synthesizes answer directly from chunk content
      - online (API key set): sends context to LLM for grounded generation
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
        n = top_n or settings.search_rerank_top_n

        # Step 0: Classify
        query_type = classify_query(query)
        interface_intent = detect_interface_intent(query)

        # Step 1: Retrieve
        t0 = time.time()
        results = self.retriever.retrieve(
            query, major_version=major_version, top_k=top_k, top_n=n,
            doc_type=doc_type, guide_slug=guide_slug,
        )
        retrieval_time = time.time() - t0

        # Step 2: Interface mismatch
        interface_mismatch = ""
        if results:
            interface_mismatch = results[0].get("_interface_mismatch", "") or ""

        # Step 3: Assemble context
        context, sources = assemble_context(results)

        # Step 4: Coverage
        facets = _extract_facets(query)
        facet_coverage = (
            _check_facet_coverage(facets, results, sources) if facets else None
        )
        confidence, answer_mode = _assess_confidence(
            results, query, interface_mismatch, facet_coverage,
        )

        # Step 5: Build LLM user message
        extra_instructions = ""
        if interface_mismatch:
            extra_instructions += (
                f"\n- IMPORTANT: {interface_mismatch} "
                f"State this clearly. Do NOT invent commands to fill the gap."
            )
        if facet_coverage:
            uncovered = [f for f, srcs in facet_coverage.items() if not srcs]
            if uncovered:
                extra_instructions += (
                    f"\n- IMPORTANT: The query asks about: "
                    f"{', '.join(facet_coverage.keys())}. "
                    f"Evidence does NOT cover: {', '.join(uncovered)}. "
                    f"State which parts are covered and which are not."
                )

        user_message = (
            f"## Question\n{query}\n\n"
            f"## Retrieved Documentation (RHEL {major_version})\n\n{context}\n\n"
            f"## Instructions\n"
            f"Answer using ONLY the chunks above. Cite as [Source N].\n"
            f"Do NOT introduce any commands or paths not in the chunks.\n"
            f"If evidence is partial, say what IS and what is NOT covered.\n"
            f"End with a **Sources:** list."
            f"{extra_instructions}"
        )

        # Step 6: Generate answer
        t1 = time.time()
        try:
            answer_text = _call_llm(SYSTEM_PROMPT, user_message)
            model_used = settings.llm_model
        except RuntimeError:
            answer_text = _build_offline_answer(
                query, results, sources, confidence, answer_mode,
                interface_mismatch, facet_coverage,
            )
            model_used = "offline (no API key)"
        except Exception as e:
            logger.error("LLM call failed: %s", e)
            answer_text = _build_offline_answer(
                query, results, sources, confidence, answer_mode,
                interface_mismatch, facet_coverage,
            )
            model_used = f"offline (LLM error: {e})"
        generation_time = time.time() - t1

        answer = Answer(
            query=query,
            text=answer_text,
            sources=sources,
            confidence=confidence,
            answer_mode=answer_mode,
            query_type=query_type,
            interface_intent=interface_intent,
            interface_mismatch=interface_mismatch,
            retrieval_time_s=round(retrieval_time, 3),
            generation_time_s=round(generation_time, 3),
            model=model_used,
            context_chunks=len(sources),
            raw_context=context,
        )

        _log_query(answer)
        return answer

    def retrieve_only(
        self,
        query: str,
        *,
        major_version: str = "9",
        top_k: int = 20,
        top_n: int | None = None,
    ) -> tuple[list[dict], str, list[Source]]:
        n = top_n or settings.search_rerank_top_n
        results = self.retriever.retrieve(
            query, major_version=major_version, top_k=top_k, top_n=n,
        )
        context, sources = assemble_context(results)
        return results, context, sources


# ── Offline answer synthesis ──────────────────────────────────────────────────

def _build_offline_answer(
    query: str,
    results: list[dict],
    sources: list[Source],
    confidence: str,
    answer_mode: str,
    interface_mismatch: str = "",
    facet_coverage: dict[str, list[int]] | None = None,
) -> str:
    """
    Synthesize a concise, structured answer from retrieved chunks.

    Instead of dumping raw chunk text, this:
      1. Extracts a short summary sentence per chunk
      2. Pulls out verbatim commands from fenced code blocks
      3. Groups by facet for multi-part queries
      4. Adds notes/limitations as needed
      5. Cites every claim with [Source N]
    """
    if not results:
        return (
            "**Confidence: Insufficient**\n\n"
            f"No relevant RHEL 9 documentation found for this query.\n"
            f"Please consult https://docs.redhat.com"
        )

    lines: list[str] = []

    # ── Mismatch warning ─────────────────────────────────────────────────
    if interface_mismatch:
        lines.append(f"> **Note:** {interface_mismatch}\n")

    # ── Multi-facet: grouped answer ──────────────────────────────────────
    if facet_coverage and any(facet_coverage.values()):
        covered = [f for f, srcs in facet_coverage.items() if srcs]
        uncovered = [f for f, srcs in facet_coverage.items() if not srcs]

        shown_sources: set[int] = set()

        for facet, source_indices in facet_coverage.items():
            lines.append(f"## {facet.title()}\n")

            if not source_indices:
                lines.append(
                    f"No specific **{facet}** instructions found in the "
                    f"retrieved evidence. Consult the full guides linked below.\n"
                )
                continue

            for idx in source_indices:
                if idx in shown_sources:
                    continue
                shown_sources.add(idx)
                i = idx - 1
                if i >= len(results) or i >= len(sources):
                    continue
                _append_synthesized_chunk(lines, results[i], sources[i], idx)

        # Remaining chunks under Notes
        remaining = [
            idx for idx in range(1, len(results) + 1)
            if idx not in shown_sources
        ]
        if remaining:
            lines.append("## Additional context\n")
            for idx in remaining:
                i = idx - 1
                if i >= len(results) or i >= len(sources):
                    continue
                _append_synthesized_chunk(lines, results[i], sources[i], idx)

    # ── Single-facet: sequential answer ──────────────────────────────────
    else:
        for i, (r, s) in enumerate(zip(results, sources), 1):
            _append_synthesized_chunk(lines, r, s, i)

    # ── Notes / limitations ──────────────────────────────────────────────
    notes: list[str] = []
    if confidence in ("medium", "low", "insufficient"):
        notes.append(
            "The retrieved evidence may not contain the complete procedure. "
            "Consult the linked sections or https://docs.redhat.com for full details."
        )
    if facet_coverage:
        uncovered = [f for f, srcs in facet_coverage.items() if not srcs]
        if uncovered:
            notes.append(
                f"No evidence found for: **{', '.join(uncovered)}**. "
                f"These operations may exist in sections not retrieved."
            )

    if notes:
        lines.append("---\n**Notes:**\n")
        for note in notes:
            lines.append(f"- {note}")
        lines.append("")

    # ── Source list ───────────────────────────────────────────────────────
    lines.append("---")
    lines.append(_format_source_list(sources))

    return "\n".join(lines)


def _append_synthesized_chunk(
    lines: list[str],
    result: dict,
    source: Source,
    source_num: int,
) -> None:
    """Append a concise synthesis of one chunk: summary + commands + citation."""
    text = result.get("chunk_text", "")
    iface = result.get("_interface", "")
    iface_tag = f" `[{iface}]`" if iface and iface != "neutral" else ""

    # Summary line
    summary = _first_sentence(text)
    if summary:
        lines.append(f"**{source.heading}**{iface_tag} [Source {source_num}]\n")
        lines.append(f"{summary}\n")
    else:
        lines.append(f"**{source.heading}**{iface_tag} [Source {source_num}]\n")

    # Commands — only fenced code blocks, verbatim
    commands = _extract_commands(text)
    if commands:
        for cmd in commands[:3]:  # max 3 code blocks per chunk
            lines.append(f"```\n{cmd}\n```\n")

    # Link
    lines.append(
        f"*Full section:* [{source.guide_title}]({source.section_url})\n"
    )
