"""
api/app.py -- Minimal production-ready FastAPI layer on top of QAEngine.

Endpoints:
    GET  /health    - Liveness / readiness probe.
    GET  /versions  - Which RHEL versions are indexed and their chunk counts.
    POST /ask       - Ask a documentation question.

Usage:
    uvicorn rh_linux_docs_agent.api.app:app --host 0.0.0.0 --port 8000
    python scripts/run_api.py            # convenience wrapper
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rh_linux_docs_agent.agent.qa import QAEngine, Answer as QAAnswer
from rh_linux_docs_agent.config import settings
from rh_linux_docs_agent.indexer.store import DocStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """POST /ask request body."""
    query: str = Field(..., min_length=1, max_length=2000, description="User question.")
    version: Literal["auto", "8", "9", "10"] = Field(
        "auto",
        description=(
            'RHEL version to search. "auto" detects from query text; '
            '"8", "9", or "10" forces that version.'
        ),
    )


class SourceOut(BaseModel):
    """One cited documentation source."""
    title: str
    heading: str
    url: str
    guide_slug: str
    content_type: str
    rerank_score: float


class AskResponse(BaseModel):
    """POST /ask response body."""
    answer: str
    confidence: str
    coverage: str
    resolved_version: str
    version_source: str
    query_type: str
    intent: str
    sources: list[SourceOut]
    latency_ms: int


class HealthResponse(BaseModel):
    status: str
    versions_indexed: list[str]


class VersionInfo(BaseModel):
    version: str
    chunks: int
    indexed: bool


class VersionsResponse(BaseModel):
    versions: list[VersionInfo]
    total_chunks: int


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Build and return the FastAPI application with a shared QAEngine."""

    engine_holder: dict[str, QAEngine | None] = {"engine": None}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: load models once
        logger.info("Loading QAEngine (embedding model + reranker)...")
        t0 = time.time()
        engine_holder["engine"] = QAEngine(use_reranker=True)
        logger.info("QAEngine ready in %.1fs", time.time() - t0)
        yield
        # Shutdown (nothing to clean up)

    api = FastAPI(
        title="RHEL Documentation Agent API",
        version="1.0.0",
        description="Ask questions about Red Hat Enterprise Linux documentation.",
        lifespan=lifespan,
    )

    def _engine() -> QAEngine:
        e = engine_holder["engine"]
        if e is None:
            raise HTTPException(503, detail="Engine is still loading. Try again shortly.")
        return e

    # -- GET /health --------------------------------------------------------
    @api.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        indexed: list[str] = []
        for v in sorted(settings.supported_versions):
            db_path = settings.db_path_for_version(v)
            if db_path.exists():
                try:
                    store = DocStore(db_path=db_path)
                    if store.get_total_count() > 0:
                        indexed.append(v)
                except Exception:
                    pass
        return HealthResponse(status="ok", versions_indexed=indexed)

    # -- GET /versions ------------------------------------------------------
    @api.get("/versions", response_model=VersionsResponse)
    async def versions() -> VersionsResponse:
        infos: list[VersionInfo] = []
        total = 0
        for v in sorted(settings.supported_versions):
            db_path = settings.db_path_for_version(v)
            chunks = 0
            indexed = False
            if db_path.exists():
                try:
                    store = DocStore(db_path=db_path)
                    chunks = store.get_total_count()
                    indexed = chunks > 0
                except Exception:
                    pass
            infos.append(VersionInfo(version=v, chunks=chunks, indexed=indexed))
            total += chunks
        return VersionsResponse(versions=infos, total_chunks=total)

    # -- POST /ask ----------------------------------------------------------
    @api.post("/ask", response_model=AskResponse)
    async def ask(req: AskRequest) -> AskResponse:
        t0 = time.time()

        major_version: str | None = None if req.version == "auto" else req.version

        engine = _engine()

        logger.info(
            "API /ask | query=%r | version=%s",
            req.query[:120],
            req.version,
        )

        answer: QAAnswer = engine.ask(req.query, major_version=major_version)

        sources_out = [
            SourceOut(
                title=s.guide_title,
                heading=s.heading,
                url=s.section_url,
                guide_slug=s.guide_slug,
                content_type=s.content_type,
                rerank_score=round(s.rerank_score, 3),
            )
            for s in answer.sources
        ]

        latency_ms = int((time.time() - t0) * 1000)

        logger.info(
            "API /ask complete | version=%s(%s) | confidence=%s | "
            "chunks=%d | sources=%d | latency=%dms",
            answer.resolved_version,
            answer.version_source,
            answer.confidence,
            answer.context_chunks,
            len(sources_out),
            latency_ms,
        )

        return AskResponse(
            answer=answer.text,
            confidence=answer.confidence,
            coverage=answer.answer_mode,
            resolved_version=answer.resolved_version,
            version_source=answer.version_source,
            query_type=answer.query_type,
            intent=answer.interface_intent,
            sources=sources_out,
            latency_ms=latency_ms,
        )

    return api


# Module-level app instance (for `uvicorn rh_linux_docs_agent.api.app:app`)
app = create_app()
