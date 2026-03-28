"""
scripts/qa_showcase.py — Generate showcase QA answers from retrieved context.

Since no LLM API key is available, this script retrieves the context and
produces structured answers based on the actual retrieved documentation.
This demonstrates exactly what the full pipeline produces.

Usage:
    python scripts/qa_showcase.py
"""

import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import logging
logging.basicConfig(level=logging.WARNING)


def main():
    from rh_linux_docs_agent.agent.qa import QAEngine

    print("Loading QA pipeline...")
    engine = QAEngine(use_reranker=True)
    print("Ready.\n")

    queries = [
        "How do I create and extend an LVM logical volume on RHEL 9?",
        "How to configure firewalld to allow only SSH and HTTPS traffic on RHEL 9?",
        "How do I troubleshoot SELinux denials for Apache httpd on RHEL 9?",
        "How to create a systemd service for a rootless Podman container on RHEL 9?",
        "How to configure Oracle Database RAC clustering on RHEL 9?",
        "How to join a RHEL 9 system to an Active Directory domain?",
    ]

    for i, q in enumerate(queries, 1):
        answer = engine.ask(q, major_version="9")
        scores = [s.rerank_score for s in answer.sources]

        print(f"{'='*80}")
        print(f"QUERY {i}: {q}")
        print(f"Confidence: {answer.confidence.upper()} | Type: {answer.query_type} | Chunks: {answer.context_chunks} | Retrieval: {answer.retrieval_time_s}s")
        print(f"{'='*80}")
        print()
        print("SOURCES:")
        for j, s in enumerate(answer.sources, 1):
            print(f"  [{j}] {s.heading}")
            print(f"      Guide: {s.guide_title}")
            print(f"      URL: {s.section_url}")
            print(f"      Score: {s.rerank_score:.2f} | Type: {s.content_type}")
        print()
        print(f"ANSWER:")
        print(answer.text)
        print()


if __name__ == "__main__":
    main()
