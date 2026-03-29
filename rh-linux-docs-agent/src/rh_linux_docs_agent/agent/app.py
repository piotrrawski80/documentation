"""
agent/app.py — Gradio web UI for the RHEL 9 documentation agent.

Uses QAEngine pipeline: hybrid retrieval -> cross-encoder reranking
-> coverage assessment -> answer synthesis (offline or LLM).

Works without an OpenRouter API key (offline mode).

Usage:
    python scripts/start_web.py
"""

import logging

import gradio as gr

from rh_linux_docs_agent.agent.qa import QAEngine, Answer
from rh_linux_docs_agent.config import settings

logger = logging.getLogger(__name__)

# ── Singleton QAEngine ───────────────────────────────────────────────────────

_engine: QAEngine | None = None


def _get_engine() -> QAEngine:
    global _engine
    if _engine is None:
        logger.info("Initializing QAEngine (first query will load models)...")
        _engine = QAEngine(use_reranker=True)
    return _engine


# ── Format answer for display ────────────────────────────────────────────────

def _format_answer(answer: Answer) -> str:
    """Build a single markdown string from an Answer for the chatbot."""
    parts: list[str] = []

    # Metadata bar
    conf_emoji = {
        "high": "\U0001f7e2", "medium": "\U0001f7e1",
        "low": "\U0001f7e0", "insufficient": "\U0001f534",
    }.get(answer.confidence, "\u26aa")

    # Version badge
    ver_badge = f"RHEL {answer.resolved_version}"
    if answer.version_source == "default":
        ver_badge += " (default)"

    badges: list[str] = [
        f"**Confidence: {answer.confidence.title()}**",
        f"Version: *{ver_badge}*",
        f"Type: *{answer.query_type}*",
    ]
    if answer.interface_intent != "neutral":
        badges.append(f"Intent: *{answer.interface_intent.upper()}*")
    if answer.answer_mode != "exact":
        badges.append(f"Coverage: *{answer.answer_mode}*")
    badges.append(f"Chunks: {answer.context_chunks}")
    badges.append(f"Retrieval: {answer.retrieval_time_s}s")
    if answer.generation_time_s > 0.01:
        badges.append(f"Generation: {answer.generation_time_s}s")

    badge_str = " \u00b7 ".join(badges)
    parts.append(f"{conf_emoji} {badge_str}")
    parts.append("")

    # Interface mismatch warning
    if answer.interface_mismatch:
        parts.append(f"> **Note:** {answer.interface_mismatch}\n")

    # Answer body
    parts.append(answer.text)

    # Source table (always appended for consistent clickable links)
    if answer.sources:
        parts.append("")
        parts.append("---")
        parts.append("**Cited sources:**\n")
        for i, s in enumerate(answer.sources, 1):
            score_str = f"{s.rerank_score:.1f}" if s.rerank_score else "\u2014"
            parts.append(
                f"{i}. **{s.guide_title}** \u2014 {s.heading} "
                f"(score {score_str}, {s.content_type})  \n"
                f"   {s.section_url}"
            )

    return "\n".join(parts)


# ── Chat handler ─────────────────────────────────────────────────────────────

def chat(message: str, history: list[dict]) -> tuple[str, list[dict]]:
    """Process a user question through the QAEngine pipeline."""
    if not message.strip():
        return "", history

    try:
        engine = _get_engine()
        answer = engine.ask(message)  # version auto-detected from query
        response = _format_answer(answer)
    except Exception as e:
        logger.error("QAEngine error: %s", e, exc_info=True)
        response = (
            f"**Error:** {e}\n\n"
            "If the database is empty, run:\n"
            "```\npython scripts/ingest.py --version 9\n```\n"
            "For other versions: `--version 8` or `--version 10`"
        )

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": response},
    ]
    return "", history


# ── UI layout ────────────────────────────────────────────────────────────────

def create_ui() -> gr.Blocks:
    with gr.Blocks() as demo:

        gr.Markdown(
            "# RHEL Documentation Agent\n"
            "Ask questions about Red Hat Enterprise Linux (RHEL 8, 9, or 10). "
            "Mention a version in your query or RHEL 9 is used by default. "
            "Answers are grounded strictly in retrieved documentation with citations."
        )

        chatbot = gr.Chatbot(label="Conversation", height=520)

        with gr.Row():
            msg_input = gr.Textbox(
                placeholder="e.g., How do I configure a static IP address on RHEL 9?",
                lines=2, scale=4, container=False, show_label=False,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Row():
            clear_btn = gr.Button("Clear conversation", variant="secondary", size="sm")

        gr.Examples(
            examples=[
                "How do I configure a static IP address on RHEL 9?",
                "How do I troubleshoot SELinux permission denials on RHEL 8?",
                "What's new in RHEL 10 for container management?",
                "Create and extend LVM logical volumes from the command line",
                "How to configure firewalld to allow SSH and HTTPS?",
                "How to run a rootless Podman container as a systemd service?",
            ],
            inputs=msg_input,
            label="Example questions",
        )

        send_btn.click(fn=chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
        msg_input.submit(fn=chat, inputs=[msg_input, chatbot], outputs=[msg_input, chatbot])
        clear_btn.click(fn=lambda: ([], ""), outputs=[chatbot, msg_input])

    return demo


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting RHEL 9 Documentation Agent on port %d", settings.web_port)
    logger.info("Open http://localhost:%d in your browser", settings.web_port)

    demo = create_ui()
    demo.launch(
        server_port=settings.web_port,
        server_name="0.0.0.0",
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 960px; margin: auto; }
            footer { display: none !important; }
        """,
    )


if __name__ == "__main__":
    main()
