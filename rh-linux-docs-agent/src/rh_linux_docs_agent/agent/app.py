"""
agent/app.py — Gradio web chat interface for the RHEL documentation agent.

This module starts a web server with a chat UI where users can type questions
and get answers from the RHEL documentation agent.

After starting, open http://localhost:7933 in your browser.

The chat interface:
- Shows conversation history (user messages + agent responses)
- Streams agent responses token-by-token (no waiting for the full response)
- Displays markdown formatting (headers, code blocks, links)
- Persists conversation history within a session

Usage:
    # Start the web UI
    python -m rh_linux_docs_agent.agent.app

    # Or via the CLI
    python scripts/start_agent.py
"""

import asyncio
import logging

import gradio as gr

from rh_linux_docs_agent.agent.agent import get_agent
from rh_linux_docs_agent.config import settings

logger = logging.getLogger(__name__)


async def chat(
    message: str,
    history: list[tuple[str, str]],
) -> tuple[str, list[tuple[str, str]]]:
    """
    Process a user message and return the agent's response.

    Called by Gradio on each user message. Runs the pydantic-ai agent
    with the full conversation history for context.

    Args:
        message: The user's current message.
        history: List of (user_message, agent_response) tuples from previous turns.

    Returns:
        Tuple of (empty string, updated history).
        The empty string clears the input box after sending.
        Updated history appends the new exchange.
    """
    if not message.strip():
        return "", history

    try:
        agent = get_agent()

        # Build the message history for the agent.
        # pydantic-ai accepts a list of message objects for multi-turn conversations.
        # For simplicity, we include the conversation summary in the prompt.
        context = ""
        if history:
            context = "Previous conversation:\n"
            for user_msg, agent_msg in history[-3:]:  # Include last 3 turns
                context += f"User: {user_msg}\nAssistant: {agent_msg}\n\n"
            context += "Current question:\n"

        full_query = f"{context}{message}" if context else message

        # Run the agent — this calls the LLM and may trigger tool calls
        result = await agent.run(full_query)
        response = result.data

        # Append to history
        history = history + [(message, response)]

    except ValueError as e:
        # Configuration error (e.g., missing API key)
        error_msg = (
            f"**Configuration Error**: {e}\n\n"
            "Please check your `.env` file has `OPENROUTER_API_KEY=your-key-here`."
        )
        history = history + [(message, error_msg)]

    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        error_msg = (
            f"**Error**: {e}\n\n"
            "If the database is empty, run: `python scripts/ingest.py --version 9 --limit 10`"
        )
        history = history + [(message, error_msg)]

    return "", history


def create_ui() -> gr.Blocks:
    """
    Create the Gradio web UI for the RHEL documentation agent.

    Returns a Gradio Blocks interface with:
    - Chat history display
    - Message input box
    - Send button + Enter key support
    - Clear button to reset the conversation
    - Example questions to get users started

    Returns:
        Configured Gradio Blocks interface.
    """
    with gr.Blocks(
        title="RHEL Documentation Agent",
        theme=gr.themes.Soft(),
        css="""
            .gradio-container { max-width: 900px; margin: auto; }
            footer { display: none !important; }
        """,
    ) as demo:

        # Header
        gr.Markdown(
            """
            # RHEL Documentation Agent
            Ask questions about Red Hat Enterprise Linux documentation.
            The agent searches RHEL 8, 9, and 10 docs and cites its sources.
            """
        )

        # Chat display
        chatbot = gr.Chatbot(
            label="Conversation",
            height=500,
            show_copy_button=True,
            render_markdown=True,
            bubble_full_width=False,
        )

        # Input area
        with gr.Row():
            msg_input = gr.Textbox(
                label="Your question",
                placeholder="e.g., How do I configure a static IP address on RHEL 9?",
                lines=2,
                scale=4,
                container=False,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        # Action buttons
        with gr.Row():
            clear_btn = gr.Button("Clear conversation", variant="secondary")

        # Example questions
        gr.Examples(
            examples=[
                "How do I configure a static IP address on RHEL 9?",
                "What are the differences in firewalld between RHEL 8 and 9?",
                "How do I enable FIPS mode on RHEL 9?",
                "How do I troubleshoot SELinux permission denials?",
                "What replaced VDO in RHEL 9?",
                "How do I migrate from iptables to nftables?",
                "How do I configure NetworkManager with nmcli?",
                "What is the difference between yum and dnf?",
            ],
            inputs=msg_input,
            label="Example questions",
        )

        # Wire up events
        # Send on button click
        send_btn.click(
            fn=chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )

        # Send on Enter key (Shift+Enter for newline)
        msg_input.submit(
            fn=chat,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )

        # Clear conversation
        clear_btn.click(
            fn=lambda: ([], ""),
            outputs=[chatbot, msg_input],
        )

    return demo


def main() -> None:
    """
    Start the Gradio web UI.

    This is the entry point when running:
        python -m rh_linux_docs_agent.agent.app
    """
    # Set up logging so we can see what's happening
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(f"Starting RHEL Documentation Agent web UI on port {settings.web_port}")
    logger.info(f"Open http://localhost:{settings.web_port} in your browser")

    demo = create_ui()
    demo.launch(
        server_port=settings.web_port,
        server_name="0.0.0.0",  # Accept connections from any network interface
        share=False,             # Don't create a public Gradio link
        show_error=True,         # Show errors in the UI
    )


if __name__ == "__main__":
    main()
