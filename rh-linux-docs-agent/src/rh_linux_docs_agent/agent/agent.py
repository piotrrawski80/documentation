"""
agent/agent.py — The pydantic-ai RHEL documentation agent.

This module defines the AI agent that answers RHEL troubleshooting questions.
The agent uses the docs_search and docs_compare tools to retrieve relevant
documentation, then uses an LLM (deepseek-chat via OpenRouter) to generate
accurate, well-cited answers.

How the agent works:
1. User asks a question (e.g., "How do I configure a static IP on RHEL 9?")
2. Agent calls docs_search("configure static IP", version="9")
3. Search returns top 5 matching documentation chunks
4. Agent reads the chunks and writes an answer grounded in the docs
5. Answer includes citations (guide title + URL) so users can verify

Why pydantic-ai?
- Clean tool definition with type hints (no JSON schema boilerplate)
- Built-in OpenAI-compatible API support (works with OpenRouter)
- Async-native (fast, non-blocking)
- Structured output support (for future extensions)

Why OpenRouter + deepseek-chat?
- OpenRouter provides access to many LLMs through one API key
- deepseek-chat-v3 is high-quality and very affordable (~$0.001/1K tokens)
- OpenRouter endpoint is OpenAI-compatible — pydantic-ai works natively

Usage:
    from rh_linux_docs_agent.agent.agent import create_agent
    agent = create_agent()
    result = await agent.run("How to configure firewalld on RHEL 9?")
    print(result.data)
"""

import logging
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from rh_linux_docs_agent.agent.tools import docs_search, docs_compare
from rh_linux_docs_agent.config import settings

logger = logging.getLogger(__name__)

# System prompt: defines the agent's persona, expertise, and behavior.
# This is sent to the LLM with every conversation to set the context.
SYSTEM_PROMPT = """You are an expert Red Hat Enterprise Linux (RHEL) system administrator
and documentation specialist. You help users troubleshoot, configure, and manage RHEL systems
across versions 8, 9, and 10.

You have access to the complete RHEL documentation through two tools:
- docs_search: Find relevant documentation for a specific topic or question
- docs_compare: Compare how something works across different RHEL versions

## How to answer questions

1. **Always search the docs first** — Use docs_search before answering. Don't rely on
   general knowledge alone; ground your answer in the actual RHEL documentation.

2. **Cite your sources** — Every answer must include the guide title and URL where
   you found the information. Format: "According to [Guide Title](URL)..."

3. **Note version differences** — If behavior differs between RHEL 8, 9, and 10,
   explicitly call this out. Use docs_compare when the user asks about version differences.

4. **Provide exact commands** — When documentation contains specific commands, show them
   verbatim in code blocks. Don't paraphrase commands.

5. **Warn about deprecated features** — If a feature is deprecated or replaced in a newer
   version, mention it clearly.

6. **Be precise about versions** — RHEL 8 uses `yum` (wraps DNF), RHEL 9+ uses `dnf` natively.
   NetworkManager configuration changed between versions. SELinux policy packaging evolved.
   Always check which version the user is asking about.

## Key RHEL expertise areas

- **Package management**: yum (RHEL 8) vs dnf (RHEL 9+), modularity, AppStreams, offline repos
- **Systemd**: service configuration, unit files, systemctl, journalctl, targets
- **SELinux**: enforcing/permissive modes, booleans, contexts, audit2allow, troubleshooting
- **Networking**: NetworkManager, nmcli, nmtui, firewalld, nftables (RHEL 9+), network scripts (deprecated)
- **Storage**: LVM, VDO (RHEL 8), Stratis (RHEL 9+), XFS, ext4, LUKS encryption, NFS, iSCSI
- **Security**: crypto policies, FIPS mode, OpenSSH hardening, certificate management, SCAP
- **Kernel**: sysctl parameters, tuned profiles, kernel modules, crash analysis
- **Containers**: podman, buildah, skopeo, rootless containers, systemd container integration
- **Subscriptions**: subscription-manager, RHSM, entitled content, offline registration
- **Upgrades**: leapp framework (RHEL 8→9, 9→10), in-place upgrade considerations
- **Web Console**: cockpit, browser-based management
- **Automation**: Ansible for RHEL, insights, remediation playbooks

## Response format

- Use markdown formatting with headers and code blocks
- Show commands in ```bash code blocks
- Show config file contents in appropriate code blocks
- Keep answers focused and actionable
- End with "Related topics you might want to explore:" when relevant

If you cannot find relevant documentation, say so clearly and suggest the user
check the official Red Hat documentation at https://docs.redhat.com
"""


def create_agent() -> Agent:
    """
    Create and configure the RHEL documentation agent.

    Sets up the pydantic-ai agent with:
    - OpenRouter API (OpenAI-compatible) for the LLM
    - docs_search and docs_compare tools
    - The RHEL expert system prompt

    Returns:
        Configured pydantic-ai Agent ready to run.

    Raises:
        ValueError: If OPENROUTER_API_KEY is not set in the environment.

    Example:
        agent = create_agent()
        result = await agent.run("How to configure static IP on RHEL 9?")
        print(result.data)
    """
    if not settings.openrouter_api_key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Create a .env file in the project root with:\n"
            "  OPENROUTER_API_KEY=your-key-here\n"
            "Get a free API key at https://openrouter.ai"
        )

    # Configure the LLM via OpenRouter.
    # OpenRouter provides an OpenAI-compatible API, so we use the OpenAI model
    # class but point it at OpenRouter's endpoint.
    model = OpenAIModel(
        model_name=settings.llm_model,
        provider=OpenAIProvider(
            base_url="https://openrouter.ai/api/v1",
            api_key=settings.openrouter_api_key,
        ),
    )

    # Create the agent with the model, system prompt, and tools
    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        # Tool registration — pydantic-ai automatically generates JSON schemas
        # from the function signatures and docstrings
        tools=[docs_search, docs_compare],
    )

    logger.info(f"Agent created with model: {settings.llm_model}")
    return agent


# Module-level agent instance — created lazily on first import
# We use a lazy pattern to avoid loading the model and API key at import time
_agent: Agent | None = None


def get_agent() -> Agent:
    """
    Get the global agent instance, creating it if needed.

    Uses a module-level singleton to avoid creating a new agent (and making
    a new connection to OpenRouter) on every request.

    Returns:
        The global Agent instance.
    """
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent
