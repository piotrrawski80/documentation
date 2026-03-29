"""Start the Gradio web UI for the RHEL 9 documentation agent."""
import os
import sys
from pathlib import Path

# Set working directory to project root so relative paths (data/lancedb/) resolve correctly
project_root = Path(__file__).resolve().parent.parent
os.chdir(project_root)

# Ensure src/ is on the path when running outside pip install -e
src = str(project_root / "src")
if src not in sys.path:
    sys.path.insert(0, src)

from rh_linux_docs_agent.agent.app import main

if __name__ == "__main__":
    main()
