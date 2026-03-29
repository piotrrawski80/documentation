"""
scripts/run_api.py -- Start the RHEL Documentation Agent API server.

Usage:
    python scripts/run_api.py                  # default: localhost:8000
    python scripts/run_api.py --port 9000      # custom port
    python scripts/run_api.py --host 0.0.0.0   # listen on all interfaces
"""

import sys
import argparse
import logging
from pathlib import Path

# Ensure src/ is importable when running as a script.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the RHEL docs API server")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes (dev only)")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "rh_linux_docs_agent.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
