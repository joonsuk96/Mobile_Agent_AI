"""Run the LangChain-based Android agent: uv run python click.py."""
import sys
from pathlib import Path

# Works both with uv's editable package and a plain Python environment.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from mobile_jarvis_app.agent import main

if __name__ == "__main__":
    main()
