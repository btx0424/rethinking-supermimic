"""File-path entry point."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from supermimic.play import main

if __name__ == "__main__":
    main()
