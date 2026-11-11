"""File-path entry point for the independent MaxCoord trainer."""
import sys

from supermimic.train import main

if __name__ == "__main__":
    if not any(arg.startswith("algo=") for arg in sys.argv[1:]):
        sys.argv.append("algo=supermimic_maxcoord")
    main()
