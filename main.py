from pathlib import Path
import sys

# garante que ./src entra no sys.path (Windows/Ubuntu)
sys.path.insert(0, str((Path(__file__).resolve().parent / "src").resolve()))

from csv_plotter.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
