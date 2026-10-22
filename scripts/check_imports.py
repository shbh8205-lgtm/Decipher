import importlib
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("SESSION_SECRET", "import-check-only-not-a-secret")


def main() -> int:
    try:
        importlib.import_module("app.main")
    except ImportError as error:
        print(
            f"Import check failed: could not import app.main ({error}). "
            "Install dependencies from requirements.txt and retry.",
            file=sys.stderr,
        )
        return 1
    print("Import check passed: app.main imported successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
