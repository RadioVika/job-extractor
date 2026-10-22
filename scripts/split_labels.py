"""Split the labeled postings into a fixed dev/test split, saved to data/splits.json.

Run once from the repo root. The split is permanent: the script refuses to overwrite
an existing data/splits.json.
"""

import json
import random
import sys
from pathlib import Path

LABELS = Path("data/labels")
SPLITS = Path("data/splits.json")
SEED = 42
DEV_SIZE = 100


def main() -> None:
    if SPLITS.exists():
        sys.exit(f"{SPLITS} already exists; the split is permanent, not overwriting")
    ids = [p.stem for p in sorted(LABELS.glob("p*.json"))]
    random.Random(SEED).shuffle(ids)
    split = {"seed": SEED, "dev": sorted(ids[:DEV_SIZE]), "test": sorted(ids[DEV_SIZE:])}
    SPLITS.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    print(f"dev: {len(split['dev'])}, test: {len(split['test'])} -> {SPLITS}")


if __name__ == "__main__":
    main()
