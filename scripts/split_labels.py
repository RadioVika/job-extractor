"""Split the labeled postings into a fixed dev/test split, saved to data/splits.json.

Also writes dev_short: a random sample of DEV_SHORT_SIZE postings from dev, for quick runs.

Run once from the repo root. The split is permanent: if data/splits.json already exists,
dev and test are kept as they are and only a missing dev_short is added.
"""

import json
import random
import sys
from pathlib import Path

LABELS = Path("data/labels")
SPLITS = Path("data/splits.json")
SEED = 42
DEV_SIZE = 100
DEV_SHORT_SIZE = 20


def main() -> None:
    if SPLITS.exists():
        split = json.loads(SPLITS.read_text(encoding="utf-8"))
        if "dev_short" in split:
            sys.exit(f"{SPLITS} already has dev_short; the split is permanent, not overwriting")
    else:
        ids = [p.stem for p in sorted(LABELS.glob("p*.json"))]
        random.Random(SEED).shuffle(ids)
        split = {"seed": SEED, "dev": sorted(ids[:DEV_SIZE]), "test": sorted(ids[DEV_SIZE:])}
    split["dev_short"] = sorted(random.Random(SEED).sample(split["dev"], DEV_SHORT_SIZE))
    SPLITS.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    sizes = ", ".join(f"{name}: {len(split[name])}" for name in ("dev", "dev_short", "test"))
    print(f"{sizes} -> {SPLITS}")


if __name__ == "__main__":
    main()
