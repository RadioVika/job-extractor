import json
from pathlib import Path

import pytest

from job_extractor.schema import JobPosting

LABELS = Path(__file__).parent.parent / "data" / "labels"


@pytest.mark.parametrize("path", sorted(LABELS.glob("*.json")), ids=lambda p: p.name)
def test_label_matches_schema(path: Path):
    JobPosting.model_validate_json(path.read_text(encoding="utf-8"))


def test_splits_cover_all_labels():
    splits = json.loads((LABELS.parent / "splits.json").read_text(encoding="utf-8"))
    dev, test = set(splits["dev"]), set(splits["test"])
    assert len(dev) == 100
    assert len(test) == 50
    assert not dev & test
    assert dev | test == {p.stem for p in LABELS.glob("*.json")}
    assert len(splits["dev_short"]) == 20
    assert set(splits["dev_short"]) <= dev
