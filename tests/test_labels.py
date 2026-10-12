from pathlib import Path

import pytest

from job_extractor.schema import JobPosting

LABELS = Path(__file__).parent.parent / "data" / "labels"


@pytest.mark.parametrize("path", sorted(LABELS.glob("*.json")), ids=lambda p: p.name)
def test_label_matches_schema(path: Path):
    JobPosting.model_validate_json(path.read_text(encoding="utf-8"))
