"""Model list prices from prices.yaml, for the cost metric."""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml

from job_extractor.extract import ROOT

PRICES = ROOT / "prices.yaml"


@dataclass(frozen=True)
class Price:
    """USD per 1M tokens, in effect since the entry's date in prices.yaml."""

    input: float
    output: float
    since: date | None = field(default=None, compare=False)

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input + output_tokens * self.output) / 1_000_000


def price_for(model: str, path: Path = PRICES, today: date | None = None) -> Price | None:
    """The price of model in effect today: its latest entry dated today or earlier.

    None if the file has no entry for the model.
    """
    today = today or date.today()
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    current = [e for e in entries if e["model"] == model and e["date"] <= today]
    if not current:
        return None
    entry = max(current, key=lambda e: e["date"])
    price = entry["price"]
    return Price(input=float(price["input"]), output=float(price["output"]), since=entry["date"])
