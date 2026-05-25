from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple


@dataclass(frozen=True)
class MultiFactorBacktestConfig:
    provider_uri: str
    factors: List[Tuple[str, str]]  # [(name, expression), ...]
    region: str = "cn"
    instruments: str = "csi300"
    start_date: str = "2018-01-01"
    end_date: str = "2023-12-31"
    top_quantile: float = 0.2
    transaction_cost: float = 0.0015
    output_dir: Path = Path("factors/output")

    def validate(self) -> None:
        if not self.factors:
            raise ValueError("factors list cannot be empty.")
        seen_names: set[str] = set()
        for name, expr in self.factors:
            if not name or not expr:
                raise ValueError(f"Invalid factor: name='{name}', expr='{expr}'")
            if name in seen_names:
                raise ValueError(f"Duplicated factor name: {name}")
            seen_names.add(name)
        if not 0 < self.top_quantile <= 0.5:
            raise ValueError("top_quantile must be in (0, 0.5].")
        if self.transaction_cost < 0:
            raise ValueError("transaction_cost must be non-negative.")
