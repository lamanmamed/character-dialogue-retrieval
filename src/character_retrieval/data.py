"""Loading dialogue CSV files and building one document per character."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"Character_name", "Line"}


def load_dialogue_csv(path: str | Path) -> pd.DataFrame:
    """Load the tab-separated dialogue format used by the experiment."""
    frame = pd.read_csv(path, sep="\t", skip_blank_lines=True)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Missing required columns: {missing_text}")
    return frame


def build_character_documents(
    frame: pd.DataFrame,
    max_lines_per_character: int,
) -> dict[str, str]:
    """Join the first N non-empty lines spoken by each character.

    The original experiment used at most 300 lines per character for training
    and at most 50 lines per character for validation and testing.
    """
    if max_lines_per_character < 1:
        raise ValueError("max_lines_per_character must be at least 1")

    lines_by_character: dict[str, list[str]] = defaultdict(list)

    for name, line in zip(frame["Character_name"], frame["Line"]):
        if pd.isna(line) or str(line).strip() == "":
            continue

        name = str(name)
        if len(lines_by_character[name]) >= max_lines_per_character:
            continue

        lines_by_character[name].append(str(line))

    return {
        name: " _EOL_ ".join(lines)
        for name, lines in lines_by_character.items()
    }
