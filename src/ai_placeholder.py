from pathlib import Path
from typing import Optional


class Ai:
    def __init__(self, path: Path) -> None:
        pass

    def next_complete_move(self, game) -> tuple[Optional[int], int, Optional[int]]:
        raw = input("Enter move (<int|none>,<int>,<int|none>): ").strip()
        return parse_move(raw)


def parse_move(s: str) -> tuple[Optional[int], int, Optional[int]]:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError("Expected format: <int|none>,<int>,<int|none>")

    def to_int_or_none(value: str) -> Optional[int]:
        return int(value) if value else None

    a = to_int_or_none(parts[0])
    b = to_int_or_none(parts[1])
    c = to_int_or_none(parts[2])

    if b is None:
        raise ValueError("Middle value must be a valid integer")

    return a, b, c
