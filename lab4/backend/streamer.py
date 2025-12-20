from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List

DEFAULT_FIELDS: List[str] = ["datetime", "Open", "High", "Low", "Close", "Volume"]


@dataclass
class CandleStreamer:
    add_from_path: Path
    feed_path: Path
    state_path: Path

    def _load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text())
        return {"offset": 0, "initialized": False, "has_header": None}

    def _save_state(self, state: dict) -> None:
        self.state_path.write_text(json.dumps(state))

    def _detect_header_and_init(self, state: dict) -> tuple[list[str], int, bool]:
        """
        Supports add_from.csv with or without header.
        - If header exists, offset starts after header line
        - If no header, offset starts at 0
        """
        with self.add_from_path.open("r", newline="") as f:
            first = f.readline()
            if not first:
                raise RuntimeError("add_from.csv is empty")

            first_cells = [c.strip() for c in next(csv.reader([first]))]
            looks_like_header = (
                len(first_cells) >= 2
                and first_cells[0].lower() in ("datetime", "time", "timestamp")
                and any(x.lower() == "open" for x in first_cells)
            )

            if not state.get("initialized", False):
                state["has_header"] = bool(looks_like_header)
                state["offset"] = f.tell() if looks_like_header else 0
                state["initialized"] = True
                self._save_state(state)

        has_header = bool(state.get("has_header", False))
        offset = int(state["offset"])
        fieldnames = first_cells if has_header else DEFAULT_FIELDS
        return fieldnames, offset, has_header

    def pop_next_candle(self) -> Dict[str, Any]:
        state = self._load_state()
        fieldnames, offset, _has_header = self._detect_header_and_init(state)

        with self.add_from_path.open("r", newline="") as f:
            f.seek(offset)
            line = f.readline()
            if not line:
                raise RuntimeError("No more candles left in add_from.csv")

            state["offset"] = f.tell()
            self._save_state(state)

        cells = [c.strip() for c in next(csv.reader([line]))]

        if len(cells) != len(fieldnames):
            raise RuntimeError(
                f"Bad row length in add_from.csv: got {len(cells)} cells, expected {len(fieldnames)}"
            )

        row = dict(zip(fieldnames, cells))

        out: Dict[str, Any] = {}
        for k in DEFAULT_FIELDS:
            if k not in row:
                raise RuntimeError(f"Missing field '{k}' in add_from candle row")
            out[k] = row[k]
        return out

    def append_to_feed(self, candle: Dict[str, Any]) -> None:
        """
        Always append in DEFAULT_FIELDS order.
        """
        write_header = not self.feed_path.exists() or self.feed_path.stat().st_size == 0

        with self.feed_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DEFAULT_FIELDS)
            if write_header:
                w.writeheader()
            w.writerow({k: candle.get(k) for k in DEFAULT_FIELDS})
