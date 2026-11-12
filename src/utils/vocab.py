"""Vocabulario construido solo con train; <PAD>=0, <UNK>=1 (plan v2 §6)."""
from __future__ import annotations

import json
from pathlib import Path

CATS = ("species", "item", "ability", "move")


class Vocab:
    def __init__(self, maps: dict[str, dict[str, int]]):
        self.maps = maps

    @classmethod
    def build(cls, teams: list[list[dict]], min_count: int = 1) -> "Vocab":
        from collections import Counter
        cnt = {c: Counter() for c in CATS}
        for team in teams:
            for m in team:
                cnt["species"][m["species"]] += 1
                cnt["item"][m["item"]] += 1
                cnt["ability"][m["ability"]] += 1
                for mv in m["moves"]:
                    cnt["move"][mv] += 1
        maps = {}
        for c in CATS:
            toks = sorted(t for t, n in cnt[c].items() if n >= min_count)
            maps[c] = {"<PAD>": 0, "<UNK>": 1, **{t: i + 2 for i, t in enumerate(toks)}}
        return cls(maps)

    def idx(self, cat: str, tok: str) -> int:
        return self.maps[cat].get(tok, 1)

    def size(self, cat: str) -> int:
        return len(self.maps[cat])

    def coverage(self, teams: list[list[dict]]) -> dict[str, float]:
        """Fraccion de tokens (por categoria) presentes en el vocabulario de train."""
        hit = {c: 0 for c in CATS}
        tot = {c: 0 for c in CATS}
        for team in teams:
            for m in team:
                for c, toks in (("species", [m["species"]]), ("item", [m["item"]]),
                                ("ability", [m["ability"]]), ("move", m["moves"])):
                    for t in toks:
                        tot[c] += 1
                        hit[c] += t in self.maps[c]
        return {c: round(hit[c] / tot[c], 4) if tot[c] else 1.0 for c in CATS}

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.maps), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocab":
        return cls(json.loads(path.read_text(encoding="utf-8")))
