"""Estadisticas de uso por especie, solo con train y leave-one-out (plan v2 §4, §6).


Cuentas fraccionales sobre C(o): un Pokemon no lider de un registro con v<2
cuenta como 'reserva' con la fraccion de brings de C(o) en que es reserva.
LOO: las features de un registro de train restan la contribucion de su propia
batalla (los dos lados), para que la etiqueta no se filtre por la feature.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from utils.brings import BRINGS, consistent_brings

GLOBAL_PRIOR = np.array([2 / 6, 2 / 6, 2 / 6])  # lider, reserva, fuera


def record_roles(obs: dict) -> np.ndarray:
    """(6, 3): fraccion de C(o) en que cada posicion es lider / reserva / fuera."""
    C = consistent_brings(obs["leads"], obs["revealed"])
    out = np.zeros((6, 3))
    for k in C:
        L, R = BRINGS[k]
        for i in range(6):
            out[i, 0 if i in L else (1 if i in R else 2)] += 1
    return out / len(C)


def battle_contrib(rec: dict) -> dict[str, np.ndarray]:
    """especie -> [n, lead, back, out] aportado por una batalla (ambos lados)."""
    c = defaultdict(lambda: np.zeros(4))
    for side in (0, 1):
        roles = record_roles(rec["obs"][side])
        for i, mon in enumerate(rec["teams"][side]):
            c[mon["species"]] += np.concatenate([[1.0], roles[i]])
    return dict(c)


class UsageStats:
    def __init__(self, counts: dict[str, list[float]] | None = None, prior: float = 5.0,
                 train_ids: list[str] | None = None):
        self.counts = {k: np.asarray(v, dtype=float) for k, v in (counts or {}).items()}
        self.prior = prior
        self.train_ids = sorted(train_ids or [])
        self.max_n = max((v[0] for v in self.counts.values()), default=1.0)

    @classmethod
    def fit(cls, train_records: list[dict], prior: float = 5.0) -> "UsageStats":
        counts = defaultdict(lambda: np.zeros(4))
        for r in train_records:
            for sp, v in battle_contrib(r).items():
                counts[sp] += v
        return cls(dict(counts), prior, [r["battle_id"] for r in train_records])

    def rates(self, species: str, exclude: dict[str, np.ndarray] | None = None) -> np.ndarray:
        c = self.counts.get(species, np.zeros(4)).copy()
        if exclude and species in exclude:
            c -= exclude[species]
        n = max(c[0], 0.0)
        return (np.maximum(c[1:], 0) + self.prior * GLOBAL_PRIOR) / (n + self.prior)

    def features(self, species: str, exclude=None) -> np.ndarray:
        """[P(lider), P(reserva), P(fuera), log1p(n)/log1p(n_max)] (m_x del estado)."""
        c = self.counts.get(species, np.zeros(4))
        n = c[0] - (exclude[species][0] if exclude and species in exclude else 0)
        return np.concatenate([self.rates(species, exclude),
                               [np.log1p(max(n, 0)) / np.log1p(self.max_n)]]).astype(np.float32)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"prior": self.prior, "train_ids": self.train_ids,
                                    "counts": {k: v.tolist() for k, v in self.counts.items()}}),
                        encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "UsageStats":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["counts"], d["prior"], d["train_ids"])
