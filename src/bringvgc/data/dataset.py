"""Carga de registros procesados con la barrera de test (F3: nada de inferencia en test).

Cualquier modelo que quiera leer registros de test tiene que pedir allow_test=True,
y eso solo lo hace la corrida preregistrada de la Fase 5.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path

from ..config import resolve
from .usage import UsageStats
from .vocab import Vocab

PROCESSED = "data/processed"


@lru_cache(maxsize=2)
def _all_records(proc: str) -> list[dict]:
    with open(resolve(proc) / "records.pkl", "rb") as f:
        return pickle.load(f)


def load_records(partition: str, allow_test: bool = False, proc: str = PROCESSED) -> list[dict]:
    if partition == "test" and not allow_test:
        raise PermissionError("Test congelado hasta la Fase 5 (allow_test=True solo en la corrida "
                              "preregistrada).")
    return [r for r in _all_records(proc) if r["valid"] and r["partition"] == partition]


def load_artifacts(proc: str = PROCESSED) -> tuple[Vocab, UsageStats]:
    p = resolve(proc)
    return Vocab.load(p / "vocab.json"), UsageStats.load(p / "usage.json")


def processed_available(proc: str = PROCESSED) -> bool:
    p = resolve(proc)
    return all((p / f).exists() for f in ("records.pkl", "vocab.json", "usage.json"))
