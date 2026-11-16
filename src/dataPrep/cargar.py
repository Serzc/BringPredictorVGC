"""Carga de los registros limpios, con la barrera de test.

Cualquier codigo que quiera leer registros de test tiene que pedir allow_test=True,
y eso solo lo hace la corrida final (python src/main.py test).
"""
from __future__ import annotations

import pickle
from functools import lru_cache

from config.config import CLEAN_DIR
from utils.features import Codificador
from utils.usage import UsageStats
from utils.vocab import Vocab


@lru_cache(maxsize=1)
def _todos() -> list[dict]:
    with open(CLEAN_DIR / "records.pkl", "rb") as f:
        return pickle.load(f)


def load_records(partition: str, allow_test: bool = False) -> list[dict]:
    """Registros validos de una particion (train / val / test)."""
    if partition == "test" and not allow_test:
        raise PermissionError("Test congelado hasta la corrida final (allow_test=True solo en main.py test).")
    return [r for r in _todos() if r["valid"] and r["partition"] == partition]


def load_artifacts() -> tuple[Vocab, UsageStats]:
    """Vocabulario y estadisticas de uso, ambos calculados solo con train."""
    return Vocab.load(CLEAN_DIR / "vocab.json"), UsageStats.load(CLEAN_DIR / "usage.json")


def load_codificador() -> Codificador:
    return Codificador(*load_artifacts())


def clean_available() -> bool:
    return all((CLEAN_DIR / f).exists() for f in ("records.pkl", "vocab.json", "usage.json"))
