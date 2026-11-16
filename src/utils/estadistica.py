"""Estadistica de la evaluacion (plan v2 §9.2).
- Wilson 95 % por par (descriptivo).
- Intervalo t 95 % sobre pares (unidad = par; el win rate del par se promedia
  sobre semillas antes).
- Diferencia pareada por par entre dos metodos: media +- error estandar.
La regla preregistrada del 55 % esta en training/eval/tablas.py.
Sin scipy: cuantiles t tabulados (df 1..30) y aproximacion de Cornish-Fisher para df > 30.
"""
from __future__ import annotations

import math

import numpy as np

Z975 = 1.959963984540054
_T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179,
         2.160, 2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064,
         2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t975(df: int) -> float:
    if df < 1:
        return float("nan")
    if df <= 30:
        return _T975[df - 1]
    z = Z975
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def wilson(wins: int, n: int, z: float = Z975) -> tuple[float, float, float]:
    """(p, low, high) del IC de Wilson."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = wins / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h


def t_interval(x) -> tuple[float, float, float, float]:
    """(media, low, high, sd) con intervalo t al 95 % sobre las unidades de x."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    m = float(x.mean())
    if n < 2:
        return m, float("nan"), float("nan"), float("nan")
    sd = float(x.std(ddof=1))
    h = t975(n - 1) * sd / math.sqrt(n)
    return m, m - h, m + h, sd


def paired_difference(a, b) -> tuple[float, float]:
    """Diferencia pareada por par (a - b): media y error estandar."""
    d = np.asarray(a, float) - np.asarray(b, float)
    se = float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else float("nan")
    return float(d.mean()), se


def tabla_markdown(df) -> str:
    """DataFrame de pandas -> tabla Markdown (sin depender de tabulate)."""
    cols = [str(c) for c in df.columns]
    lineas = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for fila in df.itertuples(index=False):
        lineas.append("| " + " | ".join("" if v is None else str(v) for v in fila) + " |")
    return "\n".join(lineas)