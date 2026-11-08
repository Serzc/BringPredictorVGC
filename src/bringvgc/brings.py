"""Espacio de acciones: 90 brings = C(6,2) lideres x C(4,2) reservas (eq. 2 del survey).

Indices de Pokemon 0..5 = posicion en la hoja OTS (mismo orden que el /team de Showdown).
bring_id = lead_pair_id * 6 + reserve_pair_id, con reserve_pair_id relativo a los
4 restantes (en orden creciente).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

LEAD_PAIRS: list[tuple[int, int]] = list(combinations(range(6), 2))          # 15
RESERVE_PAIRS: list[list[tuple[int, int]]] = []                              # 15 x 6
for _l in LEAD_PAIRS:
    rest = [i for i in range(6) if i not in _l]
    RESERVE_PAIRS.append(list(combinations(rest, 2)))

BRINGS: list[tuple[tuple[int, int], tuple[int, int]]] = [
    (LEAD_PAIRS[p], RESERVE_PAIRS[p][r]) for p in range(15) for r in range(6)]
N_BRINGS = len(BRINGS)  # 90

_INDEX = {}
for _k, (_L, _R) in enumerate(BRINGS):
    _INDEX[(_L, _R)] = _k


def bring_index(leads, reserves) -> int:
    return _INDEX[(tuple(sorted(leads)), tuple(sorted(reserves)))]


def bring_to_team_order(k: int) -> str:
    """'/team' de Showdown: posiciones 1..6, 2 lideres y luego 2 reservas."""
    L, R = BRINGS[k]
    return "".join(str(i + 1) for i in (*L, *R))


# Matrices auxiliares (90 x 6): pertenencia de cada Pokemon a lideres / reservas / fuera
LEAD_MASK = np.zeros((N_BRINGS, 6), dtype=np.float32)
RES_MASK = np.zeros((N_BRINGS, 6), dtype=np.float32)
for _k, (_L, _R) in enumerate(BRINGS):
    LEAD_MASK[_k, list(_L)] = 1
    RES_MASK[_k, list(_R)] = 1
OUT_MASK = 1 - LEAD_MASK - RES_MASK

# Indices auxiliares para la cabeza de reservas: (15, 6, 2)
RES_IDX = np.array([[list(p) for p in RESERVE_PAIRS[l]] for l in range(15)], dtype=np.int64)
LEAD_IDX = np.array([list(p) for p in LEAD_PAIRS], dtype=np.int64)  # (15, 2)


def consistent_brings(leads, revealed) -> list[int]:
    """C(o) de la eq. (6) del survey: brings b=(L,R) con V subset R subset c minus L.

    leads: 2 indices observados como lideres. revealed: indices de reservas reveladas (0..2).
    |C| = C(4-v, 2-v) in {1, 3, 6}.
    """
    L = tuple(sorted(leads))
    V = set(revealed)
    if len(L) != 2 or L[0] == L[1] or V & set(L) or len(V) > 2:
        raise ValueError(f"observacion invalida: L={leads} V={revealed}")
    p = LEAD_PAIRS.index(L)
    return [p * 6 + r for r, R in enumerate(RESERVE_PAIRS[p]) if V <= set(R)]


def consistent_mask(leads, revealed) -> np.ndarray:
    m = np.zeros(N_BRINGS, dtype=bool)
    m[consistent_brings(leads, revealed)] = True
    return m
