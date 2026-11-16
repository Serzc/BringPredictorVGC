"""Espacio de acciones: 90 brings = C(6,2) pares lideres x C(4,2) pares de reservas (eq. 2 del survey).

Indices de Pokemon 0..5 = posicion en la hoja OTS (mismo orden que el /team de Showdown).
bring_id = par_lider * 6 + par_reserva, con par_reserva contado sobre los 4 restantes
(en orden creciente). Asi el paso 1 (lideres, 15 opciones) y el paso 2 (reservas,
6 opciones) de la politica autorregresiva se leen directo del bring_id.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

# --- Enumeracion de los 15 pares lideres y, para cada uno, los 6 pares de reservas ---
LEAD_PAIRS: list[tuple[int, int]] = list(combinations(range(6), 2))          # 15
RESERVE_PAIRS: list[list[tuple[int, int]]] = []                              # 15 x 6
for _lider in LEAD_PAIRS:
    _resto = [i for i in range(6) if i not in _lider]
    RESERVE_PAIRS.append(list(combinations(_resto, 2)))

BRINGS: list[tuple[tuple[int, int], tuple[int, int]]] = [
    (LEAD_PAIRS[p], RESERVE_PAIRS[p][r]) for p in range(15) for r in range(6)]
N_BRINGS = len(BRINGS)  # 90

_INDEX = {(L, R): k for k, (L, R) in enumerate(BRINGS)}


def bring_index(leads, reserves) -> int:
    """(lideres, reservas) -> bring_id, sin importar el orden dentro de cada par."""
    return _INDEX[(tuple(sorted(leads)), tuple(sorted(reserves)))]


def bring_to_team_order(k: int) -> str:
    """Comando '/team' de Showdown: posiciones 1..6, primero 2 lideres y luego 2 reservas."""
    L, R = BRINGS[k]
    return "".join(str(i + 1) for i in (*L, *R))


# --- Matrices de pertenencia (se usan para pasar de puntajes por Pokemon a puntajes por par) ---
# LEAD_MON[p, m] = 1 si el Pokemon m esta en el par lider p                          (15, 6)
# RES_MON[p, r, m] = 1 si el Pokemon m esta en el par de reservas r dado el lider p  (15, 6, 6)
LEAD_MON = np.zeros((15, 6))
RES_MON = np.zeros((15, 6, 6))
for _p, _L in enumerate(LEAD_PAIRS):
    LEAD_MON[_p, list(_L)] = 1
    for _r, _R in enumerate(RESERVE_PAIRS[_p]):
        RES_MON[_p, _r, list(_R)] = 1

# Pertenencia de cada Pokemon a lideres / reservas / fuera, por bring (90, 6)
LEAD_MASK = np.repeat(LEAD_MON, 6, axis=0)
RES_MASK = RES_MON.reshape(90, 6)
OUT_MASK = 1 - LEAD_MASK - RES_MASK


def consistent_brings(leads, revealed) -> list[int]:
    """C(o) de la eq. (6) del survey: brings b=(L,R) con V subconjunto de R subconjunto de c menos L.

    leads: 2 indices observados como lideres. revealed: reservas reveladas (0, 1 o 2).
    |C| = C(4-v, 2-v), es decir 1, 3 o 6.
    """
    L = tuple(sorted(leads))
    V = set(revealed)
    if len(L) != 2 or L[0] == L[1] or V & set(L) or len(V) > 2:
        raise ValueError(f"observacion invalida: L={leads} V={revealed}")
    p = LEAD_PAIRS.index(L)
    return [p * 6 + r for r, R in enumerate(RESERVE_PAIRS[p]) if V <= set(R)]


def consistent_mask(leads, revealed) -> np.ndarray:
    """Version booleana (90,) de C(o)."""
    m = np.zeros(N_BRINGS, dtype=bool)
    m[consistent_brings(leads, revealed)] = True
    return m
