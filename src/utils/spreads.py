"""A8: imputacion de naturaleza y stat points.

Los logs (showteam) traen vacios naturaleza y stat points. Regla fija, igual
para ambos lados y todos los metodos (config `a8_rule`):

- `role_spread` (por defecto): rol inferido de las categorias de los movimientos
  (datos de movimientos de poke-env, gen 9):
    * fisico   (mas movimientos de dano fisicos que especiales): Adamant, HP 2 / Atk 32 / Spe 32
    * especial (mas especiales que fisicos):                     Modest,  HP 2 / SpA 32 / Spe 32
    * mixto    (empate con al menos uno de dano):                 Hardy,   HP 2 / Atk 16 / SpA 16 / Spe 32
    * soporte  (ningun movimiento de dano):                       Hardy,   HP 32 / Def 17 / SpD 17
  Todos suman 66 y ninguno pasa de 32 por stat (limites verificados en el
  Showdown 913da36: sim/team-validator.ts, sim/dex-formats.ts).
- `random`: el reparto aleatorio de la Fase 0 (extract_teams.py), solo para
  reproducir esos resultados.

Alternativas descartadas (ver docs/EXPLICACION_ETAPA2.md): reparto neutro fijo
para todos (ignora que un atacante fisico nunca invierte en SpA); stats segun
uso real (no hay fuente abierta para Reg M-A); reparto consciente de Trick Room
(0 Spe) (otra regla mas que defender; queda como trabajo futuro).
"""
from __future__ import annotations

import random
from functools import lru_cache

from poke_env.data import GenData

MAX_PER_STAT = 32
TOTAL = 66
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
NEUTRAL_NATURES = ["Hardy", "Docile", "Bashful", "Quirky"]
NATURE, EVS = 5, 6

ROLE_SPREADS = {
    "physical": ("Adamant", [2, 32, 0, 0, 0, 32]),
    "special": ("Modest", [2, 0, 0, 32, 0, 32]),
    "mixed": ("Hardy", [2, 16, 0, 16, 0, 32]),
    "support": ("Hardy", [32, 0, 17, 0, 17, 0]),
}


@lru_cache(maxsize=1)
def _moves() -> dict:
    return GenData.from_gen(9).moves


def infer_role(moves: list[str]) -> str:
    dex = _moves()
    phys = spec = 0
    for m in moves:
        d = dex.get(m)
        if not d:
            continue
        if d.get("category") == "Physical":
            phys += 1
        elif d.get("category") == "Special":
            spec += 1
    if phys == 0 and spec == 0:
        return "support"
    if phys > spec:
        return "physical"
    if spec > phys:
        return "special"
    return "mixed"


def random_spread(rng: random.Random) -> list[int]:
    spread = [0] * 6
    for _ in range(TOTAL):
        stat = rng.choice([i for i in range(6) if spread[i] < MAX_PER_STAT])
        spread[stat] += 1
    return spread


def is_legal(spread: list[int]) -> bool:
    return len(spread) == 6 and all(0 <= x <= MAX_PER_STAT for x in spread) and sum(spread) <= TOTAL


def impute_packed(packed: str, rule: str = "role_spread", seed: int = 0) -> str:
    """Rellena naturaleza y stat points vacios de un equipo packed."""
    rng = random.Random(seed)
    mons = []
    for chunk in packed.split("]"):
        f = chunk.split("|")
        f += [""] * (12 - len(f))
        if rule == "role_spread":
            moves = [m.lower() for m in f[4].split(",") if m]
            moves = ["".join(ch for ch in m if ch.isalnum()) for m in moves]
            nature, spread = ROLE_SPREADS[infer_role(moves)]
        elif rule == "random":
            nature, spread = rng.choice(NEUTRAL_NATURES), random_spread(rng)
        else:
            raise ValueError(f"regla A8 desconocida: {rule}")
        if not f[NATURE]:
            f[NATURE] = nature
        if not f[EVS]:
            f[EVS] = ",".join(str(x) for x in spread)
        mons.append("|".join(f[:12]))
    return "]".join(mons)
