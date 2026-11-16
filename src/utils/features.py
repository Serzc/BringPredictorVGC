"""Features hechas a mano del estado s = <c_self, c_opp, m> (plan v2 §4). Reemplazan al DeepSets.

Por cada Pokemon x de mi equipo, frente al equipo rival completo:

| feature                 | que mide                                                                 |
|-------------------------|--------------------------------------------------------------------------|
| log_p_lider             | log P(lider) de la especie en train (cuentas fraccionales sobre C(o))    |
| log_p_reserva           | log P(reserva)                                                           |
| log_p_fuera             | log P(se queda fuera)                                                    |
| uso_norm                | log(1+n)/log(1+n_max): que tan comun es la especie en train              |
| ofensiva_tipos          | media sobre rivales y de max_mov efectividad(mov de x, tipos de y)       |
| defensa_tipos           | media sobre rivales y de max_t efectividad(tipo t de y, tipos de x)      |
|                         | (mas alto = x recibe mas dano: es una debilidad)                         |
| especie=<nombre>        | one-hot de la especie (vocabulario de train; <UNK> si no se vio)         |

Nunca entra: numero de revelados, turnos, resultado, rating, nombres, historial Bo3
(son proxies del resultado o rompen el MDP; ver docs/MDP.md).
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from poke_env.data import GenData

from utils.usage import UsageStats
from utils.vocab import Vocab

NOMBRES_BASE = ["log_p_lider", "log_p_reserva", "log_p_fuera", "uso_norm",
                "ofensiva_tipos", "defensa_tipos"]


# --- Tabla de tipos y datos de movimientos (poke-env, gen 9) ---
@lru_cache(maxsize=1)
def _gen():
    return GenData.from_gen(9)


@lru_cache(maxsize=None)
def tipos_especie(species: str) -> tuple[str, ...]:
    """Tipos de una especie (en mayusculas, como los usa la tabla de poke-env)."""
    d = _gen().pokedex.get(species) or {}
    return tuple(t.upper() for t in d.get("types", ["Normal"]))


def efectividad(tipo_ataque: str, tipos_defensa) -> float:
    """Multiplicador de dano de un tipo de ataque contra un Pokemon con esos tipos."""
    chart = _gen().type_chart
    e = 1.0
    for t in tipos_defensa:
        e *= chart[t][tipo_ataque]
    return e


@lru_cache(maxsize=None)
def tipos_ataque(moves: tuple[str, ...]) -> tuple[str, ...]:
    """Tipos de los movimientos de dano de un Pokemon (se ignoran los de estado)."""
    dex = _gen().moves
    return tuple(dex[m]["type"].upper() for m in moves
                 if m in dex and dex[m].get("category") != "Status" and dex[m].get("basePower", 0) > 0)


def matriz_ofensiva(team_a: list[dict], team_b: list[dict]) -> np.ndarray:
    """(6, 6): off[i, j] = mejor efectividad de los movimientos de a_i contra b_j (0 si no tiene de dano)."""
    off = np.zeros((6, 6))
    for i, x in enumerate(team_a):
        atk = tipos_ataque(tuple(x["moves"]))
        for j, y in enumerate(team_b):
            off[i, j] = max((efectividad(t, tipos_especie(y["species"])) for t in atk), default=0.0)
    return off


def matriz_stab(team_a: list[dict], team_b: list[dict]) -> np.ndarray:
    """(6, 6): stab[i, j] = mejor efectividad de un tipo propio de b_j contra a_i (presion que recibe a_i)."""
    out = np.zeros((6, 6))
    for i, x in enumerate(team_a):
        tx = tipos_especie(x["species"])
        for j, y in enumerate(team_b):
            out[i, j] = max(efectividad(t, tx) for t in tipos_especie(y["species"]))
    return out


class Codificador:
    """Convierte (mi equipo, equipo rival) en una matriz (6, F) de features por Pokemon."""

    def __init__(self, vocab: Vocab, usage: UsageStats):
        self.vocab, self.usage = vocab, usage
        especies = sorted(vocab.maps["species"], key=vocab.maps["species"].get)
        self.n_especies = len(especies)                 # incluye <PAD> (0) y <UNK> (1)
        self.nombres = NOMBRES_BASE + [f"especie={e}" for e in especies]
        self.F = len(self.nombres)

    def mon_features(self, team_self: list[dict], team_opp: list[dict], exclude=None) -> np.ndarray:
        """Features (6, F). `exclude` = aporte de la propia batalla (leave-one-out en train)."""
        phi = np.zeros((6, self.F))
        # 1) uso de la especie en train (en log para que el modelo pueda imitar al baseline Usage)
        for i, m in enumerate(team_self):
            f = self.usage.features(m["species"], exclude)          # [P(L), P(R), P(F), uso_norm]
            phi[i, 0:3] = np.log(f[:3])
            phi[i, 3] = f[3]
        # 2) matchup de tipos contra los 6 rivales (promedio)
        phi[:, 4] = matriz_ofensiva(team_self, team_opp).mean(1)
        phi[:, 5] = matriz_stab(team_self, team_opp).mean(1)
        # 3) one-hot de especie
        for i, m in enumerate(team_self):
            phi[i, len(NOMBRES_BASE) + self.vocab.idx("species", m["species"])] = 1.0
        return phi
