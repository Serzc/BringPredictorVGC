"""Clase padre MetodoBring: todo lo que elige un bring (baselines, BC, agente RL) hereda de aqui.

Asi la arena trata a todos igual: le pide a cada metodo una distribucion sobre
los 90 brings con probs(mi_equipo, equipo_rival) y elige con elegir(...).
La politica de batalla pi_b es la misma para todos (sim/jugador.py); el metodo
solo cambia el Team Preview.

| Metodo        | Archivo                 | Modo                  | Determinista |
|---------------|-------------------------|-----------------------|--------------|
| random        | baselines/random.py     | uniforme              | no           |
| usage         | baselines/usage.py      | softmax del puntaje   | no (rival de la RQ) |
| usage_argmax  | baselines/usage.py      | argmax del puntaje    | si           |
| tipos         | baselines/tipos.py      | regla de tipos        | si           |
| planner       | baselines/planner.py    | argmax E_q[v-hat]     | si dado v-hat |
| bc            | models/bc.py            | rho_BC greedy         | si dado el checkpoint |
| agent         | models/rl_agent.py      | politica RL greedy    | si dado el checkpoint |
"""
from __future__ import annotations

import numpy as np


class MetodoBring:
    """Interfaz comun. Las subclases implementan probs(); elegir() es igual para todas."""

    name = "base"
    deterministic = False   # True: siempre se juega el argmax (no se muestrea)

    def probs(self, team_self: list[dict], team_opp: list[dict]) -> np.ndarray:
        """Distribucion (90,) sobre los brings, dados mi equipo y el rival (hojas OTS parseadas)."""
        raise NotImplementedError

    def elegir(self, team_self, team_opp, rng: np.random.Generator, greedy: bool = False) -> int:
        """Devuelve un bring_id: argmax si el metodo es determinista o greedy=True; si no, muestrea."""
        p = self.probs(team_self, team_opp)
        if greedy or self.deterministic:
            return int(np.argmax(p))
        return int(rng.choice(90, p=p))


def one_hot(k: int) -> np.ndarray:
    """Distribucion que pone toda la masa en el bring k (para metodos deterministas)."""
    p = np.zeros(90)
    p[k] = 1.0
    return p
