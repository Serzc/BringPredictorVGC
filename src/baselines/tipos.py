"""Baseline Tipos: heuristica de dominio basada solo en la tabla de tipos.

puntaje(x) = media_y max_mov efectividad(mov de x, y) - media_y max_t efectividad(tipo de y, x)
Se traen los 4 Pokemon con mejor puntaje y lideran los 2 mejores (empate -> orden de la hoja).

Por que existe: es la heuristica "simple" que pide la rubrica. Si el agente RL no
le gana, lo que aprendio se explica con la tabla de tipos. Ignora habilidades,
objetos, megas y Tera (limitacion declarada).
"""
from __future__ import annotations

import numpy as np

from models.base import MetodoBring, one_hot
from utils.brings import bring_index
from utils.features import matriz_ofensiva, matriz_stab


class TiposBring(MetodoBring):
    name = "tipos"
    deterministic = True

    def mon_scores(self, team_self, team_opp) -> np.ndarray:
        """Puntaje (6,): ventaja ofensiva media menos presion defensiva media."""
        return matriz_ofensiva(team_self, team_opp).mean(1) - matriz_stab(team_self, team_opp).mean(1)

    def probs(self, team_self, team_opp):
        s = self.mon_scores(team_self, team_opp)
        orden = sorted(range(6), key=lambda i: (-s[i], i))
        return one_hot(bring_index(orden[:2], orden[2:4]))
