"""Baseline Random: elige uno de los 90 brings con probabilidad uniforme 1/90.

Por que existe: es el piso. Un metodo que no le gana a Random no aprendio nada
util sobre el bring. Se evalua con varias semillas porque muestrea.
"""
from __future__ import annotations

import numpy as np

from models.base import MetodoBring


class RandomBring(MetodoBring):
    name = "random"
    deterministic = False

    def probs(self, team_self, team_opp):
        return np.full(90, 1 / 90)
