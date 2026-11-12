"""Baseline Usage: trae lo que la poblacion suele traer con esas especies.

BORRADOR asistido por IA.
puntaje(b) = sum_{x en L} log P(lider|x) + sum_{x en R} log P(reserva|x) + sum_{x fuera} log P(fuera|x)
con tasas por especie medidas solo en train (cuentas fraccionales sobre C(o)), sin arquetipo (A9).

Por que existe: es el rival de la pregunta de investigacion ("population-usage baseline").
- usage (estocastico): softmax del puntaje. Es el RIVAL en todas las evaluaciones.
- usage_argmax (determinista): el bring mas tipico. Es el baseline de literatura del survey.
"""
from __future__ import annotations

import numpy as np

from models.base import MetodoBring, one_hot
from utils.brings import LEAD_MASK, OUT_MASK, RES_MASK


class UsageBring(MetodoBring):
    def __init__(self, usage, argmax: bool = False, temperature: float = 1.0):
        self.usage, self.temperature = usage, temperature
        self.deterministic = argmax
        self.name = "usage_argmax" if argmax else "usage"

    def scores(self, team_self) -> np.ndarray:
        """Puntaje (90,) de cada bring segun las tasas de uso de cada especie."""
        log_tasas = np.log(np.stack([self.usage.rates(m["species"]) for m in team_self]))  # (6, 3)
        return LEAD_MASK @ log_tasas[:, 0] + RES_MASK @ log_tasas[:, 1] + OUT_MASK @ log_tasas[:, 2]

    def probs(self, team_self, team_opp):
        s = self.scores(team_self) / self.temperature
        if self.deterministic:
            return one_hot(int(np.argmax(s)))
        e = np.exp(s - s.max())
        return e / e.sum()
