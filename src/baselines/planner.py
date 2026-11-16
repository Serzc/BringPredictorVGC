"""Baseline Planificador-v-hat: busqueda exhaustiva sobre los 90 brings con fitness aprendida (A3).

bring = argmax_{b1} sum_{b2} q_Usage(b2) * v-hat(b1, b2)
donde q_Usage es la distribucion del baseline Usage estocastico para el rival.

Por que existe: responde "¿basta con un buen evaluador de matchups y buscar?".
Reemplaza al algoritmo genetico del plan v1: con solo 90 acciones, enumerar es
exacto y mas barato que un GA. v-hat solo se usa aqui, nunca como recompensa del RL.
"""
from __future__ import annotations

import numpy as np

from baselines.usage import UsageBring
from models.base import MetodoBring, one_hot


class PlannerBring(MetodoBring):
    name = "planner"
    deterministic = True

    def __init__(self, vhat, opp_model: UsageBring):
        self.vhat, self.opp = vhat, opp_model

    def expected(self, team_self, team_opp) -> np.ndarray:
        """(90,) win rate esperado de cada bring propio contra el bring rival de Usage."""
        V = self.vhat.v_matrix(team_self, team_opp)                 # (90, 90)
        return V @ self.opp.probs(team_opp, team_self)

    def probs(self, team_self, team_opp):
        return one_hot(int(np.argmax(self.expected(team_self, team_opp))))
