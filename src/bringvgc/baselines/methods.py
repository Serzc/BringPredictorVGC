"""Metodos de bring (plan v2 §8). Cada uno da una distribucion sobre los 90 brings.

BORRADOR asistido por IA. Interfaz comun: probs(team_self, team_opp) -> (90,)
y choose(..., rng, greedy). En la arena cada metodo vive dentro de un
MethodPlayer que solo cambia el Team Preview; pi_b es la misma para todos.

| Metodo        | Modo                | Determinista |
| random        | uniforme            | no (semillas de muestreo) |
| usage         | softmax del score   | no (rival de la RQ)       |
| usage_argmax  | argmax del score    | si                        |
| tipos         | regla de tipos      | si                        |
| bc            | rho_BC greedy       | si dado el checkpoint     |
| planner       | argmax E_q[v-hat]   | si dado v-hat             |
| agent         | politica RL greedy  | si dado el checkpoint     |
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
from poke_env.data import GenData

from ..brings import BRINGS, LEAD_MASK, OUT_MASK, RES_MASK, bring_index
from ..features.encode import encode_state, encode_team, stack


class Method:
    name = "base"
    deterministic = False

    def probs(self, team_self, team_opp) -> np.ndarray:
        raise NotImplementedError

    def choose(self, team_self, team_opp, rng: np.random.Generator, greedy: bool = False) -> int:
        p = self.probs(team_self, team_opp)
        return int(np.argmax(p)) if (greedy or self.deterministic) else int(rng.choice(90, p=p))


class RandomMethod(Method):
    name = "random"

    def probs(self, team_self, team_opp):
        return np.full(90, 1 / 90)


class UsageMethod(Method):
    """score(b) = sum_L log P(lider|x) + sum_R log P(reserva|x) + sum_fuera log P(fuera|x).

    Tasas por especie de train (cuentas fraccionales sobre C(o)), sin arquetipo (A9).
    """

    def __init__(self, usage, argmax: bool = False, temperature: float = 1.0):
        self.usage, self.T = usage, temperature
        self.deterministic = argmax
        self.name = "usage_argmax" if argmax else "usage"

    def scores(self, team_self) -> np.ndarray:
        r = np.log(np.stack([self.usage.rates(m["species"]) for m in team_self]))  # (6,3)
        return LEAD_MASK @ r[:, 0] + RES_MASK @ r[:, 1] + OUT_MASK @ r[:, 2]

    def probs(self, team_self, team_opp):
        s = self.scores(team_self) / self.T
        if self.deterministic:
            p = np.zeros(90)
            p[int(np.argmax(s))] = 1
            return p
        e = np.exp(s - s.max())
        return e / e.sum()


@lru_cache(maxsize=1)
def _gen():
    return GenData.from_gen(9)


def _types(mon) -> list[str]:
    d = _gen().pokedex.get(mon["species"]) or {}
    return [t.upper() for t in d.get("types", ["Normal"])]


def _eff(att: str, def_types: list[str]) -> float:
    chart = _gen().type_chart
    e = 1.0
    for t in def_types:
        e *= chart[t][att]
    return e


class TiposMethod(Method):
    """Ventaja de tipos: score(x) = media_y max_mov eff(mov, y) - media_y max_t eff(t_y, x).

    Se traen los 4 mejores y lideran los 2 mejores. Ignora habilidades, megas y Tera.
    """
    name = "tipos"
    deterministic = True

    def mon_scores(self, team_self, team_opp) -> np.ndarray:
        moves = _gen().moves
        out = []
        for x in team_self:
            atk_types = [moves[m]["type"].upper() for m in x["moves"]
                         if m in moves and moves[m].get("category") != "Status"
                         and moves[m].get("basePower", 0) > 0]
            off = np.mean([max([_eff(t, _types(y)) for t in atk_types], default=0.0) for y in team_opp])
            de = np.mean([max(_eff(t, _types(x)) for t in _types(y)) for y in team_opp])
            out.append(off - de)
        return np.array(out)

    def probs(self, team_self, team_opp):
        s = self.mon_scores(team_self, team_opp)
        order = sorted(range(6), key=lambda i: (-s[i], i))  # empate -> orden de la hoja
        p = np.zeros(90)
        p[bring_index(order[:2], order[2:4])] = 1
        return p


class PolicyMethod(Method):
    """rho_BC o el agente RL: greedy (argmax autorregresivo) o muestreo."""

    def __init__(self, policy, vocab, usage, name="bc", greedy=True):
        self.policy, self.vocab, self.usage = policy, vocab, usage
        self.name, self.deterministic = name, greedy

    def state(self, team_self, team_opp):
        return encode_state(team_self, team_opp, self.vocab, self.usage)

    @torch.no_grad()
    def probs(self, team_self, team_opp):
        b = {k: torch.as_tensor(v) for k, v in stack([self.state(team_self, team_opp)]).items()}
        if self.deterministic:
            p = np.zeros(90)
            p[int(self.policy.greedy(b)[0])] = 1
            return p
        return self.policy.log_probs(b)[0].exp().double().numpy()


class PlannerMethod(Method):
    """Planificador-v-hat: argmax_b sum_{b2} q_Usage(b2) v-hat(b, b2), enumeracion exacta (A3)."""
    name = "planner"
    deterministic = True

    def __init__(self, vhat, vocab, usage, opp_model: UsageMethod):
        self.vhat, self.vocab, self.usage, self.opp = vhat, vocab, usage, opp_model

    @torch.no_grad()
    def expected(self, team_self, team_opp) -> np.ndarray:
        t1 = torch.as_tensor(encode_team(team_self, self.vocab, self.usage)[0]).unsqueeze(0)
        t2 = torch.as_tensor(encode_team(team_opp, self.vocab, self.usage)[0]).unsqueeze(0)
        V = self.vhat.v_matrix(t1, t2)[0].double().numpy()      # (90,90)
        return V @ self.opp.probs(team_opp, team_self)

    def probs(self, team_self, team_opp):
        p = np.zeros(90)
        p[int(np.argmax(self.expected(team_self, team_opp)))] = 1
        return p


def build_method(name: str, usage, vocab=None, bc_ckpt=None, vhat_ckpt=None, agent_ckpt=None):
    if name == "random":
        return RandomMethod()
    if name == "usage":
        return UsageMethod(usage)
    if name == "usage_argmax":
        return UsageMethod(usage, argmax=True)
    if name == "tipos":
        return TiposMethod()
    if name in ("bc", "agent"):
        from ..train.bc import load_policy
        ck = bc_ckpt if name == "bc" else agent_ckpt
        if not ck:
            raise SystemExit(f"falta el checkpoint para '{name}'")
        return PolicyMethod(load_policy(ck), vocab, usage, name=name)
    if name == "planner":
        from ..train.vhat import load_vhat
        if not vhat_ckpt:
            raise SystemExit("falta el checkpoint de v-hat")
        return PlannerMethod(load_vhat(vhat_ckpt), vocab, usage, UsageMethod(usage))
    raise SystemExit(f"metodo desconocido: {name}")


__all__ = ["BRINGS", "Method", "RandomMethod", "UsageMethod", "TiposMethod", "PolicyMethod",
           "PlannerMethod", "build_method"]
