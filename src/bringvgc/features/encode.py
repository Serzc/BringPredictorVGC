"""Codificacion del estado s = <c_self, c_opp, m> a tensores (plan v2 §4, §7.1).

Por Pokemon: [especie, objeto, habilidad, mov1..mov4] (enteros) + m_x (4 floats).
Nunca entra: numero de revelados, turnos, resultado, rating, nombres, historial Bo3.
"""
from __future__ import annotations

import numpy as np

from ..data.usage import UsageStats
from ..data.vocab import Vocab
from ..parsing.showdown import parse_packed_team

N_TOK = 7
N_FEAT = 4


def encode_team(team: list[dict], vocab: Vocab, usage: UsageStats, exclude=None):
    tok = np.zeros((6, N_TOK), dtype=np.int64)
    feat = np.zeros((6, N_FEAT), dtype=np.float32)
    for i, m in enumerate(team):
        tok[i, 0] = vocab.idx("species", m["species"])
        tok[i, 1] = vocab.idx("item", m["item"])
        tok[i, 2] = vocab.idx("ability", m["ability"])
        for j, mv in enumerate(m["moves"][:4]):
            tok[i, 3 + j] = vocab.idx("move", mv)
        feat[i] = usage.features(m["species"], exclude)
    return tok, feat


def encode_state(team_self, team_opp, vocab, usage, exclude=None) -> dict[str, np.ndarray]:
    ts, fs = encode_team(team_self, vocab, usage, exclude)
    to, fo = encode_team(team_opp, vocab, usage, exclude)
    return {"tok_self": ts, "feat_self": fs, "tok_opp": to, "feat_opp": fo}


def encode_packed_pair(packed_self: str, packed_opp: str, vocab, usage):
    return encode_state(parse_packed_team(packed_self), parse_packed_team(packed_opp), vocab, usage)


def stack(states: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.stack([s[k] for s in states]) for k in states[0]}
