"""Ejemplos supervisados a partir de los registros (BC y v-hat). BORRADOR asistido por IA."""
from __future__ import annotations

import numpy as np
import torch

from ..brings import consistent_mask
from ..data.usage import battle_contrib
from ..features.encode import encode_state, encode_team


def bc_examples(records: list[dict], vocab, usage, loo: bool) -> dict[str, np.ndarray]:
    """Un ejemplo por player-bring (los dos lados). loo=True solo para train."""
    cols = {k: [] for k in ("tok_self", "feat_self", "tok_opp", "feat_opp", "cmask", "csize", "lead")}
    for r in records:
        ex = battle_contrib(r) if loo else None
        for side in (0, 1):
            s = encode_state(r["teams"][side], r["teams"][1 - side], vocab, usage, ex)
            o = r["obs"][side]
            m = consistent_mask(o["leads"], o["revealed"])
            for k, v in s.items():
                cols[k].append(v)
            cols["cmask"].append(m)
            cols["csize"].append(int(m.sum()))
            cols["lead"].append(o["leads"])
    return {k: np.stack(v) for k, v in cols.items()}


def vhat_examples(records: list[dict], vocab, usage) -> dict[str, np.ndarray]:
    cols = {k: [] for k in ("tok1", "tok2", "m1", "m2", "y")}
    for r in records:
        if r["winner"] is None:
            continue
        cols["tok1"].append(encode_team(r["teams"][0], vocab, usage)[0])
        cols["tok2"].append(encode_team(r["teams"][1], vocab, usage)[0])
        cols["m1"].append(consistent_mask(r["obs"][0]["leads"], r["obs"][0]["revealed"]))
        cols["m2"].append(consistent_mask(r["obs"][1]["leads"], r["obs"][1]["revealed"]))
        cols["y"].append(1.0 if r["winner"] == 0 else 0.0)
    return {k: np.stack(v) for k, v in cols.items()}


def to_torch(d: dict, idx=None) -> dict[str, torch.Tensor]:
    out = {}
    for k, v in d.items():
        v = v if idx is None else v[idx]
        out[k] = torch.as_tensor(v)
    return out


def batches(n: int, bs: int, rng: np.random.Generator | None):
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, bs):
        yield idx[i:i + bs]


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC por Mann-Whitney (con empates promediados)."""
    order = np.argsort(p)
    ranks = np.empty(len(p))
    sp = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sp[j + 1] == sp[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
