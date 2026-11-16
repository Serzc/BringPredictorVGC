"""Ejemplos supervisados a partir de los registros limpios (para rho_BC y v-hat).

"""
from __future__ import annotations

import numpy as np

from utils.brings import LEAD_PAIRS, consistent_mask
from utils.usage import battle_contrib


def bc_examples(records: list[dict], cod, loo: bool) -> dict[str, np.ndarray]:
    """Un ejemplo por player-bring (los dos lados de cada batalla).

    loo=True (solo train): las features de uso restan la propia batalla (leave-one-out),
    para que la etiqueta no se filtre por la feature.
    """
    phi, cmask, lead_pair = [], [], []
    for r in records:
        excluir = battle_contrib(r) if loo else None
        for side in (0, 1):
            o = r["obs"][side]
            phi.append(cod.mon_features(r["teams"][side], r["teams"][1 - side], excluir))
            cmask.append(consistent_mask(o["leads"], o["revealed"]))       # C(o): brings compatibles
            lead_pair.append(LEAD_PAIRS.index(tuple(sorted(o["leads"]))))  # par lider observado (0..14)
    cmask = np.stack(cmask)
    return {"phi": np.stack(phi), "cmask": cmask, "csize": cmask.sum(1), "lead_pair": np.array(lead_pair)}


def vhat_examples(records: list[dict], vhat) -> dict[str, np.ndarray]:
    """Una fila por batalla con ganador: tokens de ambos equipos, C(o) de ambos lados y y = 1 si gano el lado 1."""
    cols = {k: [] for k in ("tok1", "tok2", "m1", "m2", "y")}
    for r in records:
        if r["winner"] is None:
            continue
        t1, t2 = r["teams"]
        cols["tok1"].append(vhat.tokens(t1))                                   # (6, 7): especie, objeto, habilidad, 4 movimientos
        cols["tok2"].append(vhat.tokens(t2))
        cols["m1"].append(consistent_mask(r["obs"][0]["leads"], r["obs"][0]["revealed"]))
        cols["m2"].append(consistent_mask(r["obs"][1]["leads"], r["obs"][1]["revealed"]))
        cols["y"].append(1.0 if r["winner"] == 0 else 0.0)
    return {k: np.stack(v) for k, v in cols.items()}


def lotes(n: int, tam: int, rng: np.random.Generator | None):
    """Indices de minilotes; barajados si se da rng."""
    idx = rng.permutation(n) if rng is not None else np.arange(n)
    for i in range(0, n, tam):
        yield idx[i:i + tam]


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """AUC por Mann-Whitney (empates promediados)."""
    orden = np.argsort(p)
    rangos = np.empty(len(p))
    sp = p[orden]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sp[j + 1] == sp[i]:
            j += 1
        rangos[orden[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    pos = y == 1
    n1, n0 = pos.sum(), (~pos).sum()
    return float((rangos[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
