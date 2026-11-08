"""Diagnosticos del predictor del bring rival sobre logs (plan v2 §9.4).

BORRADOR asistido por IA.
- Lead-pair exact match (piso 1/15 = 6,67 %), sobre el 100 % de las batallas.
- Full-bring exact match permisivo: acierto si el bring predicho esta en C(o) (piso 1/90).
Se reporta media +- desviacion estandar sobre las semillas de rho_BC.
En test solo con --allow-test (Fase 5, una sola vez).
Uso: python -m bringvgc.eval.diagnostics --partition val --ckpts results/bc_*_s0/model.pt results/bc_*_s1/model.pt ...
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ..data.dataset import load_artifacts, load_records
from ..train.bc import evaluate, load_policy
from ..train.common import bc_examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="val", choices=["val", "test"])
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--ckpts", nargs="+", required=True)
    a = ap.parse_args()
    vocab, usage = load_artifacts()
    data = bc_examples(load_records(a.partition, allow_test=a.allow_test), vocab, usage, loo=False)
    res = [evaluate(load_policy(c), data) for c in a.ckpts]
    out = {"partition": a.partition, "n_player_brings": int(len(data["csize"])), "seeds": len(res),
           "floors": {"lead_pair": 1 / 15, "full_bring": 1 / 90}}
    for k in ("lead_pair_acc", "full_bring_permissive_acc", "nll_C1"):
        v = [r[k] for r in res]
        out[k] = {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
