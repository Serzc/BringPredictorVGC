"""Diagnosticos del predictor del bring rival sobre logs (plan v2 §9.4).

- Lead-pair exact match (piso 1/15 = 6,67 %), sobre el 100 % de las batallas.
- Full-bring exact match permisivo: acierto si el bring predicho esta en C(o) (piso 1/90).
Se reporta media +- desviacion estandar sobre las semillas de rho_BC.
En test solo con allow_test=True (corrida final, una vez).
"""
from __future__ import annotations

import numpy as np

from dataPrep.cargar import load_codificador, load_records
from models.bc import PoliticaLineal
from training.ejemplos import bc_examples
from training.entrenar_bc import evaluar


def diagnosticos(checkpoints: list, particion: str = "val", allow_test: bool = False) -> dict:
    cod = load_codificador()
    data = bc_examples(load_records(particion, allow_test=allow_test), cod, loo=False)
    res = [evaluar(PoliticaLineal.cargar(c, cod), data) for c in checkpoints]
    out = {"particion": particion, "n_player_brings": int(len(data["csize"])), "semillas": len(res),
           "pisos": {"lead_pair": 1 / 15, "full_bring": 1 / 90}}
    for k in ("lead_pair_acc", "full_bring_permissive_acc", "nll_C1"):
        v = [r[k] for r in res]
        out[k] = {"media": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}
    return out
