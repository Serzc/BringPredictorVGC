"""Entrena rho_BC (politica lineal) con la perdida de etiquetas parciales (7). Plan v2 §7.3.

BORRADOR asistido por IA; responsable sugerido: Sergio.
Paso de main.py: python src/main.py bc
Early stopping: NLL parcial en val con |C|=1 (se reportan tambien |C|=3 y |C|=6).
Salida: src/output/bc_<fecha>_s<semilla>/{model.npz, metrics.csv, meta.json}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import Run, set_seed
from dataPrep.cargar import load_codificador, load_records
from models.bc import Adam, PoliticaLineal, greedy, perdida_parcial
from training.ejemplos import bc_examples, lotes


def evaluar(pol: PoliticaLineal, data: dict) -> dict:
    """Metricas en un conjunto: NLL parcial (total y por |C|), acierto del par lider y bring permisivo."""
    _, nll, _, _ = perdida_parcial(pol.theta_L, pol.theta_R, data["phi"], data["cmask"])
    g = greedy(pol.theta_L, pol.theta_R, data["phi"])
    out = {"nll_all": float(nll.mean()),
           "lead_pair_acc": float(np.mean(g // 6 == data["lead_pair"])),               # piso 1/15
           "full_bring_permissive_acc": float(np.mean(data["cmask"][np.arange(len(g)), g]))}  # piso 1/90
    for c in (1, 3, 6):
        sel = data["csize"] == c
        out[f"nll_C{c}"] = float(nll[sel].mean()) if sel.any() else float("nan")
        out[f"n_C{c}"] = int(sel.sum())
    return out


def entrenar(cfg: dict, seed: int, smoke: bool = False):
    set_seed(seed)
    run = Run("bc", cfg, seed, smoke=smoke)
    cod = load_codificador()
    tr = bc_examples(load_records("train"), cod, loo=True)       # train con leave-one-out
    va = bc_examples(load_records("val"), cod, loo=False)
    rng = np.random.default_rng(seed)
    # pesos iniciales chicos y aleatorios (la semilla cambia el punto de partida y el orden de los lotes)
    pol = PoliticaLineal(cod, rng.normal(0, 0.01, cod.F), rng.normal(0, 0.01, cod.F), name="bc")
    opt = Adam(2 * cod.F, cfg["lr"])
    mejor, mejor_epoca, sin_mejorar, filas = float("inf"), -1, 0, []
    for epoca in range(cfg["max_epochs"]):
        perdidas = []
        for idx in lotes(len(tr["csize"]), cfg["batch_size"], rng):
            perdida, _, gL, gR = perdida_parcial(pol.theta_L, pol.theta_R, tr["phi"][idx], tr["cmask"][idx], cfg["l2"])
            pol.params = opt.paso(pol.params, np.concatenate([gL, gR]))
            perdidas.append(perdida)
        m = evaluar(pol, va)
        fila = {"epoch": epoca, "train_loss": float(np.mean(perdidas)), **{f"val_{k}": v for k, v in m.items()}}
        filas.append(fila)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in fila.items()})
        # early stopping con la metrica de val elegida en la config
        criterio = m[cfg["early_stop_metric"]]
        if criterio < mejor - 1e-4:
            mejor, mejor_epoca, sin_mejorar = criterio, epoca, 0
            pol.guardar(run.dir / "model.npz")
        else:
            sin_mejorar += 1
            if sin_mejorar >= cfg["patience"]:
                break
    pd.DataFrame(filas).to_csv(run.dir / "metrics.csv", index=False)
    run.save_meta(best_epoch=mejor_epoca, best_val=filas[mejor_epoca], n_train=int(len(tr["csize"])),
                  n_val=int(len(va["csize"])), n_features=cod.F, checkpoint=str(run.dir / "model.npz"))
    return run.dir / "model.npz", filas[mejor_epoca]
