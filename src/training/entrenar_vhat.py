"""Entrena v-hat bilineal (eqs. 8-10 del survey) con la verosimilitud marginalizada (eq:marginal, 11). Plan v2 §7.4.

BORRADOR asistido por IA; responsable sugerido: Jose (revisar y hacer commit).
Paso de main.py: python src/main.py vhat
Metricas en val (test solo en la corrida final):
- team_*: prediccion a nivel de equipo con q uniforme sobre los 90 brings de cada lado. No usa
  nada de lo observado en la batalla -> sin fuga por revelacion. Es la comparable con Oblander
  y la que se usa para early stopping.
- obs_*: Pr(w|o1,o2) de (11) con los C(o) observados. Es el objetivo de entrenamiento, pero
  |C(o)| se correlaciona con el resultado (los ganadores revelan menos): solo diagnostico.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import Run, set_seed
from dataPrep.cargar import load_codificador, load_records
from models.bc import Adam
from models.vhat import VHat, aplanar, perdida_marginal
from training.ejemplos import auc, lotes, vhat_examples


def _lote(data: dict, idx, q: str) -> dict:
    """Arma un lote; q='obs' usa q uniforme sobre C(o) (eq. 11); q='team' uniforme sobre los 90 brings de cada lado."""
    m1, m2 = data["m1"][idx].astype(float), data["m2"][idx].astype(float)
    if q == "team":
        m1, m2 = np.ones_like(m1), np.ones_like(m2)
    return {"tok1": data["tok1"][idx], "tok2": data["tok2"][idx],
            "q1": m1 / m1.sum(1, keepdims=True), "q2": m2 / m2.sum(1, keepdims=True), "y": data["y"][idx]}


def evaluar(vh: VHat, data: dict, tam: int = 256) -> dict:
    """Metricas en un conjunto: nll, acierto y AUC, con q de equipo (sin fuga) y con q de lo observado (diagnostico)."""
    out = {}
    for q in ("team", "obs"):
        p = np.concatenate([perdida_marginal(vh.p, **_lote(data, idx, q), con_grad=False)[3]
                            for idx in lotes(len(data["y"]), tam, None)])
        y = data["y"]
        out[f"{q}_nll"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        out[f"{q}_acc"] = float(np.mean((p > 0.5) == (y == 1)))
        out[f"{q}_auc"] = auc(y, p)
    return out


def entrenar(cfg: dict, seed: int, smoke: bool = False):
    set_seed(seed)
    run = Run("vhat", cfg, seed, smoke=smoke)
    # Dimensiones del yaml (d = embedding por Pokemon, hidden = capa h, K = dimension de f(b)); inicializacion con la semilla
    vh = VHat(load_codificador(), cfg["d"], cfg["hidden"], cfg["K"], seed=seed)
    tr = vhat_examples(load_records("train"), vh)
    va = vhat_examples(load_records("val"), vh)
    rng = np.random.default_rng(seed)                  # orden de los minilotes
    opt = Adam(aplanar(vh.p).size, cfg["lr"])
    ruta = run.dir / f"vhat_bilineal_s{seed}.npz"
    mejor, mejor_epoca, sin_mejorar, filas = float("inf"), -1, 0, []
    for epoca in range(cfg["max_epochs"]):
        perdidas = [vh.entrenar_paso(opt, _lote(tr, idx, "obs"), cfg["l2"])
                    for idx in lotes(len(tr["y"]), cfg["batch_size"], rng)]
        m = evaluar(vh, va)
        fila = {"epoch": epoca, "train_loss": float(np.mean(perdidas)), **{f"val_{k}": v for k, v in m.items()}}
        filas.append(fila)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in fila.items()})
        if m["team_nll"] < mejor - 1e-4:
            mejor, mejor_epoca, sin_mejorar = m["team_nll"], epoca, 0
            vh.guardar(ruta)
        else:
            sin_mejorar += 1
            if sin_mejorar >= cfg["patience"]:
                break
    pd.DataFrame(filas).to_csv(run.dir / "metrics.csv", index=False)
    run.save_meta(best_epoch=mejor_epoca, best_val=filas[mejor_epoca], n_train=int(len(tr["y"])),
                  n_val=int(len(va["y"])), val_win_rate_side1=float(va["y"].mean()),
                  checkpoint=str(ruta), n_parametros=int(aplanar(vh.p).size),
                  note="1 semilla; v-hat no es el modelo de la RQ (solo Planificador y diagnosticos)")
    return ruta, filas[mejor_epoca]
