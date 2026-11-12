"""Tablas multi-semilla y regla preregistrada del 55 % (plan v2 §9.2, docs/EXPERIMENTS.md).

BORRADOR asistido por IA.
Por metodo: el win rate de cada par se promedia sobre semillas -> media e intervalo t
sobre pares (unidad = par), desviacion estandar entre semillas (de la media sobre pares)
y diferencia pareada contra una referencia. Lee src/output/arena_*/per_pair.csv.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np
import pandas as pd

from config.config import OUTPUT_DIR
from utils.estadistica import paired_difference, t_interval


def decision(win_rates_por_par, umbral: float = 0.55) -> dict:
    """Regla preregistrada: 'supera el 55 %' solo si el limite inferior del intervalo t > 0.55."""
    m, lo, hi, sd = t_interval(win_rates_por_par)
    veredicto = ("supera el 55 %" if lo > umbral else
                 "compatible, no concluyente" if m > umbral else "se rechaza")
    return {"mean": m, "t_low": lo, "t_high": hi, "sd_pairs": sd, "n_pairs": len(win_rates_por_par),
            "verdict": veredicto}


def recolectar(pares: str, rival: str = "usage", smoke: bool = False) -> dict:
    """etiqueta -> semilla -> {pair_id: win_rate}. Usa la corrida mas reciente por (etiqueta, semilla)."""
    prefijo = ("smoke_" if smoke else "") + "arena_"
    ultima = {}
    for d in sorted(OUTPUT_DIR.glob(f"{prefijo}*_vs_{rival}*_{pares}_*")):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if not (d / "per_pair.csv").exists() or "summary" not in meta:
            continue
        ultima[(meta["summary"]["label"], meta["seed"])] = d
    out = defaultdict(dict)
    for (etiqueta, s), d in ultima.items():
        df = pd.read_csv(d / "per_pair.csv")
        out[etiqueta][s] = dict(zip(df["pair_id"].astype(int), df["win_rate"].astype(float)))
    return out


def tabla(pares: str, referencia: str | None = "usage_argmax", rival: str = "usage", smoke: bool = False) -> pd.DataFrame:
    datos = recolectar(pares, rival, smoke)
    ref = None
    if referencia in datos:
        rs = datos[referencia]
        ids = sorted(set.intersection(*[set(v) for v in rs.values()]))
        ref = {i: np.mean([rs[s][i] for s in rs]) for i in ids}
    filas = []
    for metodo, semillas in sorted(datos.items()):
        ids = sorted(set.intersection(*[set(v) for v in semillas.values()]))
        por_par = [np.mean([semillas[s][i] for s in semillas]) for i in ids]     # promedio sobre semillas
        dec = decision(por_par)
        medias_semilla = [np.mean([semillas[s][i] for i in ids]) for s in semillas]
        fila = {"metodo": metodo, "semillas": len(semillas), "pares": len(ids),
                "media": round(dec["mean"], 4), "t95_bajo": round(dec["t_low"], 4), "t95_alto": round(dec["t_high"], 4),
                "sd_semillas": round(float(np.std(medias_semilla, ddof=1)), 4) if len(semillas) > 1 else "n/a (1 semilla)",
                "regla_55": dec["verdict"]}
        if ref is not None and metodo != referencia:
            comunes = [i for i in ids if i in ref]
            d, se = paired_difference([np.mean([semillas[s][i] for s in semillas]) for i in comunes],
                                      [ref[i] for i in comunes])
            fila[f"dif_vs_{referencia}"] = f"{d:+.4f} ± {se:.4f}"
        filas.append(fila)
    return pd.DataFrame(filas).fillna("-")      # celdas vacias (p. ej. la referencia contra si misma)
