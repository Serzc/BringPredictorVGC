"""Evalua un metodo de bring contra un rival en los pares congelados (plan v2 §8, §9).

Salida: src/output/arena_<metodo>_vs_<rival>_<pares>_<fecha>_s<semilla>/{battles.csv, per_pair.csv, meta.json}
Los pares de test exigen allow_test=True (solo main.py test).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from baselines.planner import PlannerBring
from baselines.random import RandomBring
from baselines.tipos import TiposBring
from baselines.usage import UsageBring
from config.config import Run, set_seed
from dataPrep.particion import load_pairs
from models.bc import PoliticaLineal
from models.vhat import VHat
from sim.arena import Arena, evaluate_pairs
from utils.estadistica import t_interval, wilson


def construir_metodo(nombre: str, cod, checkpoint=None):
    """Crea el MetodoBring pedido. bc / agent / planner necesitan su checkpoint .npz."""
    if nombre == "random":
        return RandomBring()
    if nombre == "usage":
        return UsageBring(cod.usage)
    if nombre == "usage_argmax":
        return UsageBring(cod.usage, argmax=True)
    if nombre == "tipos":
        return TiposBring()
    if checkpoint is None:
        raise SystemExit(f"falta el checkpoint para '{nombre}' (entrenar primero)")
    if nombre in ("bc", "agent"):
        return PoliticaLineal.cargar(checkpoint, cod, name=nombre)
    if nombre == "planner":
        return PlannerBring(VHat.cargar(checkpoint, cod), UsageBring(cod.usage))
    raise SystemExit(f"metodo desconocido: {nombre}")


def tabla_por_par(filas: list[dict]) -> pd.DataFrame:
    """Win rate por par con IC de Wilson al 95 %, empates y brings distintos jugados."""
    df = pd.DataFrame(filas)
    out = []
    for pid, g in df.groupby("pair_id"):
        p, lo, hi = wilson(int(g["won"].sum()), len(g))
        out.append({"pair_id": pid, "pair_battle": g["pair_battle"].iloc[0], "n": len(g), "wins": int(g["won"].sum()),
                    "draws": int((g["reward"] == 0).sum()), "win_rate": round(p, 4), "wilson_low": round(lo, 4),
                    "wilson_high": round(hi, 4), "distinct_brings_method": int(g["bring_m"].nunique()),
                    "median_turns": int(g["turns"].median())})
    return pd.DataFrame(out)


async def evaluar(arena: Arena, cod, metodo: str, pares: str, n_per_pair: int, seed: int, cfg: dict,
                  checkpoint=None, rival: str = "usage", mirror: bool = False, max_pairs: int | None = None,
                  allow_test: bool = False, smoke: bool = False) -> dict:
    set_seed(seed)
    rng = np.random.default_rng(seed)
    pairs = load_pairs(pares, allow_test=allow_test)[:max_pairs]
    if mirror:   # sanity check: mismo equipo (y mismo metodo) en los dos lados
        pairs = [{**p, "packed_B": p["packed_A"], "key_B": p["key_A"]} for p in pairs]
    m = construir_metodo(metodo, cod, checkpoint)
    o = construir_metodo(metodo if mirror else rival, cod, checkpoint if mirror else None)
    tag = f"arena_{metodo}_vs_{o.name}{'_mirror' if mirror else ''}_{pares}"
    run = Run(tag, {**cfg, "method": metodo, "opponent": o.name, "pairs": pares, "n_per_pair": n_per_pair,
                    "checkpoint": str(checkpoint) if checkpoint else None, "mirror": mirror}, seed, smoke=smoke)
    ev0, to0 = len(arena.eventos), arena.n_timeouts          # fallos/reinicios de ESTA evaluacion
    filas = await evaluate_pairs(arena, m, o, pairs, n_per_pair, rng, greedy=True, block=cfg["block"])
    for r in filas:
        r["seed"] = seed
    eventos = arena.eventos[ev0:]
    timeouts = arena.n_timeouts - to0
    if not filas:   # sin batallas: no se marca completa (main.py la vuelve a intentar)
        run.save_meta(summary={"method": metodo, "battles": 0, "timeouts": timeouts}, arena_eventos=eventos)
        return {"method": metodo, "battles": 0}
    pd.DataFrame(filas).to_csv(run.dir / "battles.csv", index=False)
    tabla = tabla_por_par(filas)
    tabla.to_csv(run.dir / "per_pair.csv", index=False)
    media, lo, hi, sd = t_interval(tabla["win_rate"].to_numpy())
    resumen = {"method": metodo, "label": metodo + ("_mirror" if mirror else ""), "opponent": o.name,
               "pairs": pares, "n_pairs": len(tabla), "battles": len(filas),
               "mean_win_rate_over_pairs": media, "t95_low": lo, "t95_high": hi, "sd_over_pairs": sd,
               "draws": int(sum(r["reward"] == 0 for r in filas)), "capped": int(sum(r["capped"] for r in filas)),
               "timeouts": timeouts,
               "reinicios_showdown": sum(e.get("tipo") == "reinicio_showdown" for e in eventos),
               "T_battles_per_s": round(arena.throughput, 3)}
    run.save_meta(summary=resumen, arena_eventos=eventos, complete=True)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in resumen.items()})
    return resumen
