"""Bucle de entrenamiento del agente RL contra batallas reales (plan v2 §7.5).

BORRADOR asistido por IA; responsable sugerido: Jose (bucle de entrenamiento).
Paso de main.py: python src/main.py rl   (Showdown lo arranca main.py)

- Episodio = una batalla real (H=2: lideres y reservas, gamma=1). On-policy, sin replay buffer.
- Estados: pares de train x 2 orientaciones (lados intercambiados); se elige uno al azar.
- Un bloque = `block` batallas concurrentes del mismo estado, cada una con su bring muestreado.
- Features de uso con las tasas completas de train (sin LOO: aqui la etiqueta es la batalla jugada).
- Cada `eval_every` episodios: evaluacion greedy en val contra Usage; se guarda el mejor por val (nunca test).
Salida: src/output/rl_<fecha>_s<semilla>/{ckpt_best_val.npz, ckpt_latest.npz, episodes.csv, curves.csv, meta.json}
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from baselines.usage import UsageBring
from config.config import Run, set_seed
from dataPrep.cargar import load_codificador
from dataPrep.parser import parse_packed_team
from dataPrep.particion import load_pairs
from models.bc import PoliticaLineal
from models.rl_agent import AgenteRL
from sim.arena import evaluate_pairs


async def entrenar(arena, cfg: dict, arena_cfg: dict, seed: int, bc_ckpt, smoke: bool = False):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    run = Run("rl", {**cfg, "bc_ckpt": str(bc_ckpt)}, seed, smoke=smoke)
    cod = load_codificador()
    bc = PoliticaLineal.cargar(bc_ckpt, cod)
    agente = AgenteRL(cod, bc, cfg["lr"], cfg["beta_H"], cfg["beta_KL"], cfg["grad_clip"])
    rival = UsageBring(cod.usage)
    val_pairs = load_pairs("val")[: cfg["val_pairs"]]

    # --- estados de entrenamiento precalculados: (par de train, orientacion) ---
    estados = []
    for pid, p in enumerate(load_pairs("train")):
        for orient, (pm, po) in (("A", (p["packed_A"], p["packed_B"])), ("B", (p["packed_B"], p["packed_A"]))):
            tm, to = parse_packed_team(pm), parse_packed_team(po)
            estados.append({"pid": pid, "orient": orient, "pm": pm, "po": po,
                            "phi": cod.mon_features(tm, to), "prob_rival": rival.probs(to, tm)})

    async def eval_val() -> dict:
        """Win rate greedy en val (media sobre pares) contra Usage estocastico."""
        t0 = time.perf_counter()
        filas = await evaluate_pairs(arena, agente, rival, val_pairs, cfg["val_n_per_pair"], rng,
                                     greedy=True, block=arena_cfg["block"])
        df = pd.DataFrame(filas)
        wr = df.groupby("pair_id")["won"].mean() if len(df) else pd.Series(dtype=float)
        return {"val_win_rate": float(wr.mean()), "val_sd_pairs": float(wr.std(ddof=1)) if len(wr) > 1 else float("nan"),
                "val_battles": len(df), "val_wall_s": round(time.perf_counter() - t0, 1)}

    episodio, mejor_val, curvas, episodios_log = 0, -1.0, [], []
    tiempo_entrenando = 0.0                     # segundos en recolectar + actualizar (sin evaluaciones)
    ev = await eval_val()                       # punto de partida = rho_BC
    curvas.append({"episode": 0, "step": 0, **ev})
    print("eval inicial (rho_BC):", ev)
    proxima_eval = cfg["eval_every"]
    while episodio < cfg["episodes"]:
        t0 = time.perf_counter()
        # --- 1) recolectar un lote de episodios con la politica actual (muestreando: explora) ---
        phis, acciones, recompensas, lote = [], [], [], []
        while len(lote) < cfg["batch_size"]:
            s = estados[int(rng.integers(len(estados)))]
            p = agente.probs_desde_estado(s["phi"])
            filas = await arena.play(s["pm"], s["po"], arena_cfg["block"],
                                     lambda: int(rng.choice(90, p=p)),
                                     lambda: int(rng.choice(90, p=s["prob_rival"])))
            for r in filas:
                r.update(pair_id=s["pid"], orientation=s["orient"])
                phis.append(s["phi"])
                acciones.append(r["bring_m"])
                recompensas.append(float(r["reward"]))
            lote += filas
        # --- 2) un paso de gradiente con REINFORCE + entropia + KL ---
        info = agente.actualizar(np.stack(phis), acciones, recompensas)
        dt = time.perf_counter() - t0
        tiempo_entrenando += dt
        for r in lote:
            episodio += 1
            episodios_log.append({"episode": episodio, "pair_id": r["pair_id"], "orientation": r["orientation"],
                                  "bring": r["bring_m"], "bring_opp": r["bring_o"], "reward": r["reward"],
                                  "turns": r["turns"]})
        ganadas = [r["turns"] for r in lote if r["reward"] == 1]
        perdidas = [r["turns"] for r in lote if r["reward"] == -1]
        fila = {"episode": episodio, "step": agente.pasos, **info,
                "win_rate_batch": float(np.mean([r["reward"] == 1 for r in lote])),
                "draws": int(sum(r["reward"] == 0 for r in lote)),
                "turns_won": float(np.mean(ganadas)) if ganadas else float("nan"),
                "turns_lost": float(np.mean(perdidas)) if perdidas else float("nan"),
                "T_train_eps_per_s": round(len(lote) / dt, 3)}
        # --- 3) evaluacion periodica en val y checkpoints ---
        if episodio >= proxima_eval or episodio >= cfg["episodes"]:
            fila.update(await eval_val())
            proxima_eval += cfg["eval_every"]
            if fila["val_win_rate"] > mejor_val:
                mejor_val = fila["val_win_rate"]
                agente.guardar(run.dir / "ckpt_best_val.npz")
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fila.items()})
        curvas.append(fila)
    agente.guardar(run.dir / "ckpt_latest.npz")
    pd.DataFrame(episodios_log).to_csv(run.dir / "episodes.csv", index=False)
    pd.DataFrame(curvas).to_csv(run.dir / "curves.csv", index=False)
    extra = dict(episodes=episodio, updates=agente.pasos, best_val_win_rate=mejor_val,
                 T_train_eps_per_s=round(episodio / tiempo_entrenando, 3) if tiempo_entrenando else 0,
                 T_arena_battles_per_s=round(arena.throughput, 3), arena_timeouts=arena.n_timeouts,
                 train_wall_s=round(tiempo_entrenando, 1), checkpoint=str(run.dir / "ckpt_best_val.npz"))
    run.save_meta(**extra)
    print(extra)
    return run.dir / "ckpt_best_val.npz", extra
