"""Bucle de entrenamiento del agente RL contra batallas reales (plan v2 §7.5).

BORRADOR asistido por IA; responsable sugerido: Jose (bucle de entrenamiento).
Reanudacion (--resume) agregada el 13-11-2026, tambien BORRADOR: revisar.
Paso de main.py: python src/main.py rl [--seed N] [--resume]   (Showdown lo arranca main.py)

- Episodio = una batalla real (H=2: lideres y reservas, gamma=1). On-policy, sin replay buffer.
- Estados: pares de train x 2 orientaciones (lados intercambiados); se elige uno al azar.
- Un bloque = `block` batallas concurrentes del mismo estado, cada una con su bring muestreado.
- Features de uso con las tasas completas de train (sin LOO: aqui la etiqueta es la batalla jugada).
- Cada `eval_every` episodios: evaluacion greedy en val contra Usage; se guarda el mejor por val (nunca test).
- En cada evaluacion (y despues de la inicial) se guarda ckpt_resume.npz de forma atomica:
  parametros, estado de Adam (m, v, t), contadores, estados de los RNG (Generator, random, np.random)
  y el mejor val hasta ahi; las filas nuevas de curves.csv / episodes.csv se anexan en ese momento.
Salida: src/output/rl_<fecha>_s<semilla>/{ckpt_best_val.npz, ckpt_latest.npz, ckpt_resume.npz,
        episodes.csv, curves.csv, meta.json}
"""
from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baselines.usage import UsageBring
from config.config import Run, leer_meta, set_seed
from dataPrep.cargar import load_codificador
from dataPrep.parser import parse_packed_team
from dataPrep.particion import load_pairs
from models.bc import PoliticaLineal
from models.rl_agent import AgenteRL
from sim.arena import evaluate_pairs

RESUME = "ckpt_resume.npz"
# columnas fijas: al anexar por partes, todas las filas tienen que tener el mismo orden de columnas
CURVA_COLS = ["episode", "step", "val_win_rate", "val_sd_pairs", "val_battles", "val_wall_s", "loss", "entropy",
              "kl_to_bc", "mean_reward", "grad_norm", "win_rate_batch", "draws", "turns_won", "turns_lost",
              "T_train_eps_per_s"]
EPISODIO_COLS = ["episode", "pair_id", "orientation", "bring", "bring_opp", "reward", "turns"]


# ---------------------------------------------------------------------------
# Utilidades de archivos (CSV por partes y escritura atomica)
# ---------------------------------------------------------------------------
def _anexar_csv(ruta: Path, filas: list[dict], cols: list[str]) -> None:
    """Agrega filas al CSV (crea el encabezado la primera vez)."""
    if filas:
        pd.DataFrame(filas).reindex(columns=cols).to_csv(ruta, mode="a", header=not ruta.exists(), index=False)


def _recortar_csv(ruta: Path, hasta_episodio: int) -> None:
    """Al reanudar: borra las filas posteriores al checkpoint (se van a volver a jugar)."""
    if ruta.exists():
        df = pd.read_csv(ruta)
        df[df["episode"] <= hasta_episodio].to_csv(ruta, index=False)


def _guardar_atomico(escribir, ruta: Path) -> None:
    """Escribe a un .tmp y lo renombra: si se corta a la mitad, el archivo anterior sigue sano."""
    tmp = ruta.with_name(ruta.name + ".tmp")
    with open(tmp, "wb") as f:
        escribir(f)
    os.replace(tmp, ruta)


def config_de_corrida(cfg: dict, bc_ckpt) -> dict:
    """La config que se guarda en meta.json (y que tiene que coincidir para reanudar), ya pasada por JSON."""
    return json.loads(json.dumps({**cfg, "bc_ckpt": str(bc_ckpt)}, default=str))


# ---------------------------------------------------------------------------
# Checkpoint de reanudacion
# ---------------------------------------------------------------------------
def guardar_resume(carpeta: Path, agente: AgenteRL, rng: np.random.Generator, config_run: dict,
                   contadores: dict) -> None:
    """ckpt_resume.npz: todo lo necesario para seguir exactamente donde se quedo (salvo Showdown)."""
    leg = np.random.get_state()          # ('MT19937', claves, pos, has_gauss, cached_gaussian)

    def escribir(f):
        np.savez(f, theta_L=agente.theta_L, theta_R=agente.theta_R,
                 adam_m=agente.opt.m, adam_v=agente.opt.v, adam_t=np.int64(agente.opt.t),
                 pasos=np.int64(agente.pasos),
                 rng_json=json.dumps(rng.bit_generator.state),
                 py_random_json=json.dumps(random.getstate()),
                 np_legacy_keys=leg[1], np_legacy_json=json.dumps([leg[0], int(leg[2]), int(leg[3]), float(leg[4])]),
                 contadores_json=json.dumps(contadores), config_json=json.dumps(config_run, sort_keys=True))
    _guardar_atomico(escribir, carpeta / RESUME)


def cargar_resume(carpeta: Path, agente: AgenteRL, rng: np.random.Generator, config_run: dict) -> dict:
    """Restaura parametros, Adam y RNGs desde ckpt_resume.npz. Devuelve los contadores."""
    d = np.load(carpeta / RESUME)
    if json.loads(str(d["config_json"])) != config_run:
        raise SystemExit(f"{carpeta / RESUME}: la config guardada no coincide con la actual; no se reanuda")
    # --- parametros y optimizador ---
    agente.theta_L, agente.theta_R = d["theta_L"].copy(), d["theta_R"].copy()
    agente.opt.m, agente.opt.v, agente.opt.t = d["adam_m"].copy(), d["adam_v"].copy(), int(d["adam_t"])
    agente.pasos = int(d["pasos"])
    # --- generadores de numeros aleatorios ---
    rng.bit_generator.state = json.loads(str(d["rng_json"]))
    v, estado, gauss = json.loads(str(d["py_random_json"]))
    random.setstate((v, tuple(estado), gauss))
    nombre, pos, has_gauss, cached = json.loads(str(d["np_legacy_json"]))
    np.random.set_state((nombre, d["np_legacy_keys"].copy(), pos, has_gauss, cached))
    return json.loads(str(d["contadores_json"]))


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
async def entrenar(arena, cfg: dict, arena_cfg: dict, seed: int, bc_ckpt, smoke: bool = False,
                   reanudar: Path | None = None):
    """Entrena una semilla. Con `reanudar` = carpeta de una corrida sin terminar, sigue desde su ckpt_resume."""
    config_run = config_de_corrida(cfg, bc_ckpt)
    if reanudar is not None:
        # --- reanudar exige exactamente la misma config (incluido el checkpoint de BC) y semilla ---
        meta = leer_meta(reanudar) or {}
        if meta.get("config") != config_run or meta.get("seed") != seed:
            dif = sorted(k for k in set(config_run) | set(meta.get("config", {}))
                         if config_run.get(k) != meta.get("config", {}).get(k))
            raise SystemExit(f"No se reanuda {reanudar.name}: la config o la semilla no coinciden "
                             f"(claves distintas: {dif or ['seed']}). Correr sin --resume para empezar de cero.")
        run = Run.reabrir(reanudar)
    else:
        run = Run("rl", config_run, seed, smoke=smoke)
    set_seed(seed)
    rng = np.random.default_rng(seed)
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

    # --- fallos/reinicios de la arena: solo los de ESTA corrida (antes se sumaban los de otras semillas) ---
    ev0, to0 = len(arena.eventos), arena.n_timeouts
    eventos_previos = list(run.meta.get("arena_eventos", []))
    timeouts_previos = int(run.meta.get("arena_timeouts_parcial", 0))
    curvas_buf, episodios_buf = [], []            # filas aun no escritas en los CSV

    def checkpoint_periodico(c: dict) -> None:
        """Anexa las filas pendientes, guarda ckpt_resume.npz y el progreso en meta.json."""
        _anexar_csv(run.dir / "curves.csv", curvas_buf, CURVA_COLS)
        _anexar_csv(run.dir / "episodes.csv", episodios_buf, EPISODIO_COLS)
        curvas_buf.clear()
        episodios_buf.clear()
        guardar_resume(run.dir, agente, rng, config_run, c)
        run.save_meta(progreso={"episode": c["episodio"], "updates": agente.pasos, "best_val": c["mejor_val"]},
                      arena_eventos=eventos_previos + arena.eventos[ev0:],
                      arena_timeouts_parcial=timeouts_previos + arena.n_timeouts - to0)

    if reanudar is not None:
        # --- reanudar: restaurar estado y descartar filas posteriores al checkpoint ---
        c = cargar_resume(run.dir, agente, rng, config_run)
        episodio, mejor_val = int(c["episodio"]), float(c["mejor_val"])
        proxima_eval, tiempo_entrenando = int(c["proxima_eval"]), float(c["tiempo_entrenando"])
        _recortar_csv(run.dir / "curves.csv", episodio)
        _recortar_csv(run.dir / "episodes.csv", episodio)
        run.meta["resumes"][-1]["episode"] = episodio
        print(f"[rl] semilla {seed}: se reanuda {run.run_id} desde el episodio {episodio}")
    else:
        episodio, mejor_val = 0, -1.0
        tiempo_entrenando = 0.0                 # segundos en recolectar + actualizar (sin evaluaciones)
        ev = await eval_val()                   # punto de partida = rho_BC
        curvas_buf.append({"episode": 0, "step": 0, **ev})
        print("eval inicial (rho_BC):", ev)
        proxima_eval = cfg["eval_every"]
        checkpoint_periodico(dict(episodio=episodio, proxima_eval=proxima_eval, mejor_val=mejor_val,
                                  tiempo_entrenando=tiempo_entrenando))

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
            episodios_buf.append({"episode": episodio, "pair_id": r["pair_id"], "orientation": r["orientation"],
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
        # --- 3) evaluacion periodica en val, mejor checkpoint y checkpoint de reanudacion ---
        hubo_eval = episodio >= proxima_eval or episodio >= cfg["episodes"]
        if hubo_eval:
            fila.update(await eval_val())
            proxima_eval += cfg["eval_every"]
            if fila["val_win_rate"] > mejor_val:
                mejor_val = fila["val_win_rate"]
                _guardar_atomico(agente.guardar, run.dir / "ckpt_best_val.npz")
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in fila.items()})
        curvas_buf.append(fila)
        if hubo_eval:
            checkpoint_periodico(dict(episodio=episodio, proxima_eval=proxima_eval, mejor_val=mejor_val,
                                      tiempo_entrenando=tiempo_entrenando))

    # --- 4) cierre: ultimas filas, checkpoint final y meta.json marcado como completo ---
    _anexar_csv(run.dir / "curves.csv", curvas_buf, CURVA_COLS)
    _anexar_csv(run.dir / "episodes.csv", episodios_buf, EPISODIO_COLS)
    _guardar_atomico(agente.guardar, run.dir / "ckpt_latest.npz")
    eventos = eventos_previos + arena.eventos[ev0:]
    extra = dict(episodes=episodio, updates=agente.pasos, best_val_win_rate=mejor_val,
                 T_train_eps_per_s=round(episodio / tiempo_entrenando, 3) if tiempo_entrenando else 0,
                 T_arena_battles_per_s=round(arena.throughput, 3),
                 arena_timeouts=timeouts_previos + arena.n_timeouts - to0,
                 arena_reinicios_showdown=sum(e.get("tipo") == "reinicio_showdown" for e in eventos),
                 arena_eventos=eventos, train_wall_s=round(tiempo_entrenando, 1),
                 checkpoint=str(run.dir / "ckpt_best_val.npz"), complete=True)
    run.save_meta(**extra)
    print({k: v for k, v in extra.items() if k != "arena_eventos"})
    return run.dir / "ckpt_best_val.npz", extra
