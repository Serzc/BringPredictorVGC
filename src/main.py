"""Punto de entrada unico de BringPredictorVGC (Etapa 2). BORRADOR asistido por IA.

Comandos (desde la raiz del repo):
  python src/main.py setup            descarga el log set, prepara vgc-bench/Showdown (una vez)
  python src/main.py all [--smoke]    data -> bc -> vhat -> val -> rl -> tablas   (arranca y apaga Showdown)
  python src/main.py test [--force]   corrida FINAL en test (una sola vez; --force para repetir, queda registrado)
Pasos sueltos (los mismos que corre 'all'):
  data | bc | vhat | val | rl [--beta-H x] | tablas
Pruebas unitarias: python -m pytest   (configurado en pyproject.toml)
--smoke usa presupuestos minimos (src/config/smoke.yaml): sirve para ver que todo corre, NO es un resultado.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # src/ en el path (imports absolutos)

from config.config import OUTPUT_DIR, load_config  # noqa: E402

TEST_DONE = OUTPUT_DIR / ".test_run_done"      # marca de que ya se corrio test (no se repite sin --force)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def ultimo_checkpoint(tipo: str, seed: int, archivo: str, smoke: bool) -> Path:
    """Checkpoint de la corrida mas reciente de ese tipo y semilla (p. ej. bc, semilla 0)."""
    prefijo = ("smoke_" if smoke else "") + tipo
    candidatos = sorted(d / archivo for d in OUTPUT_DIR.glob(f"{prefijo}_*_s{seed}") if (d / archivo).exists())
    if not candidatos:
        raise SystemExit(f"No hay checkpoint {archivo} de '{prefijo}' semilla {seed}: correr ese paso antes")
    return candidatos[-1]


def guardar_tabla(pares: str, smoke: bool, referencia: str = "usage_argmax") -> None:
    from training.eval.tablas import tabla
    from utils.estadistica import tabla_markdown
    df = tabla(pares, referencia, smoke=smoke)
    nombre = f"{'smoke_' if smoke else ''}tabla_{pares}_vs_{referencia}.md"
    aviso = "> SMOKE TEST: presupuestos minimos, no es un resultado.\n\n" if smoke else ""
    (OUTPUT_DIR / nombre).write_text(aviso + tabla_markdown(df) + "\n", encoding="utf-8")
    print(tabla_markdown(df), f"\n-> src/output/{nombre}")


def vigilante(segundos: float) -> None:
    """Tope duro de tiempo del smoke test: si se pasa, apaga Showdown y termina el proceso."""
    def abortar():
        from sim import showdown
        print(f"[main] smoke test excedio {segundos:.0f} s: se aborta", flush=True)
        if showdown.PROCESO_ACTUAL is not None:
            showdown.matar(showdown.PROCESO_ACTUAL)
        os._exit(3)
    t = threading.Timer(segundos, abortar)
    t.daemon = True
    t.start()


# ---------------------------------------------------------------------------
# Pasos offline (no necesitan Showdown)
# ---------------------------------------------------------------------------
def paso_data(smoke: bool) -> None:
    from dataPrep.construir import construir
    construir(load_config("data"), smoke)


def paso_bc(smoke: bool) -> None:
    from training.entrenar_bc import entrenar
    cfg = load_config("bc", smoke)
    for s in cfg["seeds"]:
        ck, mejor = entrenar(cfg, s, smoke)
        print(f"[bc] semilla {s}: mejor epoca en val {mejor['epoch']} -> {ck}")


def paso_vhat(smoke: bool) -> None:
    from training.entrenar_vhat import entrenar
    cfg = load_config("vhat", smoke)
    for s in cfg["seeds"]:
        ck, mejor = entrenar(cfg, s, smoke)
        print(f"[vhat] semilla {s}: mejor epoca en val {mejor['epoch']} -> {ck}")


# ---------------------------------------------------------------------------
# Pasos online (batallas en Showdown)
# ---------------------------------------------------------------------------
async def online_val(arena, cod, smoke: bool) -> None:
    """Sanity checks y baselines en val contra el rival Usage estocastico."""
    from training.eval.evaluar_arena import evaluar
    acfg, bc_seeds = load_config("arena", smoke), load_config("bc", smoke)["seeds"]
    n, mp = acfg["n_per_pair_val"], acfg.get("max_pairs")
    kw = dict(cfg=acfg, max_pairs=mp, smoke=smoke)
    await evaluar(arena, cod, "usage", "val", n, 0, **kw)                                   # sanity: Usage vs Usage
    await evaluar(arena, cod, "usage_argmax", "val", n, 0, mirror=True, **kw)               # sanity: mismo equipo
    for s in acfg["seeds"]:
        await evaluar(arena, cod, "random", "val", n, s, **kw)
    await evaluar(arena, cod, "usage_argmax", "val", n, 0, **kw)
    await evaluar(arena, cod, "tipos", "val", n, 0, **kw)
    vh = ultimo_checkpoint("vhat", 0, "vhat_bilineal_s0.npz", smoke)
    await evaluar(arena, cod, "planner", "val", n, 0, checkpoint=vh, **kw)
    for s in bc_seeds:
        await evaluar(arena, cod, "bc", "val", n, s, checkpoint=ultimo_checkpoint("bc", s, "model.npz", smoke), **kw)


async def online_rl(arena, cod, smoke: bool, beta_H: float | None) -> None:
    """Entrena el agente RL (una corrida por semilla) y lo evalua en val con el protocolo de la RQ."""
    from training.entrenar_rl import entrenar
    from training.eval.evaluar_arena import evaluar
    cfg, acfg = load_config("rl", smoke), load_config("arena", smoke)
    if beta_H is not None:
        cfg["beta_H"] = beta_H
    for s in cfg["seeds"]:
        ck, _ = await entrenar(arena, cfg, acfg, s, ultimo_checkpoint("bc", s, "model.npz", smoke), smoke)
        await evaluar(arena, cod, "agent", "val", acfg["n_per_pair_val"], s, checkpoint=ck, cfg=acfg,
                      max_pairs=acfg.get("max_pairs"), smoke=smoke)


def con_showdown(corrutina_factory, smoke: bool, fecha_limite: float | None = None) -> None:
    """Arranca Showdown, crea la arena, corre la corrutina y apaga Showdown (siempre)."""
    from dataPrep.cargar import load_codificador
    from sim.arena import Arena
    from sim.showdown import servidor_showdown
    acfg = load_config("arena", smoke)
    cod = load_codificador()
    with servidor_showdown(acfg["port"], acfg["startup_timeout_s"]):
        async def correr():
            arena = Arena.desde_config(acfg, fecha_limite)
            await corrutina_factory(arena, cod)
        asyncio.run(correr())


# ---------------------------------------------------------------------------
# Corrida final en test (una sola vez)
# ---------------------------------------------------------------------------
def paso_test(force: bool) -> None:
    if TEST_DONE.exists() and not force:
        raise SystemExit(f"Test ya se corrio ({TEST_DONE.read_text(encoding='utf-8').strip()}). "
                         "Repetirlo solo con --force; queda registrado y hay que reportarlo.")
    # la marca se escribe ANTES de empezar: si se corta a la mitad, test igual ya se miro
    previo = TEST_DONE.read_text(encoding="utf-8") if TEST_DONE.exists() else ""
    TEST_DONE.parent.mkdir(parents=True, exist_ok=True)
    TEST_DONE.write_text(previo + f"inicio {datetime.now().isoformat(timespec='seconds')}"
                         f"{' (repetida con --force)' if previo else ''}\n", encoding="utf-8")
    from training.eval.diagnosticos import diagnosticos
    from training.eval.evaluar_arena import evaluar
    acfg, bc_seeds = load_config("arena"), load_config("bc")["seeds"]
    rl_seeds, n = load_config("rl")["seeds"], acfg["n_per_pair_test"]
    kw = dict(cfg=acfg, allow_test=True)

    async def correr_test(arena, cod):
        for s in rl_seeds:
            await evaluar(arena, cod, "agent", "test", n, s, checkpoint=ultimo_checkpoint("rl", s, "ckpt_best_val.npz", False), **kw)
        for s in bc_seeds:
            await evaluar(arena, cod, "bc", "test", n, s, checkpoint=ultimo_checkpoint("bc", s, "model.npz", False), **kw)
        for s in acfg["seeds"]:
            await evaluar(arena, cod, "random", "test", n, s, **kw)
        for m in ("usage_argmax", "tipos"):
            await evaluar(arena, cod, m, "test", n, 0, **kw)
        await evaluar(arena, cod, "planner", "test", n, 0, checkpoint=ultimo_checkpoint("vhat", 0, "vhat_bilineal_s0.npz", False), **kw)

    con_showdown(correr_test, smoke=False)
    guardar_tabla("test", False, "usage_argmax")         # Tabla 1
    guardar_tabla("test", False, "bc")                   # agente vs BC (¿aporto el RL?)
    diag = diagnosticos([ultimo_checkpoint("bc", s, "model.npz", False) for s in bc_seeds], "test", allow_test=True)
    (OUTPUT_DIR / "diagnosticos_test.json").write_text(json.dumps(diag, indent=1), encoding="utf-8")
    with open(TEST_DONE, "a", encoding="utf-8") as f:
        f.write(f"fin {datetime.now().isoformat(timespec='seconds')}\n")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="BringPredictorVGC, Etapa 2")
    ap.add_argument("comando", choices=["setup", "all", "test", "data", "bc", "vhat", "val", "rl", "tablas"])
    ap.add_argument("--smoke", action="store_true", help="presupuestos minimos (no es un resultado)")
    ap.add_argument("--force", action="store_true", help="solo para 'test': repetir la corrida final")
    ap.add_argument("--beta-H", type=float, default=None, help="solo para 'rl': peso de la entropia")
    a = ap.parse_args()
    t0 = time.perf_counter()
    fecha_limite = None
    if a.smoke:
        total = float(load_config("smoke").get("total_timeout_s", 1800))
        fecha_limite = time.perf_counter() + total
        vigilante(total + 120)          # tope duro, por si algo se cuelga fuera de la arena

    if a.comando == "setup":
        from scripts.setup import main as setup_main
        setup_main()
    elif a.comando == "test":
        paso_test(a.force)
    elif a.comando == "data":
        paso_data(a.smoke)
    elif a.comando == "bc":
        paso_bc(a.smoke)
    elif a.comando == "vhat":
        paso_vhat(a.smoke)
    elif a.comando == "val":
        con_showdown(lambda ar, cod: online_val(ar, cod, a.smoke), a.smoke, fecha_limite)
        guardar_tabla("val", a.smoke)
    elif a.comando == "rl":
        con_showdown(lambda ar, cod: online_rl(ar, cod, a.smoke, a.beta_H), a.smoke, fecha_limite)
        guardar_tabla("val", a.smoke)
    elif a.comando == "tablas":
        guardar_tabla("val", a.smoke)
    elif a.comando == "all":
        paso_data(a.smoke)
        paso_bc(a.smoke)
        paso_vhat(a.smoke)

        async def online(ar, cod):
            await online_val(ar, cod, a.smoke)
            await online_rl(ar, cod, a.smoke, a.beta_H)
        con_showdown(online, a.smoke, fecha_limite)
        guardar_tabla("val", a.smoke)
    print(f"[main] '{a.comando}' terminado en {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
