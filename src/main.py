"""Punto de entrada unico de BringPredictorVGC (Etapa 2).
Comandos (desde la raiz del repo):
  python src/main.py setup            descarga el log set, prepara vgc-bench/Showdown (una vez)
  python src/main.py all [--smoke]    data -> bc -> vhat -> val -> rl -> tablas   (arranca y apaga Showdown)
  python src/main.py test [--force]   corrida FINAL en test (una sola vez; --force para repetir, queda registrado)
Pasos sueltos (los mismos que corre 'all'):
  data | bc | vhat | val [--seed N] | rl [--seed N] [--resume] [--beta-H x] | tablas
Trabajo ya hecho (13-11-2026): cada paso/semilla con una corrida completa y la misma config se SALTA
  (meta.json con complete: true o train_wall_s; ver config.corrida_completa). --force lo repite.
  --resume (rl, all): sigue la corrida RL sin terminar mas reciente de cada semilla desde ckpt_resume.npz.
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

from config.config import OUTPUT_DIR, corrida_completa, leer_meta, load_config  # noqa: E402

TEST_DONE = OUTPUT_DIR / ".test_run_done"      # marca de que ya se corrio test (no se repite sin --force)


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def _como_json(x):
    """Normaliza una config (tuplas, Path, float/int) igual que quedo en meta.json."""
    return json.loads(json.dumps(x, default=str))


def ultimo_checkpoint(tipo: str, seed: int, archivo: str, smoke: bool) -> Path:
    """Checkpoint de la corrida COMPLETA mas reciente de ese tipo y semilla (p. ej. bc, semilla 0).

    Las corridas sin terminar (p. ej. una RL cortada que solo tiene ckpt_best_val.npz) se ignoran.
    """
    prefijo = ("smoke_" if smoke else "") + tipo
    candidatos = sorted(d / archivo for d in OUTPUT_DIR.glob(f"{prefijo}_*_s{seed}")
                        if (d / archivo).exists() and corrida_completa(d))
    if not candidatos:
        raise SystemExit(f"No hay checkpoint {archivo} de una corrida completa de '{prefijo}' semilla {seed}: "
                         "correr ese paso antes")
    return candidatos[-1]


def corrida_hecha(tipo: str, seed: int | None, smoke: bool, config: dict | None = None) -> Path | None:
    """Carpeta de la corrida completa mas reciente de ese tipo/semilla con esa config (o None)."""
    prefijo = ("smoke_" if smoke else "") + tipo
    patron = f"{prefijo}_*" + (f"_s{seed}" if seed is not None else "")
    for d in sorted(OUTPUT_DIR.glob(patron), reverse=True):
        meta = leer_meta(d)
        if meta is None or meta.get("kind") != tipo or not corrida_completa(d):
            continue
        if config is None or meta.get("config") == _como_json(config):
            return d
    return None


def evaluacion_hecha(metodo: str, pares: str, seed: int, smoke: bool, n_per_pair: int,
                     checkpoint=None, mirror: bool = False, rival: str = "usage") -> Path | None:
    """Evaluacion completa con el mismo metodo, pares, semilla, n por par y checkpoint (o None)."""
    rival_real = metodo if mirror else rival
    tipo = f"arena_{metodo}_vs_{rival_real}{'_mirror' if mirror else ''}_{pares}"
    prefijo = ("smoke_" if smoke else "") + tipo
    for d in sorted(OUTPUT_DIR.glob(f"{prefijo}_*_s{seed}"), reverse=True):
        meta = leer_meta(d)
        if meta is None or meta.get("kind") != tipo or not corrida_completa(d):
            continue
        c = meta.get("config", {})
        if (c.get("n_per_pair") == n_per_pair and c.get("checkpoint") == (str(checkpoint) if checkpoint else None)
                and bool(c.get("mirror")) == mirror):
            return d
    return None


def rl_reanudable(seed: int, smoke: bool, config_run: dict) -> Path | None:
    """Corrida RL SIN terminar mas reciente de esa semilla, con ckpt_resume.npz y la MISMA config.

    None si no hay ninguna. Si hay corridas reanudables pero todas con otra config, se niega
    (reanudar con otra config mezclaria dos experimentos en una misma curva).
    """
    prefijo = ("smoke_" if smoke else "") + "rl"
    distintas = []
    for d in sorted(OUTPUT_DIR.glob(f"{prefijo}_*_s{seed}"), reverse=True):
        if not (d / "ckpt_resume.npz").exists() or corrida_completa(d):
            continue
        if (leer_meta(d) or {}).get("config") == _como_json(config_run):
            return d
        distintas.append(d.name)
    if distintas:
        raise SystemExit(f"--resume: hay corridas RL sin terminar de la semilla {seed} ({', '.join(distintas)}) "
                         "pero con otra config (beta_H, BC, presupuesto...). No se reanuda; "
                         "correr sin --resume para empezar de cero con la config actual.")
    return None


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
def paso_data(smoke: bool, force: bool = False) -> None:
    from config.config import CLEAN_DIR
    from dataPrep.construir import construir
    cfg = load_config("data")
    hecha = corrida_hecha("dataset", None, smoke, cfg)
    if hecha and (CLEAN_DIR / "records.pkl").exists() and not force:
        print(f"[data] ya hecho ({hecha.name}); se salta (--force para repetir)")
        return
    construir(cfg, smoke)


def paso_bc(smoke: bool, force: bool = False) -> None:
    from training.entrenar_bc import entrenar
    cfg = load_config("bc", smoke)
    for s in cfg["seeds"]:
        hecha = corrida_hecha("bc", s, smoke, cfg)
        if hecha and not force:
            print(f"[bc] semilla {s}: ya hecha ({hecha.name}); se salta (--force para repetir)")
            continue
        ck, mejor = entrenar(cfg, s, smoke)
        print(f"[bc] semilla {s}: mejor epoca en val {mejor['epoch']} -> {ck}")


def paso_vhat(smoke: bool, force: bool = False) -> None:
    from training.entrenar_vhat import entrenar
    cfg = load_config("vhat", smoke)
    for s in cfg["seeds"]:
        hecha = corrida_hecha("vhat", s, smoke, cfg)
        if hecha and not force:
            print(f"[vhat] semilla {s}: ya hecha ({hecha.name}); se salta (--force para repetir)")
            continue
        ck, mejor = entrenar(cfg, s, smoke)
        print(f"[vhat] semilla {s}: mejor epoca en val {mejor['epoch']} -> {ck}")


# ---------------------------------------------------------------------------
# Pasos online (batallas en Showdown)
# ---------------------------------------------------------------------------
async def online_val(arena, cod, smoke: bool, force: bool = False, solo_seed: int | None = None) -> None:
    """Sanity checks y baselines en val contra el rival Usage estocastico.

    solo_seed=N (--seed N): solo las evaluaciones de esa semilla (Random sN, BC sN; las de
    semilla unica — Usage, espejo, Usage-argmax, Tipos, Planificador — corren con N=0).
    """
    from training.eval.evaluar_arena import evaluar
    acfg, bc_seeds = load_config("arena", smoke), load_config("bc", smoke)["seeds"]
    n, mp = acfg["n_per_pair_val"], acfg.get("max_pairs")
    kw = dict(cfg=acfg, max_pairs=mp, smoke=smoke)

    async def una(metodo, seed, checkpoint=None, mirror=False):
        # --- filtro de --seed y salto de evaluaciones ya completas ---
        if solo_seed is not None and seed != solo_seed:
            return
        hecha = evaluacion_hecha(metodo, "val", seed, smoke, n, checkpoint, mirror)
        if hecha and not force:
            print(f"[val] {metodo}{' espejo' if mirror else ''} s{seed}: ya hecha ({hecha.name}); se salta")
            return
        await evaluar(arena, cod, metodo, "val", n, seed, checkpoint=checkpoint, mirror=mirror, **kw)

    await una("usage", 0)                                   # sanity: Usage vs Usage
    await una("usage_argmax", 0, mirror=True)               # sanity: mismo equipo
    for s in acfg["seeds"]:
        await una("random", s)
    await una("usage_argmax", 0)
    await una("tipos", 0)
    if solo_seed in (None, 0):
        await una("planner", 0, checkpoint=ultimo_checkpoint("vhat", 0, "vhat_bilineal_s0.npz", smoke))
    for s in bc_seeds:
        if solo_seed in (None, s):
            await una("bc", s, checkpoint=ultimo_checkpoint("bc", s, "model.npz", smoke))


async def online_rl(arena, cod, smoke: bool, beta_H: float | None, force: bool = False,
                    solo_seed: int | None = None, resume: bool = False) -> None:
    """Entrena el agente RL (una corrida por semilla) y lo evalua en val con el protocolo de la RQ.

    Por semilla: si ya hay una corrida completa con la misma config, no se reentrena (salvo --force);
    con --resume se sigue la corrida sin terminar de esa semilla; si no hay, se empieza de cero.
    """
    from training.entrenar_rl import config_de_corrida, entrenar
    from training.eval.evaluar_arena import evaluar
    cfg, acfg = load_config("rl", smoke), load_config("arena", smoke)
    if beta_H is not None:
        cfg["beta_H"] = beta_H
    seeds = cfg["seeds"] if solo_seed is None else [solo_seed]
    for s in seeds:
        bc_ck = ultimo_checkpoint("bc", s, "model.npz", smoke)
        config_run = config_de_corrida(cfg, bc_ck)
        hecha = corrida_hecha("rl", s, smoke, config_run)
        if hecha and not force:
            print(f"[rl] semilla {s}: ya entrenada ({hecha.name}); se salta (--force para repetir)")
            ck = hecha / "ckpt_best_val.npz"
        else:
            reanudar = rl_reanudable(s, smoke, config_run) if resume else None
            if resume and reanudar is None:
                print(f"[rl] semilla {s}: --resume pero no hay corrida reanudable con esta config; empieza de cero")
            ck, _ = await entrenar(arena, cfg, acfg, s, bc_ck, smoke, reanudar=reanudar)
        if beta_H is not None and beta_H != load_config("rl", smoke)["beta_H"]:
            # barrido de exploracion: se compara con curves.csv; no se evalua aparte para no
            # reemplazar la fila del agente en la tabla de val
            print(f"[rl] semilla {s}: barrido beta_H={beta_H}, sin evaluacion final en val")
            continue
        n = acfg["n_per_pair_val"]
        hecha_ev = evaluacion_hecha("agent", "val", s, smoke, n, ck)
        if hecha_ev and not force:
            print(f"[val] agent s{s}: ya evaluado ({hecha_ev.name}); se salta")
            continue
        await evaluar(arena, cod, "agent", "val", n, s, checkpoint=ck, cfg=acfg,
                      max_pairs=acfg.get("max_pairs"), smoke=smoke)


def con_showdown(corrutina_factory, smoke: bool, fecha_limite: float | None = None) -> None:
    """Arranca Showdown, crea la arena (con el vigilante que reinicia Showdown), corre y apaga Showdown."""
    from dataPrep.cargar import load_codificador
    from sim.arena import Arena
    from sim.showdown import reiniciar_showdown, servidor_showdown
    acfg = load_config("arena", smoke)
    cod = load_codificador()
    with servidor_showdown(acfg["port"], acfg["startup_timeout_s"]):
        async def correr():
            arena = Arena.desde_config(acfg, fecha_limite,
                                       reiniciar=lambda: reiniciar_showdown(acfg["port"], acfg["startup_timeout_s"]))
            await corrutina_factory(arena, cod)
        asyncio.run(correr())


# ---------------------------------------------------------------------------
# Corrida final en test (una sola vez)
# ---------------------------------------------------------------------------
def paso_test(force: bool) -> None:
    if TEST_DONE.exists() and not force:
        raise SystemExit(f"Test ya se corrio ({TEST_DONE.read_text(encoding='utf-8').strip()}). "
                         "Repetirlo solo con --force; queda registrado y hay que reportarlo.")
    # el agente de test es la corrida RL con la config de rl.yaml (no la del barrido de beta_H),
    # se busca antes de marcar test para fallar sin haberlo mirado
    from training.entrenar_rl import config_de_corrida
    rl_cfg = load_config("rl")
    rl_ckpts = {}
    for s in rl_cfg["seeds"]:
        d = corrida_hecha("rl", s, False, config_de_corrida(rl_cfg, ultimo_checkpoint("bc", s, "model.npz", False)))
        if d is None:
            raise SystemExit(f"No hay corrida RL completa con la config de rl.yaml para la semilla {s}")
        rl_ckpts[s] = d / "ckpt_best_val.npz"
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
            await evaluar(arena, cod, "agent", "test", n, s, checkpoint=rl_ckpts[s], **kw)
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
    guardar_tabla("test", False, "tipos")                # agente vs Tipos (¿supera la heuristica de dominio?)
    guardar_tabla("test", False, "random")               # agente vs Random (¿supera el piso?)
    diag = diagnosticos([ultimo_checkpoint("bc", s, "model.npz", False) for s in bc_seeds], "test", allow_test=True)
    (OUTPUT_DIR / "diagnosticos_test.json").write_text(json.dumps(diag, indent=1), encoding="utf-8")
    with open(TEST_DONE, "a", encoding="utf-8") as f:
        f.write(f"fin {datetime.now().isoformat(timespec='seconds')}\n")


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="BringPredictorVGC, Etapa 2")
    ap.add_argument("comando", choices=["setup", "all", "test", "data", "bc", "vhat", "val", "rl", "tablas"])
    ap.add_argument("--smoke", action="store_true", help="presupuestos minimos (no es un resultado)")
    ap.add_argument("--force", action="store_true",
                    help="'test': repetir la corrida final; otros pasos: repetir trabajo ya completo")
    ap.add_argument("--beta-H", type=float, default=None, help="solo para 'rl'/'all': peso de la entropia")
    ap.add_argument("--seed", type=int, default=None, help="solo para 'rl' y 'val': correr una sola semilla")
    ap.add_argument("--resume", action="store_true",
                    help="solo para 'rl'/'all': seguir la corrida RL sin terminar (misma config) desde ckpt_resume.npz")
    a = ap.parse_args()
    # --- combinaciones no permitidas: mejor un error claro que ignorar la opcion ---
    if a.seed is not None and a.comando not in ("rl", "val"):
        ap.error("--seed solo se puede usar con 'rl' o 'val'")
    if a.resume and a.comando not in ("rl", "all"):
        ap.error("--resume solo se puede usar con 'rl' o 'all'")
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
        paso_data(a.smoke, a.force)
    elif a.comando == "bc":
        paso_bc(a.smoke, a.force)
    elif a.comando == "vhat":
        paso_vhat(a.smoke, a.force)
    elif a.comando == "val":
        con_showdown(lambda ar, cod: online_val(ar, cod, a.smoke, a.force, a.seed), a.smoke, fecha_limite)
        guardar_tabla("val", a.smoke)
    elif a.comando == "rl":
        con_showdown(lambda ar, cod: online_rl(ar, cod, a.smoke, a.beta_H, a.force, a.seed, a.resume),
                     a.smoke, fecha_limite)
        guardar_tabla("val", a.smoke)
    elif a.comando == "tablas":
        guardar_tabla("val", a.smoke)
    elif a.comando == "all":
        paso_data(a.smoke, a.force)
        paso_bc(a.smoke, a.force)
        paso_vhat(a.smoke, a.force)

        async def online(ar, cod):
            await online_val(ar, cod, a.smoke, a.force)
            await online_rl(ar, cod, a.smoke, a.beta_H, a.force, resume=a.resume)
        con_showdown(online, a.smoke, fecha_limite)
        guardar_tabla("val", a.smoke)
    print(f"[main] '{a.comando}' terminado en {time.perf_counter() - t0:.0f} s")


if __name__ == "__main__":
    main()
