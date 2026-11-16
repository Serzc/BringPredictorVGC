"""Reanudacion de RL (--resume) y salto de trabajo ya hecho en main.py (13-11-2026).
- Reanudar da EXACTAMENTE los mismos parametros (y las mismas curvas) que una corrida sin cortes,
  contra una arena falsa determinista. Con Showdown real la garantia es mas debil: los resultados de
  las batallas no se pueden sembrar, asi que una reanudacion sigue la misma distribucion, no la misma
  secuencia (ver src/output/FIX_ARENA_REPORT.md).
- Salto: corridas completas (complete: true o train_wall_s) se saltan; las incompletas se ignoran.
Todo escribe en carpetas temporales: nunca toca src/output/.
"""
import asyncio
import json
import sys

import numpy as np
import pandas as pd
import pytest

import config.config as cc
import main
from dataPrep.cargar import clean_available

needs_data = pytest.mark.skipif(not clean_available(), reason="faltan los datos limpios (python src/main.py data)")


# ---------------------------------------------------------------------------
# Arena falsa determinista: el resultado depende solo de los brings elegidos
# ---------------------------------------------------------------------------
class ArenaFalsa:
    def __init__(self, cortar_tras: int | None = None):
        self.eventos, self.n_timeouts, self.throughput, self.conc = [], 0, 0.0, 4
        self.llamadas, self.cortar_tras = 0, cortar_tras

    async def play(self, pm, po, k, ch_m, ch_o):
        self.llamadas += 1
        if self.cortar_tras is not None and self.llamadas > self.cortar_tras:
            raise RuntimeError("corte simulado (como un cuelgue que mata el proceso)")
        filas = []
        for i in range(k):
            bm, bo = ch_m(), ch_o()
            r = 1 if (bm * 7 + bo * 3 + len(pm)) % 5 < 2 else -1      # regla fija, sin azar
            filas.append({"battle_tag": f"t{self.llamadas}-{i}", "bring_m": bm, "bring_o": bo,
                          "reward": r, "won": int(r == 1), "turns": 3 + bm % 4, "capped": 0})
        return filas


def _entrenar(tmp, monkeypatch, cfg, bc_ck, arena, reanudar=None):
    from training.entrenar_rl import entrenar
    monkeypatch.setattr(cc, "OUTPUT_DIR", tmp)            # Run escribe en la carpeta temporal
    return asyncio.run(entrenar(arena, cfg, {"block": 4}, 0, bc_ck, reanudar=reanudar))


@needs_data
def test_resume_da_los_mismos_parametros_que_sin_corte(tmp_path, monkeypatch):
    from dataPrep.cargar import load_codificador
    from models.bc import PoliticaLineal
    # --- BC de juguete (parametros al azar, fijos) como punto de partida ---
    cod = load_codificador()
    g = np.random.default_rng(123)
    bc_ck = tmp_path / "bc.npz"
    PoliticaLineal(cod, g.normal(0, 0.1, cod.F), g.normal(0, 0.1, cod.F)).guardar(bc_ck)
    cfg = {**cc.load_config("rl"), "episodes": 96, "batch_size": 16, "eval_every": 32,
           "val_pairs": 2, "val_n_per_pair": 4, "seeds": [0]}

    # --- A) corrida sin cortes ---
    (tmp_path / "A").mkdir()
    ck_a, extra_a = _entrenar(tmp_path / "A", monkeypatch, cfg, bc_ck, ArenaFalsa())
    # --- B) misma corrida cortada a la mitad (despues del checkpoint del episodio 32) y reanudada ---
    (tmp_path / "B").mkdir()
    with pytest.raises(RuntimeError, match="corte simulado"):
        _entrenar(tmp_path / "B", monkeypatch, cfg, bc_ck, ArenaFalsa(cortar_tras=22))
    carpeta_b = next((tmp_path / "B").glob("rl_*_s0"))
    assert (carpeta_b / "ckpt_resume.npz").exists() and not cc.corrida_completa(carpeta_b)
    ck_b, extra_b = _entrenar(tmp_path / "B", monkeypatch, cfg, bc_ck, ArenaFalsa(), reanudar=carpeta_b)

    # --- parametros identicos bit a bit, mismas curvas y mismos episodios ---
    for nombre in ("ckpt_latest.npz", "ckpt_best_val.npz"):
        a, b = np.load(ck_a.parent / nombre), np.load(ck_b.parent / nombre)
        assert np.array_equal(a["theta_L"], b["theta_L"]) and np.array_equal(a["theta_R"], b["theta_R"])
    sin_tiempos = lambda df: df.drop(columns=["val_wall_s", "T_train_eps_per_s"])      # noqa: E731
    pd.testing.assert_frame_equal(sin_tiempos(pd.read_csv(ck_a.parent / "curves.csv")),
                                  sin_tiempos(pd.read_csv(ck_b.parent / "curves.csv")))
    pd.testing.assert_frame_equal(pd.read_csv(ck_a.parent / "episodes.csv"), pd.read_csv(ck_b.parent / "episodes.csv"))
    meta_b = json.loads((ck_b.parent / "meta.json").read_text(encoding="utf-8"))
    assert meta_b["complete"] is True and meta_b["resumes"][0]["episode"] == 32
    assert extra_a["episodes"] == extra_b["episodes"] == 96 and extra_a["updates"] == extra_b["updates"]


@needs_data
def test_resume_se_niega_si_cambia_la_config(tmp_path, monkeypatch):
    from dataPrep.cargar import load_codificador
    from models.bc import PoliticaLineal
    cod = load_codificador()
    bc_ck = tmp_path / "bc.npz"
    PoliticaLineal(cod).guardar(bc_ck)
    cfg = {**cc.load_config("rl"), "episodes": 32, "batch_size": 16, "eval_every": 16,
           "val_pairs": 1, "val_n_per_pair": 4, "seeds": [0]}
    with pytest.raises(RuntimeError):
        _entrenar(tmp_path, monkeypatch, cfg, bc_ck, ArenaFalsa(cortar_tras=6))
    carpeta = next(tmp_path.glob("rl_*_s0"))
    with pytest.raises(SystemExit, match="no coinciden"):
        _entrenar(tmp_path, monkeypatch, {**cfg, "beta_H": 0.0}, bc_ck, ArenaFalsa(), reanudar=carpeta)


# ---------------------------------------------------------------------------
# Salto de trabajo hecho (main.py) con carpetas falsas en un OUTPUT_DIR temporal
# ---------------------------------------------------------------------------
def _corrida(base, nombre, kind, seed, config, **extra):
    d = base / nombre
    d.mkdir()
    (d / "meta.json").write_text(json.dumps({"run_id": nombre, "kind": kind, "seed": seed, "config": config,
                                             **extra}), encoding="utf-8")
    return d


@pytest.fixture
def salida(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "OUTPUT_DIR", tmp_path)
    return tmp_path


def test_incompletas_se_ignoran_y_completas_se_reconocen(salida):
    cfg = {"beta_H": 0.01, "bc_ckpt": "x"}
    hecha = _corrida(salida, "rl_20260101-000000_s0", "rl", 0, cfg, train_wall_s=10.0)
    (hecha / "ckpt_best_val.npz").write_bytes(b"x")
    parcial = _corrida(salida, "rl_20260102-000000_s1", "rl", 1, cfg)        # como la semilla 1 cortada
    (parcial / "ckpt_best_val.npz").write_bytes(b"x")
    assert main.corrida_hecha("rl", 0, False, cfg) == hecha
    assert main.corrida_hecha("rl", 0, False, {**cfg, "beta_H": 0.0}) is None   # otra config: no se salta
    assert main.corrida_hecha("rl", 1, False, cfg) is None
    assert main.ultimo_checkpoint("rl", 0, "ckpt_best_val.npz", False) == hecha / "ckpt_best_val.npz"
    with pytest.raises(SystemExit, match="corrida completa"):
        main.ultimo_checkpoint("rl", 1, "ckpt_best_val.npz", False)
    marcada = _corrida(salida, "rl_20260103-000000_s2", "rl", 2, cfg, complete=True)
    assert main.corrida_hecha("rl", 2, False, cfg) == marcada
    assert main.rl_reanudable(1, False, cfg) is None              # sin ckpt_resume.npz no hay reanudacion


def test_reanudable_exige_misma_config(salida):
    cfg = {"beta_H": 0.01, "bc_ckpt": "x"}
    d = _corrida(salida, "rl_20260101-000000_s1", "rl", 1, cfg)
    (d / "ckpt_resume.npz").write_bytes(b"x")
    assert main.rl_reanudable(1, False, cfg) == d
    with pytest.raises(SystemExit, match="otra config"):
        main.rl_reanudable(1, False, {**cfg, "beta_H": 0.0})


def test_online_rl_salta_semillas_hechas_y_force_las_repite(salida, monkeypatch):
    import training.entrenar_rl as er
    import training.eval.evaluar_arena as ea
    # --- BC completo para las 3 semillas; RL completo solo para la 0 (y ya evaluado) ---
    for s in (0, 1, 2):
        b = _corrida(salida, f"bc_20260101-00000{s}_s{s}", "bc", s, {})
        (b / "model.npz").write_bytes(b"x")
        meta = json.loads((b / "meta.json").read_text(encoding="utf-8"))
        meta["checkpoint"] = str(b / "model.npz")
        (b / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    cfg = cc.load_config("rl")
    cfg0 = er.config_de_corrida(cfg, salida / "bc_20260101-000000_s0" / "model.npz")
    rl0 = _corrida(salida, "rl_20260101-000000_s0", "rl", 0, cfg0, train_wall_s=1.0)
    acfg = cc.load_config("arena")
    _corrida(salida, "arena_agent_vs_usage_val_20260101-000000_s0", "arena_agent_vs_usage_val", 0,
             {"n_per_pair": acfg["n_per_pair_val"], "checkpoint": str(rl0 / "ckpt_best_val.npz"), "mirror": False},
             complete=True)
    entrenadas, evaluadas = [], []

    async def entrenar_falso(arena, cfg, acfg, s, bc_ck, smoke, reanudar=None):
        entrenadas.append(s)
        return salida / f"ck{s}.npz", {}

    async def evaluar_falso(arena, cod, metodo, pares, n, s, **kw):
        evaluadas.append(s)
    monkeypatch.setattr(er, "entrenar", entrenar_falso)
    monkeypatch.setattr(ea, "evaluar", evaluar_falso)
    asyncio.run(main.online_rl(None, None, False, None))
    assert entrenadas == [1, 2] and evaluadas == [1, 2]           # la semilla 0 se salta entera
    entrenadas.clear(), evaluadas.clear()
    asyncio.run(main.online_rl(None, None, False, None, solo_seed=1))
    assert entrenadas == [1]                                      # --seed 1: solo esa
    entrenadas.clear(), evaluadas.clear()
    asyncio.run(main.online_rl(None, None, False, None, force=True, solo_seed=0))
    assert entrenadas == [0] and evaluadas == [0]                 # --force la repite


@pytest.mark.parametrize("args", [["bc", "--seed", "1"], ["vhat", "--seed", "0"], ["val", "--resume"]])
def test_cli_rechaza_combinaciones_no_permitidas(monkeypatch, args):
    monkeypatch.setattr(sys, "argv", ["main.py", *args])
    with pytest.raises(SystemExit) as e:
        main.main()
    assert e.value.code == 2                                      # error de argparse, no se corrio nada
