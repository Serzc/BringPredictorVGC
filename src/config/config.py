"""Rutas, configuracion, semillas y meta.json de cada corrida.

BORRADOR asistido por IA. Cada corrida escribe src/output/<run_id>/meta.json con
la configuracion usada, el commit, la semilla, el tiempo de pared y el hardware
(requisito del curso: reproducibilidad y evidencia de cada numero).
"""
from __future__ import annotations

import json
import os
import platform
import random
import subprocess
import time
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import yaml

# --- Rutas fijas del proyecto (todas relativas a src/) ---
SRC = Path(__file__).resolve().parents[1]          # .../BringPredictorVGC/src
REPO = SRC.parent                                    # raiz del repo
CONFIG_DIR = SRC / "config"
DATASET_DIR = SRC / "dataset"                        # raw/ y clean/ no se versionan; splits/ si
RAW_DIR = DATASET_DIR / "raw"
CLEAN_DIR = DATASET_DIR / "clean"
SPLITS_DIR = DATASET_DIR / "splits"
OUTPUT_DIR = SRC / "output"
VGC_BENCH_DIR = REPO.parent / "vgc-bench"            # clon de vgc-bench (con Showdown como submodulo)
SHOWDOWN_DIR = Path(os.environ.get("BRINGVGC_SHOWDOWN") or VGC_BENCH_DIR / "pokemon-showdown")

# --- Versiones fijadas ---
SHOWDOWN_COMMIT = "913da36"                          # cameronangliss/pokemon-showdown (submodulo de vgc-bench)
FORMAT = "gen9championsvgc2026regma"                 # formato de Showdown de Reg M-A


def load_config(name: str, smoke: bool = False) -> dict:
    """Lee src/config/<name>.yaml. Con smoke=True aplica encima la seccion <name> de smoke.yaml."""
    with open(CONFIG_DIR / f"{name}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if smoke:
        with open(CONFIG_DIR / "smoke.yaml", encoding="utf-8") as f:
            cfg.update((yaml.safe_load(f) or {}).get(name, {}))
    return cfg


def set_seed(seed: int) -> None:
    """Fija las semillas de random y numpy.

    Showdown no se puede sembrar desde aqui: la reproducibilidad de las batallas
    es a nivel de distribucion (por eso se reportan varias semillas y muchos pares).
    """
    random.seed(seed)
    np.random.seed(seed)


def _git_commit() -> str:
    """Commit corto actual; se marca '-dirty' si hay cambios sin commitear."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _cpu_name() -> str:
    if platform.system() == "Windows":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=20)
            if out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    return platform.processor()


def _pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def hardware() -> dict:
    """Hardware y versiones de software (se guarda en cada meta.json)."""
    return {
        "os": platform.platform(), "cpu": _cpu_name(), "cpu_count": os.cpu_count(),
        "python": platform.python_version(), "numpy": _pkg("numpy"), "pandas": _pkg("pandas"),
        "poke_env": _pkg("poke-env"), "showdown_commit": SHOWDOWN_COMMIT, "host": platform.node(),
    }


class Run:
    """Carpeta src/output/<run_id>/ con su meta.json. Mide el tiempo de pared."""

    def __init__(self, kind: str, config: dict, seed: int | None = None, smoke: bool = False):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{'smoke_' if smoke else ''}{kind}_{stamp}" + (f"_s{seed}" if seed is not None else "")
        self.dir = OUTPUT_DIR / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.perf_counter()
        self.meta = {"run_id": self.run_id, "kind": kind, "seed": seed, "smoke_test": smoke,
                     "config": config, "commit": _git_commit(), "hardware": hardware(),
                     "started": datetime.now().isoformat(timespec="seconds")}
        self.save_meta()

    def save_meta(self, **extra) -> None:
        """Agrega campos al meta.json y actualiza el tiempo de pared."""
        self.meta.update(extra)
        self.meta["wall_time_s"] = round(time.perf_counter() - self.t0, 2)
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2, ensure_ascii=False, default=str),
                                            encoding="utf-8")
