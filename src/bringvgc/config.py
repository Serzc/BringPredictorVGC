"""Configs, semillas y meta.json (plan v2 §11).

Cada corrida escribe results/<run_id>/meta.json con config, commit, semilla,
tiempo de pared y hardware.
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

ROOT = Path(__file__).resolve().parents[2]
SHOWDOWN_COMMIT = "913da36 (cameronangliss/pokemon-showdown, submodulo de vgc-bench)"
FORMAT = "gen9championsvgc2026regma"


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def set_seed(seed: int) -> None:
    """Fija random, numpy y torch. Showdown no se puede sembrar desde aqui:
    la reproducibilidad de las batallas es a nivel de distribucion."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
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
    return {
        "os": platform.platform(), "cpu": _cpu_name(), "cpu_count": os.cpu_count(),
        "python": platform.python_version(), "torch": _pkg("torch"),
        "poke_env": _pkg("poke-env"), "numpy": _pkg("numpy"),
        "showdown_commit": SHOWDOWN_COMMIT, "host": platform.node(),
    }


class Run:
    """Carpeta results/<run_id>/ con meta.json; mide tiempo de pared."""

    def __init__(self, kind: str, config: dict, seed: int | None = None, root: str = "results"):
        self.run_id = f"{kind}_{datetime.now().strftime('%Y%m%d-%H%M%S')}" + (
            f"_s{seed}" if seed is not None else "")
        self.dir = resolve(root) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.perf_counter()
        self.meta = {"run_id": self.run_id, "kind": kind, "seed": seed, "config": config,
                     "commit": _git_commit(), "hardware": hardware(),
                     "started": datetime.now().isoformat(timespec="seconds")}
        self.save_meta()

    def save_meta(self, **extra) -> None:
        self.meta.update(extra)
        self.meta["wall_time_s"] = round(time.perf_counter() - self.t0, 2)
        (self.dir / "meta.json").write_text(json.dumps(self.meta, indent=2, ensure_ascii=False),
                                            encoding="utf-8")
