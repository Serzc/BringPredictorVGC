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

    @classmethod
    def reabrir(cls, carpeta: Path) -> "Run":
        """Reabre una corrida existente (para --resume): misma carpeta, mismo run_id, mismo meta.json.

        El tiempo de pared sigue sumando desde el que ya tenia; cada reanudacion queda anotada
        en meta["resumes"] con la fecha y el commit (el codigo pudo cambiar entre medio).
        """
        run = cls.__new__(cls)
        run.dir = Path(carpeta)
        run.meta = json.loads((run.dir / "meta.json").read_text(encoding="utf-8"))
        run.run_id = run.meta["run_id"]
        run.t0 = time.perf_counter() - float(run.meta.get("wall_time_s", 0.0))
        run.meta.setdefault("resumes", []).append(
            {"at": datetime.now().isoformat(timespec="seconds"), "commit": _git_commit()})
        return run

    def save_meta(self, **extra) -> None:
        """Agrega campos al meta.json y actualiza el tiempo de pared (escritura atomica)."""
        self.meta.update(extra)
        self.meta["wall_time_s"] = round(time.perf_counter() - self.t0, 2)
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        os.replace(tmp, self.dir / "meta.json")      # nunca queda un meta.json a medio escribir


# ---------------------------------------------------------------------------
# ¿Esta completa una corrida? (para saltar trabajo hecho en `main.py all` y en --resume)
# ---------------------------------------------------------------------------
def leer_meta(carpeta: Path) -> dict | None:
    """meta.json de una corrida, o None si no existe o esta corrupto."""
    try:
        return json.loads((Path(carpeta) / "meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def corrida_completa(carpeta: Path) -> bool:
    """True si la corrida termino.

    Regla general: meta.json con `complete: true` (lo escriben RL y la evaluacion desde el
    13-11-2026) o con `train_wall_s` (RL: solo se escribe al terminar).
    Corridas anteriores a esa marca: cada paso tiene su campo que solo se escribe al final
    (bc/vhat: `checkpoint` y el archivo existe; dataset: `n_valid`; evaluacion: `summary`
    con batallas > 0 y per_pair.csv). Sin esto, `all` repetiria BC, v-hat y val ya hechos.
    """
    meta = leer_meta(carpeta)
    if meta is None:
        return False
    if meta.get("complete") is True or "train_wall_s" in meta:
        return True
    kind = str(meta.get("kind", ""))
    if kind in ("bc", "vhat"):
        return "checkpoint" in meta and Path(meta["checkpoint"]).exists()
    if kind == "dataset":
        return "n_valid" in meta
    if kind.startswith("arena_"):
        return (meta.get("summary") or {}).get("battles", 0) > 0 and (Path(carpeta) / "per_pair.csv").exists()
    return False
