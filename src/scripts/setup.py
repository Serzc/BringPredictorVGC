"""Paso 'setup': deja listo todo lo que no esta en el repo. Idempotente (se puede correr varias veces).

BORRADOR asistido por IA. Se llama con: python src/main.py setup
1) Log set Reg M-A: src/dataset/raw/logs_gen9championsvgc2026regma.json
   - si ya esta y pesa exactamente lo esperado, no se toca;
   - si existe la copia local (config data.yaml: local_copy) con el tamano correcto, se copia;
   - si no, se descarga de Hugging Face en la revision fijada (urllib, sin dependencias) y se comprueba el tamano.
2) vgc-bench con Showdown como submodulo, en ../vgc-bench (al lado del repo):
   - si ya existe, se reutiliza y solo se comprueba que Showdown este en el commit 913da36;
   - si no, git clone --recurse-submodules y checkout del commit fijado.
3) npm install dentro de Showdown si falta node_modules.
Los pares congelados (src/dataset/splits/) NUNCA se tocan aqui.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import urllib.request

from config.config import RAW_DIR, REPO, SHOWDOWN_COMMIT, SHOWDOWN_DIR, VGC_BENCH_DIR, load_config

VGC_BENCH_URL = "https://github.com/cameronangliss/vgc-bench.git"


def _tamano_ok(path, esperado: int) -> bool:
    return path.exists() and path.stat().st_size == esperado


def preparar_datos(cfg: dict) -> None:
    destino = RAW_DIR / cfg["logs_file"]
    esperado = int(cfg["logs_size_bytes"])
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if _tamano_ok(destino, esperado):
        print(f"[setup] log set ya presente ({esperado:,} bytes)")
        return
    copia = (REPO / cfg["local_copy"]).resolve()
    if _tamano_ok(copia, esperado):
        shutil.copy2(copia, destino)
        print(f"[setup] log set copiado de {copia}")
        return
    # descarga en un archivo temporal; solo se renombra si el tamano coincide
    parcial = destino.with_suffix(".part")
    print(f"[setup] descargando {cfg['hf_url']} ...")
    urllib.request.urlretrieve(cfg["hf_url"], parcial)
    if not _tamano_ok(parcial, esperado):
        raise SystemExit(f"[setup] la descarga pesa {parcial.stat().st_size:,} bytes, se esperaban {esperado:,}")
    parcial.replace(destino)
    print(f"[setup] log set descargado ({esperado:,} bytes)")


def _git(*args, cwd=None) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def preparar_showdown() -> None:
    if not VGC_BENCH_DIR.exists():
        print(f"[setup] clonando vgc-bench en {VGC_BENCH_DIR} ...")
        subprocess.run(["git", "clone", "--recurse-submodules", VGC_BENCH_URL, str(VGC_BENCH_DIR)], check=True)
    if not (SHOWDOWN_DIR / "pokemon-showdown").exists():
        subprocess.run(["git", "submodule", "update", "--init"], cwd=VGC_BENCH_DIR, check=True)
    actual = _git("rev-parse", "--short=7", "HEAD", cwd=SHOWDOWN_DIR)
    if actual != SHOWDOWN_COMMIT:
        # solo se cambia de commit si no hay cambios locales en Showdown
        if _git("status", "--porcelain", "--untracked-files=no", cwd=SHOWDOWN_DIR):
            raise SystemExit(f"[setup] Showdown esta en {actual} con cambios locales; se esperaba {SHOWDOWN_COMMIT}")
        _git("checkout", SHOWDOWN_COMMIT, cwd=SHOWDOWN_DIR)
    print(f"[setup] Showdown en {SHOWDOWN_DIR} @ {SHOWDOWN_COMMIT}")
    if not (SHOWDOWN_DIR / "node_modules").exists():
        npm = "npm.cmd" if sys.platform == "win32" else "npm"
        subprocess.run([npm, "install"], cwd=SHOWDOWN_DIR, check=True)
    print("[setup] dependencias de node listas")


def main() -> None:
    preparar_datos(load_config("data"))
    preparar_showdown()
    print("[setup] listo. Siguiente paso: python src/main.py all   (o all --smoke para probar)")
