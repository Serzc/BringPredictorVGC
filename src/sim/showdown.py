"""Arranca y apaga el servidor local de Showdown (cameronangliss/pokemon-showdown @ 913da36).

BORRADOR asistido por IA.
Uso:
    with servidor_showdown(puerto=8000):
        ... batallas ...
Reglas:
- antes de arrancar se comprueba que el puerto este libre (si no, error claro: puede
  haber otro Showdown abierto; cerrarlo a mano);
- si el puerto no abre en `timeout_s` segundos (60 por defecto), se aborta;
- al salir (aunque haya error o Ctrl+C) se mata el proceso y sus hijos con
  `taskkill /T /F` en Windows (o kill del grupo en otros sistemas), dentro de un finally.
"""
from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import contextmanager

from config.config import OUTPUT_DIR, SHOWDOWN_DIR

PROCESO_ACTUAL: subprocess.Popen | None = None   # lo usa el vigilante de tiempo de main.py


def puerto_abierto(puerto: int, host: str = "localhost") -> bool:
    """True si algo esta escuchando en ese puerto."""
    try:
        with socket.create_connection((host, puerto), timeout=1):
            return True
    except OSError:
        return False


def matar(proc: subprocess.Popen) -> None:
    """Mata Showdown y todos sus procesos hijos."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True)
    else:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


@contextmanager
def servidor_showdown(puerto: int = 8000, timeout_s: float = 60.0):
    # --- 1) el puerto tiene que estar libre ---
    if puerto_abierto(puerto):
        raise SystemExit(f"El puerto {puerto} ya esta ocupado (¿otro Showdown abierto?). "
                         "Cerrar ese proceso y volver a correr.")
    if not (SHOWDOWN_DIR / "pokemon-showdown").exists():
        raise SystemExit(f"No esta Showdown en {SHOWDOWN_DIR}. Correr primero: python src/main.py setup")

    # --- 2) arrancar con --no-security (servidor local, sin login) y log a archivo ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log = open(OUTPUT_DIR / "showdown_server.log", "w", encoding="utf-8")
    extra = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32"
             else {"start_new_session": True})
    proc = subprocess.Popen(["node", "pokemon-showdown", "start", str(puerto), "--no-security"],
                            cwd=SHOWDOWN_DIR, stdout=log, stderr=subprocess.STDOUT, **extra)
    global PROCESO_ACTUAL
    PROCESO_ACTUAL = proc
    try:
        # --- 3) esperar a que abra el puerto ---
        t0 = time.perf_counter()
        while not puerto_abierto(puerto):
            if proc.poll() is not None:
                raise RuntimeError(f"Showdown termino al arrancar (ver {OUTPUT_DIR / 'showdown_server.log'})")
            if time.perf_counter() - t0 > timeout_s:
                raise RuntimeError(f"Showdown no abrio el puerto {puerto} en {timeout_s:.0f} s")
            time.sleep(0.5)
        print(f"[showdown] escuchando en el puerto {puerto} (pid {proc.pid}, {time.perf_counter() - t0:.1f} s)")
        yield proc
    finally:
        # --- 4) apagar siempre ---
        matar(proc)
        log.close()
        print("[showdown] apagado")
