"""Arranca, reinicia y apaga el servidor local de Showdown (cameronangliss/pokemon-showdown @ 913da36).

Incluye reinicio vigilado y logs con fecha agregados el 13-11-2026.
Uso:
    with servidor_showdown(puerto=8000):
        ... batallas ...
        reiniciar_showdown(8000)      # lo llama el vigilante de la arena tras N bloques colgados
Reglas:
- antes de arrancar se comprueba que el puerto este libre (si no, error claro: puede
  haber otro Showdown abierto; cerrarlo a mano);
- si el puerto no abre en `timeout_s` segundos (60 por defecto), se aborta;
- cada arranque (y cada reinicio) escribe su propio log con fecha:
  src/output/showdown_server_<AAAAMMDD-HHMMSS>.log  (antes se pisaba showdown_server.log);
- solo se mata el proceso que arranco este modulo (PROCESO_ACTUAL), nunca otro que tenga el puerto;
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
from datetime import datetime

from config.config import OUTPUT_DIR, SHOWDOWN_DIR

PROCESO_ACTUAL: subprocess.Popen | None = None   # el Showdown que arranco ESTE proceso (y solo ese)
LOG_ACTUAL = None                                # archivo de log abierto del Showdown actual
RUTA_LOG_ACTUAL = None                           # ruta de ese log (para mensajes y meta.json)


def puerto_abierto(puerto: int, host: str = "localhost") -> bool:
    """True si algo esta escuchando en ese puerto."""
    try:
        with socket.create_connection((host, puerto), timeout=1):
            return True
    except OSError:
        return False


def matar(proc: subprocess.Popen) -> None:
    """Mata Showdown y todos sus procesos hijos. Solo recibe un Popen que arrancamos nosotros."""
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


def _arrancar(puerto: int, timeout_s: float) -> subprocess.Popen:
    """Arranca un Showdown nuevo con su propio log con fecha y espera a que abra el puerto."""
    global PROCESO_ACTUAL, LOG_ACTUAL, RUTA_LOG_ACTUAL
    # --- 1) el puerto tiene que estar libre (si no, es de otro: no se toca) ---
    if puerto_abierto(puerto):
        raise SystemExit(f"El puerto {puerto} ya esta ocupado (¿otro Showdown abierto?). "
                         "Cerrar ese proceso y volver a correr.")
    if not (SHOWDOWN_DIR / "pokemon-showdown").exists():
        raise SystemExit(f"No esta Showdown en {SHOWDOWN_DIR}. Correr primero: python src/main.py setup")

    # --- 2) log nuevo por arranque: asi un reinicio no borra la evidencia del cuelgue ---
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    ruta = OUTPUT_DIR / f"showdown_server_{stamp}.log"
    n = 1
    while ruta.exists():                     # dos arranques en el mismo segundo: no pisar
        ruta = OUTPUT_DIR / f"showdown_server_{stamp}_{n}.log"
        n += 1
    log = open(ruta, "w", encoding="utf-8")

    # --- 3) arrancar con --no-security (servidor local, sin login) ---
    extra = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32"
             else {"start_new_session": True})
    proc = subprocess.Popen(["node", "pokemon-showdown", "start", str(puerto), "--no-security"],
                            cwd=SHOWDOWN_DIR, stdout=log, stderr=subprocess.STDOUT, **extra)
    PROCESO_ACTUAL, LOG_ACTUAL, RUTA_LOG_ACTUAL = proc, log, ruta

    # --- 4) esperar a que abra el puerto; si falla, se mata lo que arrancamos ---
    t0 = time.perf_counter()
    try:
        while not puerto_abierto(puerto):
            if proc.poll() is not None:
                raise RuntimeError(f"Showdown termino al arrancar (ver {ruta})")
            if time.perf_counter() - t0 > timeout_s:
                raise RuntimeError(f"Showdown no abrio el puerto {puerto} en {timeout_s:.0f} s")
            time.sleep(0.5)
    except BaseException:
        _apagar_actual()
        raise
    print(f"[showdown] escuchando en el puerto {puerto} (pid {proc.pid}, "
          f"{time.perf_counter() - t0:.1f} s, log {ruta.name})", flush=True)
    return proc


def _apagar_actual() -> None:
    """Mata el Showdown que arrancamos (si hay) y cierra su log. No toca ningun otro proceso."""
    global PROCESO_ACTUAL, LOG_ACTUAL
    if PROCESO_ACTUAL is not None:
        matar(PROCESO_ACTUAL)
    if LOG_ACTUAL is not None:
        LOG_ACTUAL.close()
    PROCESO_ACTUAL, LOG_ACTUAL = None, None


def reiniciar_showdown(puerto: int = 8000, timeout_s: float = 60.0, espera_puerto_s: float = 30.0) -> dict:
    """Mata SOLO el Showdown que arranco este proceso, espera a que se libere el puerto y arranca otro.

    Devuelve un dict para meta.json (pid viejo/nuevo, log nuevo, segundos).
    Si no arrancamos ningun Showdown, o si el puerto sigue ocupado despues de matar el nuestro
    (o sea, lo tiene otro proceso), se aborta con error: nunca se mata algo ajeno.
    """
    if PROCESO_ACTUAL is None:
        raise RuntimeError("reiniciar_showdown: este proceso no arranco ningun Showdown; no se mata nada")
    t0 = time.perf_counter()
    pid_viejo = PROCESO_ACTUAL.pid
    _apagar_actual()                                   # 1) matar el nuestro (taskkill /T sobre SU pid)
    # 2) esperar a que el puerto se cierre; si no se cierra, es de otro proceso y no se toca
    while puerto_abierto(puerto):
        if time.perf_counter() - t0 > espera_puerto_s:
            raise RuntimeError(f"el puerto {puerto} sigue ocupado tras matar el pid {pid_viejo}: "
                               "lo tiene otro proceso; no se reinicia")
        time.sleep(0.5)
    proc = _arrancar(puerto, timeout_s)                # 3) arrancar otro (mismas reglas que el primero)
    info = {"pid_viejo": pid_viejo, "pid_nuevo": proc.pid, "log": RUTA_LOG_ACTUAL.name,
            "segundos": round(time.perf_counter() - t0, 1)}
    print(f"[showdown] reiniciado: {info}", flush=True)
    return info


@contextmanager
def servidor_showdown(puerto: int = 8000, timeout_s: float = 60.0):
    proc = _arrancar(puerto, timeout_s)
    try:
        yield proc
    finally:
        # --- apagar siempre el Showdown ACTUAL (puede ser otro si hubo reinicios) ---
        _apagar_actual()
        print("[showdown] apagado", flush=True)
