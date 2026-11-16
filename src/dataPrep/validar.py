"""Validacion de equipos con el TeamValidator del Showdown fijado (via node).

Motivo: hay equipos en los logs que el validador rechaza (p. ej. movimientos que la
especie no aprende); si uno llega a la arena, el reto se rechaza y el bloque se cuelga.
Solo hace falta cuando se eligen pares nuevos (los congelados ya estan validados).
"""
from __future__ import annotations

import subprocess

from config.config import FORMAT, SHOWDOWN_DIR, SRC
from utils.spreads import impute_packed


def validate_packed(teams: list[str], a8_rule: str = "role_spread") -> list[str]:
    """Devuelve 'OK' o la lista de problemas (JSON) por equipo, ya rellenado con la regla A8."""
    payload = "\n".join(impute_packed(t, a8_rule) for t in teams)
    out = subprocess.run(["node", str(SRC / "scripts" / "validate_teams.js"), str(SHOWDOWN_DIR), FORMAT],
                         input=payload, capture_output=True, text=True, encoding="utf-8", check=True)
    res = out.stdout.split("\n")
    if len(res) != len(teams):
        raise RuntimeError(f"validador: {len(res)} respuestas para {len(teams)} equipos\n{out.stderr}")
    return res
