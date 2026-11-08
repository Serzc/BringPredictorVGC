"""Validacion de equipos con el TeamValidator del Showdown fijado (via node).

Motivo: hay equipos en los logs que el validador rechaza (p. ej. movimientos que la
especie no aprende); si uno llega a la arena, el reto se rechaza y el bloque se cuelga.
Ruta de Showdown: variable BRINGVGC_SHOWDOWN o ../vgc-bench/pokemon-showdown.
"""
from __future__ import annotations

import os
import subprocess

from ..config import FORMAT, ROOT
from ..features.spreads import impute_packed

DEFAULT_SD = ROOT.parent / "vgc-bench" / "pokemon-showdown"


def validate_packed(teams: list[str], a8_rule: str = "role_spread") -> list[str]:
    """Devuelve 'OK' o la lista de problemas (JSON) por equipo, ya imputado con A8."""
    sd = os.environ.get("BRINGVGC_SHOWDOWN") or str(DEFAULT_SD)
    payload = "\n".join(impute_packed(t, a8_rule) for t in teams)
    out = subprocess.run(["node", str(ROOT / "scripts" / "validate_teams.js"), sd, FORMAT],
                         input=payload, capture_output=True, text=True, encoding="utf-8", check=True)
    res = out.stdout.split("\n")
    if len(res) != len(teams):
        raise RuntimeError(f"validador: {len(res)} respuestas para {len(teams)} equipos\n{out.stderr}")
    return res
