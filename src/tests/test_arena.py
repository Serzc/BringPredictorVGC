"""Test de la arena: el servidor respeta el bring enviado por BringPlayer (como bring_ok de la Fase 0).

Usa el primer par de val (equipos ya validados). Arranca Showdown con sim/showdown.py si el
puerto 8000 esta libre y Showdown esta instalado (python src/main.py setup); si no, se salta.
"""
import asyncio

import pytest

from config.config import SHOWDOWN_DIR
from dataPrep.parser import parse_packed_team, species_num
from dataPrep.particion import load_pairs
from sim.showdown import puerto_abierto
from utils.brings import BRINGS, bring_index

pytestmark = [pytest.mark.arena,
              pytest.mark.skipif(puerto_abierto(8000), reason="puerto 8000 ocupado (otro Showdown abierto)"),
              pytest.mark.skipif(not (SHOWDOWN_DIR / "pokemon-showdown").exists(),
                                 reason="Showdown no instalado (python src/main.py setup)")]


def test_bring_is_respected():
    from sim.arena import Arena
    from sim.jugador import BringPlayer
    from sim.showdown import servidor_showdown

    pair = load_pairs("val")[0]
    eventos = {}
    original = BringPlayer._handle_battle_message

    async def handle(self, split_messages):
        tag = split_messages[0][0][1:]
        for msg in split_messages[1:]:
            if len(msg) > 2 and msg[1] in ("switch", "drag", "turn"):
                eventos.setdefault((self.username, tag), []).append(msg[1:])
        await original(self, split_messages)

    BringPlayer._handle_battle_message = handle
    k_m = bring_index((1, 2), (3, 4))
    k_o = bring_index((0, 5), (2, 3))
    try:
        with servidor_showdown(8000):
            async def correr():
                arena = Arena(concurrency=4, timeout_s=120)
                filas = await arena.play(pair["packed_A"], pair["packed_B"], 4, lambda: k_m, lambda: k_o)
                return arena, filas
            arena, filas = asyncio.run(correr())
            battles = dict(arena.pm.battles)
    finally:
        BringPlayer._handle_battle_message = original
    assert len(filas) == 4
    team = parse_packed_team(pair["packed_A"])
    illusion = {i for i, m in enumerate(team) if m["ability"] == "illusion"}
    L, R = BRINGS[k_m]
    for r in filas:
        assert r["bring_m"] == k_m and r["bring_o"] == k_o
        role = battles[r["battle_tag"]].player_role
        leads, vistos, empezo = [], set(), False
        for e in eventos[(arena.pm.username, r["battle_tag"])]:
            if e[0] == "turn":
                empezo = True
                continue
            if e[1][:2] != role:
                continue
            pos = next(i for i, m in enumerate(team) if m["num"] == species_num(e[2]))
            vistos.add(pos)
            if not empezo and len(leads) < 2:
                leads.append(pos)
        if not illusion:
            assert sorted(leads) == sorted(L)
            assert vistos <= set(L) | set(R)
