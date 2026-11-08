"""Test de la arena: el servidor respeta el bring enviado por MethodPlayer (como bring_ok de la Fase 0).

Usa el primer par de val (equipos ya validados). Necesita Showdown local en el
puerto 8000 y splits/; si no estan, se salta.
"""
import asyncio
import json
import socket

import pytest

from bringvgc.brings import BRINGS, bring_index
from bringvgc.config import resolve
from bringvgc.parsing.showdown import parse_packed_team, species_num


def _server_up():
    try:
        with socket.create_connection(("localhost", 8000), timeout=1):
            return True
    except OSError:
        return False


pytestmark = [pytest.mark.arena,
              pytest.mark.skipif(not _server_up(), reason="Showdown no esta corriendo"),
              pytest.mark.skipif(not resolve("splits/val_pairs.json").exists(), reason="faltan splits")]


def test_bring_is_respected():
    from bringvgc.sim.arena import Arena, MethodPlayer

    pair = json.loads(resolve("splits/val_pairs.json").read_text(encoding="utf-8"))[0]
    events = {}
    orig = MethodPlayer._handle_battle_message

    async def handle(self, split_messages):
        tag = split_messages[0][0][1:]
        for msg in split_messages[1:]:
            if len(msg) > 2 and msg[1] in ("switch", "drag", "turn"):
                events.setdefault((self.username, tag), []).append(msg[1:])
        await orig(self, split_messages)

    MethodPlayer._handle_battle_message = handle
    try:
        arena = Arena(concurrency=4, timeout_s=120)
        k_m = bring_index((1, 2), (3, 4))
        k_o = bring_index((0, 5), (2, 3))
        rows = asyncio.run(arena.play(pair["packed_A"], pair["packed_B"], 4, lambda: k_m, lambda: k_o))
        battles = dict(arena.pm.battles)
    finally:
        MethodPlayer._handle_battle_message = orig
    assert len(rows) == 4
    team = parse_packed_team(pair["packed_A"])
    illusion = {i for i, m in enumerate(team) if m["ability"] == "illusion"}
    L, R = BRINGS[k_m]
    for r in rows:
        assert r["bring_m"] == k_m and r["bring_o"] == k_o
        role = battles[r["battle_tag"]].player_role
        leads, seen, started = [], set(), False
        for e in events[(arena.pm.username, r["battle_tag"])]:
            if e[0] == "turn":
                started = True
                continue
            if e[1][:2] != role:
                continue
            pos = next(i for i, m in enumerate(team) if m["num"] == species_num(e[2]))
            seen.add(pos)
            if not started and len(leads) < 2:
                leads.append(pos)
        if not illusion:
            assert sorted(leads) == sorted(L)
            assert seen <= set(L) | set(R)
