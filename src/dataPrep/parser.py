"""Parser del protocolo de Showdown -> registros (c1, c2, o1, o2, w, t, serie). Lee y limpia los logs.

BORRADOR asistido por IA. NO esta verificado contra replays anotados a mano:
el test de 20 replays (tests/test_parser_annotated.py) queda pendiente de que
los estudiantes llenen las fixtures mirando cada replay.

Reglas (plan v2 §6):
- lideres = los dos primeros switch-in de cada lado despues de |start;
- reservas reveladas V = cualquier otra entrada al campo (switch, drag, replace);
- formas (Mega, Floette-Eternal, Rotom-*), apodos e Illusion se normalizan
  comparando por numero de Pokedex: la Species Clause garantiza que un equipo no
  repite numero, asi que el numero identifica la posicion en la hoja;
- Illusion: |replace| reasigna la ultima entrada de ese slot a la especie real.
  Si Zoroark nunca se revela, la entrada queda atribuida al disfraz; se marca
  `illusion_unresolved` y se reporta.
"""
from __future__ import annotations

import re
from functools import lru_cache

from poke_env.data import GenData

PACKED_FIELDS = ["name", "species", "item", "ability", "moves", "nature", "evs",
                 "gender", "ivs", "shiny", "level", "extra"]
_SERIES = re.compile(r"game-bestof3-([a-z0-9]+-\d+)")
_GAME_NO = re.compile(r"<strong>Game (\d)</strong>")


def to_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


@lru_cache(maxsize=1)
def _dex() -> dict:
    return GenData.from_gen(9).pokedex


@lru_cache(maxsize=None)
def species_num(name: str) -> int | None:
    """Numero de Pokedex de una especie o forma. Quita sufijos de forma si hace falta."""
    dex = _dex()
    parts = name.split(",")[0].strip().split("-")
    while parts:
        sid = to_id("-".join(parts))
        if sid in dex:
            return dex[sid]["num"]
        parts = parts[:-1]
    return None


def parse_packed_team(packed: str) -> list[dict]:
    """Formato packed de Showdown -> lista de 6 dicts (hoja OTS)."""
    mons = []
    for chunk in packed.split("]"):
        f = chunk.split("|")
        f += [""] * (len(PACKED_FIELDS) - len(f))
        d = dict(zip(PACKED_FIELDS, f))
        species = d["species"] or d["name"]
        mons.append({
            "species": to_id(species), "species_name": species,
            "item": to_id(d["item"]), "ability": to_id(d["ability"]),
            "moves": [to_id(m) for m in d["moves"].split(",") if m],
            "num": species_num(species),
        })
    return mons


def team_key(team: list[dict]) -> str:
    """Identidad de equipo para el filtro de leakage: conjunto de especies.

    Mas estricto que comparar la hoja completa: un equipo con las mismas 6
    especies y un movimiento distinto cuenta como 'visto'.
    """
    return "|".join(sorted(m["species"] for m in team))


def parse_battle(battle_id: str, ts: int, log: str) -> dict:
    """Un log de batalla -> registro con equipos, observacion (lideres y reservas reveladas) y ganador."""
    lines = log.split("\n")
    rec = {"battle_id": battle_id, "ts": int(ts), "series_id": None, "game_no": None,
           "teams": [None, None], "packed": [None, None], "obs": [None, None],
           "winner": None, "turns": 0, "valid": True, "reason": "",
           "illusion_unresolved": [False, False], "n_entries": [0, 0]}
    names = {}
    m = _SERIES.search(log)
    if m:
        rec["series_id"] = m.group(1)
        g = _GAME_NO.search(log)
        rec["game_no"] = int(g.group(1)) if g else None

    # --- recorrido del log linea por linea ---
    started = False
    leads = {0: [], 1: []}
    slot_last = {}            # "p1a" -> (side, pos) de la ultima entrada en ese slot
    entries = {0: [], 1: []}  # posiciones en la hoja que entraron al campo (en orden)
    replaced = {0: False, 1: False}
    tie = False
    for line in lines:
        if not line.startswith("|"):
            continue
        p = line.split("|")
        kind = p[1] if len(p) > 1 else ""
        if kind == "player" and len(p) > 3 and p[3]:
            names.setdefault(p[2], p[3])
        elif kind == "showteam":
            side = int(p[2][1]) - 1
            packed = "|".join(p[3:])
            rec["packed"][side] = packed
            rec["teams"][side] = parse_packed_team(packed)
        elif kind == "teamsize" and len(p) > 3 and p[3] != "4":
            rec["valid"], rec["reason"] = False, f"teamsize_not_4:{p[3]}"
        elif kind == "start":
            started = True
        elif kind in ("switch", "drag", "replace") and started:
            ident, details = p[2], p[3]
            side = int(ident[1]) - 1
            team = rec["teams"][side]
            if team is None:
                rec["valid"], rec["reason"] = False, "no_showteam"
                continue
            num = species_num(details)
            pos = next((i for i, mon in enumerate(team) if mon["num"] == num), None)
            if pos is None:
                rec["valid"], rec["reason"] = False, f"species_not_in_team:{details}"
                continue
            # entrada al campo: se ubica la posicion en la hoja por numero de Pokedex
            slot = ident.split(":")[0]
            if kind == "replace":
                # Illusion roto: la entrada anterior de este slot era el disfraz
                replaced[side] = True
                prev = slot_last.get(slot)
                if prev is not None and prev in entries[side]:
                    j = len(entries[side]) - 1 - entries[side][::-1].index(prev)
                    entries[side][j] = pos
                    if prev in leads[side]:
                        leads[side][leads[side].index(prev)] = pos
            else:
                entries[side].append(pos)
                if len(leads[side]) < 2:
                    leads[side].append(pos)
            slot_last[slot] = pos
        elif kind == "turn":
            rec["turns"] = int(p[2])
        elif kind == "win":
            rec["winner_name"] = p[2]
        elif kind == "tie":
            tie = True

    # --- resultado: ganador por nombre de jugador (empate -> None) ---
    if rec["teams"][0] is None or rec["teams"][1] is None:
        rec["valid"], rec["reason"] = False, "no_showteam"
        return rec
    if tie:
        rec["winner"] = None
    elif "winner_name" in rec:
        w = rec.pop("winner_name")
        rec["winner"] = 0 if names.get("p1") == w else (1 if names.get("p2") == w else None)
        if rec["winner"] is None:
            rec["valid"], rec["reason"] = False, "winner_unknown"
    else:
        rec["valid"], rec["reason"] = False, "no_result"

    if not rec["valid"]:
        return rec
    # --- observacion por lado: L = 2 lideres, V = reservas reveladas (0..2); se descarta lo incoherente ---
    for side in (0, 1):
        team = rec["teams"][side]
        has_illusion = any(mon["ability"] == "illusion" for mon in team)
        rec["illusion_unresolved"][side] = has_illusion and not replaced[side]
        L = leads[side]
        V = sorted(set(entries[side]) - set(L))
        rec["n_entries"][side] = len(set(entries[side]))
        if len(L) != 2 or len(set(L)) != 2:
            rec["valid"], rec["reason"] = False, rec["reason"] or "bad_leads"
            continue
        if len(V) > 2:
            rec["valid"], rec["reason"] = False, rec["reason"] or "more_than_4_seen"
            continue
        if len(set(m["num"] for m in team)) != 6 or None in [m["num"] for m in team]:
            rec["valid"], rec["reason"] = False, rec["reason"] or "bad_team_species"
            continue
        rec["obs"][side] = {"leads": sorted(L), "revealed": V, "v": len(V)}
    return rec
