"""Fase 0: spikes del arena (b, c, d del plan, ETAPA2_PLAN.md §12).

BORRADOR (asistido por IA). Revisarlo y defenderlo antes de hacer commit.

Que comprueba:
  (b) BringPlayer.teampreview: el servidor respeta el bring enviado. Los 2
      primeros salen de lideres, los 2 siguientes son reservas y los 2 ultimos
      nunca entran a la batalla.
  (c) SimpleHeuristicsPlayer en dobles: termina batallas completas y usa la
      megaevolucion cuando trae una megapiedra.
  (d) Throughput T (batallas/s) para varios niveles de concurrencia.
  + Chequeo de simetria: bring aleatorio vs bring aleatorio, con la misma pi_b
    y los lados intercambiados, deberia dar ~50 %.

Requisitos: Showdown local corriendo (puerto 8000) y data/sample_teams.txt
generado con scripts/extract_teams.py.

Uso:
  python scripts/phase0_arena_spikes.py --battles 100 --concurrency 1 4 8 --seed 0

Salida en results/phase0/:
  battles_<run>.csv   una fila por batalla (vista desde el lado "A")
  summary_<run>.csv   una fila por nivel de concurrencia
  meta_<run>.json     hardware, versiones, argumentos
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import platform
import random
import re
import time
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

from poke_env.battle.move import Move
from poke_env.player import SimpleHeuristicsPlayer

FMT = "gen9championsvgc2026regma"
TEAMS_FILE = Path("data/sample_teams.txt")
OUT_DIR = Path("results/phase0")
EVENTS = {"switch", "drag", "-mega", "turn"}


def to_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def is_mega_stone(item: str | None) -> bool:
    # Aproximacion para el spike: las megapiedras terminan en "ite", "itex" o "itey".
    # Eviolite es la excepcion conocida.
    if not item:
        return False
    item = to_id(item)
    return item != "eviolite" and bool(re.search(r"ite[xy]?$", item))


class BringPlayer(SimpleHeuristicsPlayer):
    """pi_b = SimpleHeuristicsPlayer. Solo cambia la eleccion del bring.

    Aqui el bring es aleatorio (con semilla). En las fases siguientes cada
    metodo (Usage, BC, agente RL...) sera una subclase que solo cambia
    choose_bring().
    """

    def __init__(self, *args, seed: int = 0, auto_mega: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.rng = random.Random(seed)
        self.auto_mega = auto_mega
        self.sent_bring: dict[str, list[str]] = {}   # battle_tag -> 4 especies (orden enviado)
        self.left_out: dict[str, list[str]] = {}     # battle_tag -> 2 especies fuera
        self.has_mega_stone: dict[str, bool] = {}    # algun Pokemon traido lleva megapiedra
        self.events: dict[str, list[list[str]]] = {}

    def choose_bring(self, battle) -> list[int]:
        """Devuelve 4 indices (1..6): 2 lideres y luego 2 reservas."""
        return self.rng.sample(range(1, 7), 4)

    def teampreview(self, battle):
        mons = list(battle.team.values())
        bring = self.choose_bring(battle)
        for i in bring:
            mons[i - 1]._selected_in_teampreview = True
        self.sent_bring[battle.battle_tag] = [to_id(mons[i - 1].species) for i in bring]
        self.left_out[battle.battle_tag] = [
            to_id(m.species) for k, m in enumerate(mons, 1) if k not in bring
        ]
        self.has_mega_stone[battle.battle_tag] = any(
            is_mega_stone(mons[i - 1].item) for i in bring
        )
        return "/team " + "".join(str(i) for i in bring)

    def choose_move(self, battle):
        # Hallazgo del spike (c): SimpleHeuristicsPlayer del fork nunca pone
        # mega=True, asi que nunca megaevoluciona. Con auto_mega, pi_b pasa a ser
        # "SimpleHeuristics + megaevolucionar en cuanto se pueda" (el primer slot
        # que pueda, porque solo se permite una mega por turno y por batalla).
        # Es un cambio a pi_b: decidir si se adopta y declararlo en el articulo.
        order = super().choose_move(battle)
        if not self.auto_mega or not hasattr(order, "first_order"):
            return order
        for slot, single in enumerate((order.first_order, order.second_order)):
            if (single is not None and isinstance(getattr(single, "order", None), Move)
                    and battle.can_mega_evolve[slot]):
                single.mega = True
                break
        return order

    async def _handle_battle_message(self, split_messages):
        # Guarda los eventos que hacen falta para verificar (b) y (c).
        # Se sobrescribe un metodo privado de poke-env: aceptable en un spike.
        tag = split_messages[0][0][1:]
        log = self.events.setdefault(tag, [])
        for msg in split_messages[1:]:
            if len(msg) > 1 and msg[1] in EVENTS:
                log.append(msg[1:])
        await super()._handle_battle_message(split_messages)


def matches(seen: str, chosen: list[str]) -> str | None:
    """Empareja la especie vista en batalla (puede ser forma Mega) con la enviada."""
    for c in chosen:
        if seen == c or seen.startswith(c) or c.startswith(seen):
            return c
    return None


def check_battle(player: BringPlayer, battle) -> dict:
    tag, role = battle.battle_tag, battle.player_role
    sent, out = player.sent_bring.get(tag, []), player.left_out.get(tag, [])
    leads, appeared, megas, started = [], set(), 0, False
    for ev in player.events.get(tag, []):
        kind = ev[0]
        if kind == "turn":
            started = True
            continue
        side = ev[1][:2] if len(ev) > 1 else ""
        if kind == "-mega":
            megas += side == role
            continue
        if side != role:
            continue
        species = to_id(ev[2].split(",")[0])
        m = matches(species, sent + out)
        appeared.add(m or species)
        if not started and len(leads) < 2:
            leads.append(m or species)
    leads_ok = sorted(leads) == sorted(sent[:2])
    no_left_out = not (appeared & set(out))
    within_bring = appeared <= set(sent)
    return {
        "sent_bring": " ".join(sent),
        "left_out": " ".join(out),
        "leads_seen": " ".join(leads),
        "appeared": " ".join(sorted(appeared)),
        "leads_ok": leads_ok,
        "bring_ok": leads_ok and no_left_out and within_bring,
        "has_mega_stone": player.has_mega_stone.get(tag, False),
        "mega_used": megas > 0,
    }


async def run_level(teams: list[str], n_battles: int, conc: int, seed: int, writer, run_id,
                    auto_mega: bool = True):
    """Juega n_battles: pares de equipos rotando, mitad con cada orientacion."""
    kw = dict(battle_format=FMT, max_concurrent_battles=conc, auto_mega=auto_mega)
    a = BringPlayer(team=teams[0], seed=seed, **kw)
    b = BringPlayer(team=teams[1], seed=seed + 1, **kw)
    pair_rng = random.Random(seed)
    per_block = max(2, conc)  # batallas por bloque (un par y una orientacion)
    rows, done, swapped = [], 0, False

    start = time.perf_counter()
    while done < n_battles:
        if not swapped:
            i, j = pair_rng.sample(range(len(teams)), 2)
        ta, tb = (teams[j], teams[i]) if swapped else (teams[i], teams[j])
        ia, ib = (j, i) if swapped else (i, j)
        a.update_team(ta)
        b.update_team(tb)
        k = min(per_block, n_battles - done)
        before = set(a.battles)
        await a.battle_against(b, n_battles=k)
        for tag in [t for t in a.battles if t not in before]:
            ba, bb = a.battles[tag], b.battles.get(tag)
            row = {
                "run_id": run_id, "concurrency": conc, "battle_tag": tag,
                "team_A": ia, "team_B": ib, "swapped": swapped,
                "winner": "A" if ba.won else ("B" if ba.lost else "draw"),
                "turns": ba.turn,
            }
            row.update({f"A_{k2}": v for k2, v in check_battle(a, ba).items()})
            if bb is not None:
                row.update({f"B_{k2}": v for k2, v in check_battle(b, bb).items()})
            rows.append(row)
            writer.writerow(row)
        done += k
        swapped = not swapped
    wall = time.perf_counter() - start

    n = len(rows)
    sides = [(r[f"{s}_bring_ok"], r[f"{s}_has_mega_stone"], r[f"{s}_mega_used"])
             for r in rows for s in "AB" if f"{s}_bring_ok" in r]
    with_stone = [s for s in sides if s[1]]
    return {
        "run_id": run_id, "concurrency": conc, "battles": n,
        "wall_s": round(wall, 2), "T_battles_per_s": round(n / wall, 3) if wall else 0,
        "winrate_A": round(sum(r["winner"] == "A" for r in rows) / n, 3) if n else 0,
        "draws": sum(r["winner"] == "draw" for r in rows),
        "bring_ok_rate": round(sum(s[0] for s in sides) / len(sides), 3) if sides else 0,
        "sides_with_mega_stone": len(with_stone),
        "mega_used_rate_when_stone": round(sum(s[2] for s in with_stone) / len(with_stone), 3)
        if with_stone else None,
        "median_turns": sorted(r["turns"] for r in rows)[n // 2] if n else 0,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=100, help="batallas por nivel de concurrencia")
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-mega", action="store_true",
                    help="pi_b = SimpleHeuristicsPlayer original, sin megaevolucion")
    args = ap.parse_args()

    teams = [t for t in TEAMS_FILE.read_text(encoding="utf-8").splitlines() if t.strip()]
    if len(teams) < 2:
        raise SystemExit("Hacen falta >= 2 equipos en data/sample_teams.txt")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    meta = {
        "run_id": run_id, "args": vars(args), "format": FMT, "n_teams": len(teams),
        "python": platform.python_version(), "poke_env": version("poke-env"),
        "showdown_commit": "913da36 (cameronangliss/pokemon-showdown, submodulo de vgc-bench)",
        "os": platform.platform(), "cpu": platform.processor(), "cpu_count": os.cpu_count(),
        "pi_b": "SimpleHeuristicsPlayer" + ("" if args.no_mega else " + auto_mega"),
        "note": "Stats imputados al azar (temporal, ver extract_teams.py / A8)",
    }
    (OUT_DIR / f"meta_{run_id}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summaries = []
    with open(OUT_DIR / f"battles_{run_id}.csv", "w", newline="", encoding="utf-8") as f:
        writer = None
        for conc in args.concurrency:
            print(f"-> concurrencia {conc}: {args.battles} batallas...")

            class LazyWriter:  # crea el encabezado con la primera fila
                def writerow(self, row):
                    nonlocal writer
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
                        writer.writeheader()
                    writer.writerow(row)

            s = await run_level(teams, args.battles, conc, args.seed, LazyWriter(), run_id,
                                auto_mega=not args.no_mega)
            summaries.append(s)
            print("   ", s)

    with open(OUT_DIR / f"summary_{run_id}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summaries[0]))
        w.writeheader()
        w.writerows(summaries)

    print("\nCriterios de salida de la Fase 0:")
    for s in summaries:
        print(f"  conc={s['concurrency']}: T={s['T_battles_per_s']}/s | "
              f"bring_ok={s['bring_ok_rate']} (esperado 1.0) | "
              f"winrate_A={s['winrate_A']} (esperado ~0.5) | "
              f"mega con piedra={s['mega_used_rate_when_stone']} | empates={s['draws']}")
    best = max(summaries, key=lambda s: s["T_battles_per_s"])["T_battles_per_s"]
    verdict = ("presupuesto completo (~22 h)" if best >= 10 else
               "repartir entre las dos PCs" if best >= 3 else
               "aplicar recortes de §9.4 o plan B (R2)")
    print(f"  Mejor T = {best}/s -> {verdict}")


if __name__ == "__main__":
    asyncio.run(main())
