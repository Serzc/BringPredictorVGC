"""Arena: Showdown local (913da36) + poke-env (fork vgc-bench). Plan v2 §5, §8, §9.

BORRADOR asistido por IA; responsable sugerido: Sergio.

- MethodPlayer = BringPlayer de la Fase 0 sin el registro de eventos: solo cambia
  teampreview(); choose_move() es pi_b = SimpleHeuristicsPlayer + auto_mega,
  identica en los dos lados por construccion.
- Cada bloque = un par de equipos, una orientacion, k batallas concurrentes.
- Recompensa: +1 / -1; 0 si empate o si la batalla pasa del tope de turnos.
- Si un bloque se cuelga mas de `timeout_s`, se registra, se recrean los
  jugadores y se sigue (F2).
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Callable

from poke_env import AccountConfiguration
from poke_env.battle.move import Move
from poke_env.player import SimpleHeuristicsPlayer

from ..brings import bring_to_team_order
from ..config import FORMAT
from ..features.spreads import impute_packed


class MethodPlayer(SimpleHeuristicsPlayer):
    def __init__(self, *args, auto_mega: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.auto_mega = auto_mega
        self.chooser: Callable[[], int] = lambda: 0
        self.chosen: dict[str, int] = {}

    def teampreview(self, battle):
        k = int(self.chooser())
        self.chosen[battle.battle_tag] = k
        return "/team " + bring_to_team_order(k)

    def choose_move(self, battle):
        order = super().choose_move(battle)
        if not self.auto_mega or not hasattr(order, "first_order"):
            return order
        for slot, single in enumerate((order.first_order, order.second_order)):
            if (single is not None and isinstance(getattr(single, "order", None), Move)
                    and battle.can_mega_evolve[slot]):
                single.mega = True
                break
        return order


class Arena:
    def __init__(self, concurrency: int = 8, auto_mega: bool = True, a8_rule: str = "role_spread",
                 max_turns: int = 50, timeout_s: float = 600.0, log=print):
        self.conc, self.auto_mega, self.a8 = concurrency, auto_mega, a8_rule
        self.max_turns, self.timeout_s, self.log = max_turns, timeout_s, log
        self.n_timeouts = 0
        self.n_battles = 0
        self.busy_s = 0.0
        self._make_players()

    _counter = 0

    def _make_players(self):
        # Nombres unicos por proceso: dos procesos con los nombres por defecto de
        # poke-env ("MethodPlayer 1") chocan en el login del servidor.
        Arena._counter += 1
        tag = f"{os.getpid() % 100000}{Arena._counter}"
        kw = dict(battle_format=FORMAT, max_concurrent_battles=self.conc, auto_mega=self.auto_mega)
        self.pm = MethodPlayer(account_configuration=AccountConfiguration(f"bm{tag}", None), **kw)
        self.po = MethodPlayer(account_configuration=AccountConfiguration(f"bo{tag}", None), **kw)

    def team(self, packed: str) -> str:
        return impute_packed(packed, self.a8)

    async def play(self, packed_m: str, packed_o: str, k: int,
                   chooser_m: Callable[[], int], chooser_o: Callable[[], int]) -> list[dict]:
        """Juega k batallas: metodo con packed_m contra rival con packed_o."""
        self.pm.update_team(self.team(packed_m))
        self.po.update_team(self.team(packed_o))
        self.pm.chooser, self.po.chooser = chooser_m, chooser_o
        before = set(self.pm.battles)
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(self.pm.battle_against(self.po, n_battles=k), self.timeout_s)
        except asyncio.TimeoutError:
            self.n_timeouts += 1
            self.log(f"[arena] bloque colgado > {self.timeout_s}s: se descarta y se recrean jugadores")
            self._make_players()
            return []
        self.busy_s += time.perf_counter() - t0
        rows = []
        for tag in [t for t in self.pm.battles if t not in before]:
            b = self.pm.battles[tag]
            if not b.finished:
                continue
            capped = b.turn > self.max_turns
            r = 0 if (capped or not (b.won or b.lost)) else (1 if b.won else -1)
            rows.append({"battle_tag": tag, "bring_m": self.pm.chosen.get(tag),
                         "bring_o": self.po.chosen.get(tag), "reward": r, "won": int(r == 1),
                         "turns": b.turn, "capped": int(capped)})
        self.n_battles += len(rows)
        return rows

    @property
    def throughput(self) -> float:
        return self.n_battles / self.busy_s if self.busy_s else 0.0


async def evaluate_pairs(arena: Arena, method, opponent, pairs: list[dict], n_per_pair: int,
                         rng, greedy: bool = True, block: int | None = None,
                         parse=None) -> list[dict]:
    """Protocolo de la RQ: por par, n/2 con el metodo en A y n/2 con los lados invertidos."""
    from ..parsing.showdown import parse_packed_team
    parse = parse or parse_packed_team
    block = block or max(2, arena.conc)
    rows = []
    for pid, p in enumerate(pairs):
        for orient in ("A", "B"):
            pm, po = (p["packed_A"], p["packed_B"]) if orient == "A" else (p["packed_B"], p["packed_A"])
            tm, to = parse(pm), parse(po)
            prob_m = method.probs(tm, to)
            prob_o = opponent.probs(to, tm)
            det_m = greedy or method.deterministic

            def ch_m():
                return int(prob_m.argmax()) if det_m else int(rng.choice(90, p=prob_m))

            def ch_o():
                return int(prob_o.argmax()) if opponent.deterministic else int(rng.choice(90, p=prob_o))

            done, target = 0, n_per_pair // 2
            while done < target:
                k = min(block, target - done)
                got = await arena.play(pm, po, k, ch_m, ch_o)
                for r in got:
                    r.update(pair_id=pid, pair_battle=p["battle_id"], orientation=orient,
                             method=method.name, opponent=opponent.name)
                rows += got
                done += k
    return rows
