"""Arena: batallas reales en el Showdown local con poke-env (fork vgc-bench). Plan v2 §5, §8, §9.

BORRADOR asistido por IA; responsable sugerido: Sergio.
- Dos BringPlayer (sim/jugador.py): el del metodo ("m") y el del rival ("o").
- Un bloque = un par de equipos, una orientacion, k batallas concurrentes.
- Recompensa: +1 gana / -1 pierde; 0 si empate o si pasa del tope de turnos.
- Si un bloque se cuelga mas de `timeout_s`, se cuenta, se recrean los jugadores y se sigue.
- `fecha_limite` (opcional, time.perf_counter): pasada esa hora se aborta (tope del smoke test).
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Callable

from poke_env import AccountConfiguration, ServerConfiguration

from config.config import FORMAT
from dataPrep.parser import parse_packed_team
from sim.jugador import BringPlayer
from utils.spreads import impute_packed


class Arena:
    _contador = 0     # para nombres de usuario unicos dentro del proceso

    def __init__(self, concurrency: int = 8, auto_mega: bool = True, a8_rule: str = "role_spread",
                 max_turns: int = 50, timeout_s: float = 600.0, port: int = 8000,
                 fecha_limite: float | None = None, log=print):
        self.conc, self.auto_mega, self.a8 = concurrency, auto_mega, a8_rule
        self.max_turns, self.timeout_s, self.port = max_turns, timeout_s, port
        self.fecha_limite, self.log = fecha_limite, log
        self.n_timeouts = 0          # bloques colgados descartados
        self.n_battles = 0           # batallas terminadas
        self.busy_s = 0.0            # segundos jugando (para medir T = batallas/s)
        self._crear_jugadores()

    @classmethod
    def desde_config(cls, cfg: dict, fecha_limite=None):
        return cls(cfg["concurrency"], cfg["auto_mega"], cfg["a8_rule"], cfg["max_turns"], cfg["timeout_s"],
                   cfg.get("port", 8000), fecha_limite)

    def _crear_jugadores(self):
        # nombres unicos por proceso: dos procesos con los nombres por defecto chocan en el login
        Arena._contador += 1
        tag = f"{os.getpid() % 100000}{Arena._contador}"
        server = ServerConfiguration(f"ws://localhost:{self.port}/showdown/websocket",
                                     "https://play.pokemonshowdown.com/action.php?")
        kw = dict(battle_format=FORMAT, max_concurrent_battles=self.conc, auto_mega=self.auto_mega,
                  server_configuration=server)
        self.pm = BringPlayer(account_configuration=AccountConfiguration(f"bm{tag}", None), **kw)
        self.po = BringPlayer(account_configuration=AccountConfiguration(f"bo{tag}", None), **kw)

    async def play(self, packed_m: str, packed_o: str, k: int,
                   chooser_m: Callable[[], int], chooser_o: Callable[[], int]) -> list[dict]:
        """Juega k batallas: metodo con packed_m contra el rival con packed_o. Devuelve una fila por batalla."""
        if self.fecha_limite is not None and time.perf_counter() > self.fecha_limite:
            raise TimeoutError("se paso el tiempo total permitido (total_timeout_s)")
        # equipos con naturaleza y stat points rellenados por la regla A8 (igual para ambos lados)
        self.pm.update_team(impute_packed(packed_m, self.a8))
        self.po.update_team(impute_packed(packed_o, self.a8))
        self.pm.chooser, self.po.chooser = chooser_m, chooser_o
        antes = set(self.pm.battles)
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(self.pm.battle_against(self.po, n_battles=k), self.timeout_s)
        except asyncio.TimeoutError:
            self.n_timeouts += 1
            self.log(f"[arena] bloque colgado > {self.timeout_s}s: se descarta y se recrean jugadores")
            self._crear_jugadores()
            return []
        self.busy_s += time.perf_counter() - t0
        filas = []
        for tag in [t for t in self.pm.battles if t not in antes]:
            b = self.pm.battles[tag]
            if not b.finished:
                continue
            tope = b.turn > self.max_turns
            r = 0 if (tope or not (b.won or b.lost)) else (1 if b.won else -1)
            filas.append({"battle_tag": tag, "bring_m": self.pm.chosen.get(tag), "bring_o": self.po.chosen.get(tag),
                          "reward": r, "won": int(r == 1), "turns": b.turn, "capped": int(tope)})
        self.n_battles += len(filas)
        return filas

    @property
    def throughput(self) -> float:
        """T medido: batallas por segundo de juego."""
        return self.n_battles / self.busy_s if self.busy_s else 0.0


async def evaluate_pairs(arena: Arena, method, opponent, pairs: list[dict], n_per_pair: int, rng,
                         greedy: bool = True, block: int | None = None) -> list[dict]:
    """Protocolo de la RQ: por par, n/2 batallas con el metodo en el lado A y n/2 con los lados invertidos."""
    block = block or max(2, arena.conc)
    filas = []
    for pid, p in enumerate(pairs):
        for orient in ("A", "B"):
            pm, po = (p["packed_A"], p["packed_B"]) if orient == "A" else (p["packed_B"], p["packed_A"])
            tm, to = parse_packed_team(pm), parse_packed_team(po)
            prob_m = method.probs(tm, to)              # distribucion del metodo (fija para el bloque)
            prob_o = opponent.probs(to, tm)            # distribucion del rival
            det_m = greedy or method.deterministic

            def ch_m():
                return int(prob_m.argmax()) if det_m else int(rng.choice(90, p=prob_m))

            def ch_o():
                return int(prob_o.argmax()) if opponent.deterministic else int(rng.choice(90, p=prob_o))

            hechas, objetivo = 0, n_per_pair // 2
            while hechas < objetivo:
                k = min(block, objetivo - hechas)
                got = await arena.play(pm, po, k, ch_m, ch_o)
                for r in got:
                    r.update(pair_id=pid, pair_battle=p["battle_id"], orientation=orient,
                             method=method.name, opponent=opponent.name)
                filas += got
                hechas += k
    return filas
