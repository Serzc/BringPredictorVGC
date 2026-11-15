"""Arena: batallas reales en el Showdown local con poke-env (fork vgc-bench). Plan v2 §5, §8, §9.

BORRADOR asistido por IA; responsable sugerido: Sergio. Limpieza y vigilante agregados el 13-11-2026.
- Dos BringPlayer (sim/jugador.py): el del metodo ("m") y el del rival ("o").
- Un bloque = un par de equipos, una orientacion, k batallas concurrentes.
- Recompensa: +1 gana / -1 pierde; 0 si empate o si pasa del tope de turnos.
- Si un bloque se cuelga mas de `timeout_s` (o los jugadores no logran entrar al servidor en
  `login_timeout_s`), el bloque se descarta y:
    1) los jugadores viejos se rinden (/forfeit) y salen (/leave) de sus batallas abiertas y se
       cierra su websocket (antes quedaban vivos en Showdown y se acumulaban: efecto bola de nieve);
    2) si van `max_timeouts_seguidos` fallos seguidos, se reinicia Showdown (`reiniciar`, lo pasa main.py);
    3) se crean jugadores nuevos.
  Cada fallo y cada reinicio queda en `self.eventos` (va al meta.json de la corrida).
- `fecha_limite` (opcional, time.perf_counter): pasada esa hora se aborta (tope del smoke test).
"""
from __future__ import annotations

import asyncio
import os
import time
from datetime import datetime
from typing import Callable

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.concurrency import handle_threaded_coroutines

from config.config import FORMAT
from dataPrep.parser import parse_packed_team
from sim.jugador import BringPlayer
from utils.spreads import impute_packed


class Arena:
    _contador = 0     # para nombres de usuario unicos dentro del proceso

    def __init__(self, concurrency: int = 8, auto_mega: bool = True, a8_rule: str = "role_spread",
                 max_turns: int = 50, timeout_s: float = 600.0, port: int = 8000,
                 fecha_limite: float | None = None, log=print, reiniciar: Callable[[], dict] | None = None,
                 max_timeouts_seguidos: int = 2, login_timeout_s: float = 30.0, limpieza_timeout_s: float = 30.0):
        self.conc, self.auto_mega, self.a8 = concurrency, auto_mega, a8_rule
        self.max_turns, self.timeout_s, self.port = max_turns, timeout_s, port
        self.fecha_limite, self.log = fecha_limite, log
        # --- vigilante: quien reinicia Showdown y cuantos fallos seguidos se toleran ---
        self.reiniciar = reiniciar                   # None = sin reinicio (p. ej. en tests)
        self.max_seguidos = max_timeouts_seguidos
        self.login_timeout_s, self.limpieza_timeout_s = login_timeout_s, limpieza_timeout_s
        self.n_timeouts = 0          # bloques colgados descartados (total del proceso)
        self.seguidos = 0            # fallos seguidos (se pone en 0 con cada bloque que termina bien)
        self.n_reinicios = 0         # reinicios de Showdown hechos por el vigilante
        self.eventos: list[dict] = []    # historial de fallos/reinicios (para meta.json)
        self.n_battles = 0           # batallas terminadas
        self.busy_s = 0.0            # segundos jugando (para medir T = batallas/s)
        self._terminadas_antes: dict[str, set] = {}   # batallas terminadas en el bloque anterior (ver _podar)
        self._crear_jugadores()

    @classmethod
    def desde_config(cls, cfg: dict, fecha_limite=None, reiniciar=None):
        return cls(cfg["concurrency"], cfg["auto_mega"], cfg["a8_rule"], cfg["max_turns"], cfg["timeout_s"],
                   cfg.get("port", 8000), fecha_limite, reiniciar=reiniciar,
                   max_timeouts_seguidos=cfg.get("max_timeouts_seguidos", 2),
                   login_timeout_s=cfg.get("login_timeout_s", 30.0),
                   limpieza_timeout_s=cfg.get("limpieza_timeout_s", 30.0))

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
        self._terminadas_antes = {}

    # ------------------------------------------------------------------
    # Limpieza de jugadores colgados (API verificada en .venv/poke_env 0.15.0:
    # Player.battles, Player.ps_client.send_message(msg, room), PSClient.stop_listening(),
    # PSClient._listening_coroutine (future del listener), PSClient._active_tasks, PSClient.loop,
    # poke_env.concurrency.handle_threaded_coroutines)
    # ------------------------------------------------------------------
    async def _en_poke_loop(self, coro, jugador, ya_en_hilo: bool = False) -> None:
        """Corre `coro` en el hilo de poke-env (ahi vive el websocket) con tope de tiempo; nunca lanza.

        ya_en_hilo=True: la corrutina ya salta sola al hilo de poke-env (p. ej. stop_listening()).
        """
        try:
            if not ya_en_hilo:
                coro = handle_threaded_coroutines(coro, jugador.ps_client.loop)
            await asyncio.wait_for(coro, self.limpieza_timeout_s)
        except Exception as e:                         # websocket ya muerto, servidor caido, etc.
            self.log(f"[arena] limpieza de {jugador.username}: {type(e).__name__}: {e}")

    async def _limpiar_jugadores(self, *jugadores) -> int:
        """Rinde y abandona las batallas abiertas de los jugadores viejos y cierra su conexion.

        Devuelve cuantas batallas abiertas se rindieron. Es 'mejor esfuerzo': si el websocket
        ya esta muerto, Showdown igual cierra la sesion cuando se corta la conexion.
        """
        rendidas = 0
        for j in jugadores:
            # --- 1) /forfeit + /leave en cada batalla que no termino ---
            abiertas = [tag for tag, b in list(j.battles.items()) if not b.finished]
            for tag in abiertas:
                await self._en_poke_loop(j.ps_client.send_message("/forfeit", room=tag), j)
                await self._en_poke_loop(j.ps_client.send_message(f"/leave {tag}"), j)
            rendidas += len(abiertas)
        for j in jugadores:
            # --- 2) cerrar el websocket: el listener sale de su bucle ---
            await self._en_poke_loop(j.ps_client.stop_listening(), j, ya_en_hilo=True)
            # --- 3) cancelar lo que quede vivo en el hilo de poke-env (listener y tareas pendientes) ---
            cliente, loop = j.ps_client, j.ps_client.loop
            fut = getattr(cliente, "_listening_coroutine", None)
            if fut is not None and not fut.done():
                fut.cancel()
            for tarea in list(getattr(cliente, "_active_tasks", ())):
                loop.call_soon_threadsafe(tarea.cancel)
        return rendidas

    async def _login_ok(self) -> bool:
        """True si los dos jugadores ya entraron al servidor (espera hasta login_timeout_s)."""
        async def esperar(ev):
            await asyncio.wait_for(ev.wait(), self.login_timeout_s)
        try:
            for j in (self.pm, self.po):
                if not j.ps_client.logged_in.is_set():
                    await handle_threaded_coroutines(esperar(j.ps_client.logged_in), j.ps_client.loop)
            return True
        except (asyncio.TimeoutError, TimeoutError):
            return False

    async def _fallo(self, motivo: str) -> list[dict]:
        """Bloque descartado: limpiar jugadores, reiniciar Showdown si toca y crear jugadores nuevos."""
        self.n_timeouts += 1
        self.seguidos += 1
        rendidas = await self._limpiar_jugadores(self.pm, self.po)
        ev = {"t": datetime.now().isoformat(timespec="seconds"), "tipo": "fallo", "motivo": motivo,
              "seguidos": self.seguidos, "batallas_rendidas": rendidas}
        self.eventos.append(ev)
        self.log(f"[arena] bloque descartado ({motivo}); {rendidas} batallas abiertas rendidas; "
                 f"fallos seguidos: {self.seguidos}")
        # --- vigilante: N fallos seguidos => reiniciar Showdown (mismo puerto, mismas reglas) ---
        if self.reiniciar is not None and self.seguidos >= self.max_seguidos:
            self.log(f"[arena] {self.seguidos} fallos seguidos: se reinicia Showdown")
            info = self.reiniciar()           # bloqueante (<= ~90 s); si falla, la excepcion corta la corrida
            self.n_reinicios += 1
            self.seguidos = 0
            self.eventos.append({"t": datetime.now().isoformat(timespec="seconds"), "tipo": "reinicio_showdown",
                                 **(info or {})})
        self._crear_jugadores()
        return []

    def _podar(self) -> None:
        """Olvida batallas terminadas hace al menos un bloque (antes el dict crecia sin limite).

        Se espera un bloque entero para que ya haya llegado el |deinit| de esas batallas: si llegara
        un mensaje de una batalla olvidada, poke-env lo esperaria para siempre (_get_battle).
        """
        for j in (self.pm, self.po):
            for tag in self._terminadas_antes.get(j.username, ()):
                j.battles.pop(tag, None)
                j.chosen.pop(tag, None)
            self._terminadas_antes[j.username] = {t for t, b in j.battles.items() if b.finished}

    async def play(self, packed_m: str, packed_o: str, k: int,
                   chooser_m: Callable[[], int], chooser_o: Callable[[], int]) -> list[dict]:
        """Juega k batallas: metodo con packed_m contra el rival con packed_o. Devuelve una fila por batalla."""
        if self.fecha_limite is not None and time.perf_counter() > self.fecha_limite:
            raise TimeoutError("se paso el tiempo total permitido (total_timeout_s)")
        # jugadores recien creados (o websocket caido): si no entran al servidor, no se espera 600 s
        if not await self._login_ok():
            return await self._fallo(f"login > {self.login_timeout_s}s")
        self._podar()
        # equipos con naturaleza y stat points rellenados por la regla A8 (igual para ambos lados)
        self.pm.update_team(impute_packed(packed_m, self.a8))
        self.po.update_team(impute_packed(packed_o, self.a8))
        self.pm.chooser, self.po.chooser = chooser_m, chooser_o
        antes = set(self.pm.battles)
        t0 = time.perf_counter()
        try:
            await asyncio.wait_for(self.pm.battle_against(self.po, n_battles=k), self.timeout_s)
        except asyncio.TimeoutError:
            return await self._fallo(f"bloque colgado > {self.timeout_s}s")
        self.seguidos = 0
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
