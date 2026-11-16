"""Arena robusta (13-11-2026): limpieza al colgarse un bloque, vigilante que reinicia Showdown
y reinicio que nunca mata un proceso ajeno.
No necesitan Showdown: los jugadores son falsos (no abren websockets) y subprocess esta simulado.
Los mensajes de limpieza corren en el hilo real de poke-env (POKE_LOOP), como en una corrida de verdad.
"""
import asyncio
from types import SimpleNamespace

import pytest
from poke_env.concurrency import POKE_LOOP

import sim.arena as arena_mod
import sim.showdown as sd


# ---------------------------------------------------------------------------
# Jugadores falsos: registran lo que la arena les manda, sin red
# ---------------------------------------------------------------------------
class ClienteFalso:
    def __init__(self, registro, nombre, logueado=True):
        self.loop = POKE_LOOP
        self.registro, self.nombre = registro, nombre
        self.logged_in = asyncio.Event()
        if logueado:
            self.logged_in.set()
        self._listening_coroutine = SimpleNamespace(done=lambda: False, cancel=lambda: registro.append((nombre, "cancel_listener")))
        self._active_tasks = set()

    async def send_message(self, message, room=""):
        self.registro.append((self.nombre, "msg", room, message))

    async def stop_listening(self):
        self.registro.append((self.nombre, "stop_listening"))


class JugadorFalso:
    n_tags = 0                                 # contador global de batallas falsas

    def __init__(self, registro, nombre, colgado=True, logueado=True):
        self.username, self.colgado = nombre, colgado
        self.ps_client = ClienteFalso(registro, nombre, logueado)
        self.battles, self.chosen, self.chooser = {}, {}, None

    def update_team(self, team):
        pass

    async def battle_against(self, otro, n_battles):
        if self.colgado:
            await asyncio.sleep(3600)          # bloque colgado: nunca termina
        for i in range(n_battles):             # bloque normal: k batallas terminadas (tags unicos)
            JugadorFalso.n_tags += 1
            tag = f"battle-ok-{JugadorFalso.n_tags}"
            self.battles[tag] = SimpleNamespace(finished=True, turn=5, won=True, lost=False)


@pytest.fixture
def arena_falsa(monkeypatch):
    """Arena cuyos jugadores son falsos; cada _crear_jugadores crea un par nuevo (bm<n>, bo<n>)."""
    registro, creados = [], []
    monkeypatch.setattr(arena_mod, "impute_packed", lambda packed, regla: packed)

    def crear(self):
        n = len(creados)
        colgado = getattr(self, "_proximos_colgados", True)
        logueado = getattr(self, "_proximos_logueados", True)
        self.pm = JugadorFalso(registro, f"bm{n}", colgado, logueado)
        self.po = JugadorFalso(registro, f"bo{n}", colgado, logueado)
        self._terminadas_antes = {}
        creados.append((self.pm, self.po))
    monkeypatch.setattr(arena_mod.Arena, "_crear_jugadores", crear)
    reinicios = []
    a = arena_mod.Arena(concurrency=4, timeout_s=0.2, login_timeout_s=0.2, limpieza_timeout_s=2,
                        log=lambda *x: None, reiniciar=lambda: reinicios.append(1) or {"pid_nuevo": 99},
                        max_timeouts_seguidos=2)
    return a, registro, creados, reinicios


# ---------------------------------------------------------------------------
# 1) Bloque colgado: se rinden las batallas abiertas, se cierra el websocket y se crean jugadores nuevos
# ---------------------------------------------------------------------------
def test_bloque_colgado_limpia_jugadores_viejos(arena_falsa):
    a, registro, creados, reinicios = arena_falsa
    viejo_m, viejo_o = a.pm, a.po
    viejo_m.battles["battle-x-1"] = SimpleNamespace(finished=False)     # abierta: hay que rendirla
    viejo_m.battles["battle-x-2"] = SimpleNamespace(finished=True)      # terminada: no se toca
    filas = asyncio.run(a.play("A", "B", 4, lambda: 0, lambda: 0))
    assert filas == []
    assert a.n_timeouts == 1 and a.seguidos == 1 and reinicios == []
    assert ("bm0", "msg", "battle-x-1", "/forfeit") in registro
    assert ("bm0", "msg", "", "/leave battle-x-1") in registro
    assert not any(r[2:] == ("battle-x-2", "/forfeit") for r in registro if r[1] == "msg")
    for nombre in ("bm0", "bo0"):                                       # los dos websockets viejos se cierran
        assert (nombre, "stop_listening") in registro
        assert (nombre, "cancel_listener") in registro
    assert a.pm is not viejo_m and a.po is not viejo_o and len(creados) == 2
    assert a.eventos[-1]["tipo"] == "fallo" and a.eventos[-1]["batallas_rendidas"] == 1


# ---------------------------------------------------------------------------
# 2) Vigilante: al 2.o fallo seguido se reinicia Showdown; un bloque bueno pone el contador en 0
# ---------------------------------------------------------------------------
def test_vigilante_reinicia_tras_n_fallos_seguidos(arena_falsa):
    a, registro, creados, reinicios = arena_falsa
    asyncio.run(a.play("A", "B", 4, lambda: 0, lambda: 0))
    assert reinicios == []
    a._proximos_colgados = False                  # los jugadores que se creen despues ya no se cuelgan
    asyncio.run(a.play("A", "B", 4, lambda: 0, lambda: 0))
    assert reinicios == [1] and a.seguidos == 0 and a.n_reinicios == 1
    assert [e["tipo"] for e in a.eventos] == ["fallo", "fallo", "reinicio_showdown"]
    assert a.eventos[-1]["pid_nuevo"] == 99
    filas = asyncio.run(a.play("A", "B", 3, lambda: 7, lambda: 8))
    assert len(filas) == 3 and a.seguidos == 0 and a.n_timeouts == 2


def test_login_fallido_cuenta_como_fallo_sin_esperar_el_bloque(arena_falsa):
    a, registro, creados, reinicios = arena_falsa
    a.pm.ps_client.logged_in.clear()              # el websocket no llego a entrar al servidor
    filas = asyncio.run(a.play("A", "B", 4, lambda: 0, lambda: 0))
    assert filas == [] and a.n_timeouts == 1
    assert a.eventos[-1]["motivo"].startswith("login")


def test_poda_olvida_batallas_terminadas_un_bloque_despues(arena_falsa):
    a, registro, creados, reinicios = arena_falsa
    a.pm.colgado = a.po.colgado = False
    asyncio.run(a.play("A", "B", 2, lambda: 0, lambda: 0))
    primeras = set(a.pm.battles)
    asyncio.run(a.play("A", "B", 2, lambda: 0, lambda: 0))
    assert primeras <= set(a.pm.battles)          # siguen: solo paso un bloque
    asyncio.run(a.play("A", "B", 2, lambda: 0, lambda: 0))
    assert not (primeras & set(a.pm.battles))     # ya se olvidaron


# ---------------------------------------------------------------------------
# 3) reiniciar_showdown: solo mata el pid que arranco este proceso
# ---------------------------------------------------------------------------
class PopenFalso:
    def __init__(self, pid):
        self.pid, self.vivo = pid, True

    def poll(self):
        return None if self.vivo else 0

    def wait(self, timeout=None):
        self.vivo = False


@pytest.fixture
def taskkills(monkeypatch):
    """Registra los taskkill/kill en vez de ejecutarlos."""
    llamadas = []

    def run_falso(cmd, **kw):
        llamadas.append(cmd)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(sd.subprocess, "run", run_falso)
    monkeypatch.setattr(sd.os, "killpg", lambda *a: llamadas.append(["killpg", *a]), raising=False)
    monkeypatch.setattr(sd.os, "getpgid", lambda pid: pid, raising=False)
    monkeypatch.setattr(sd, "LOG_ACTUAL", None)
    return llamadas


def test_reinicio_sin_proceso_propio_no_mata_nada(monkeypatch, taskkills):
    monkeypatch.setattr(sd, "PROCESO_ACTUAL", None)
    monkeypatch.setattr(sd, "puerto_abierto", lambda p, host="localhost": True)   # otro tiene el puerto
    with pytest.raises(RuntimeError, match="no arranco ningun Showdown"):
        sd.reiniciar_showdown(8000)
    assert taskkills == []


def test_reinicio_no_toca_un_proceso_ajeno_en_el_puerto(monkeypatch, taskkills):
    propio = PopenFalso(12345)
    monkeypatch.setattr(sd, "PROCESO_ACTUAL", propio)
    monkeypatch.setattr(sd, "puerto_abierto", lambda p, host="localhost": True)   # sigue ocupado: es ajeno
    monkeypatch.setattr(sd, "_arrancar", lambda *a: pytest.fail("no debe arrancar con el puerto ocupado"))
    with pytest.raises(RuntimeError, match="otro proceso"):
        sd.reiniciar_showdown(8000, espera_puerto_s=0.6)
    pids = [c[c.index("/PID") + 1] if "/PID" in c else str(c[1]) for c in taskkills]
    assert pids and all(p == "12345" for p in pids)              # solo el propio, nunca otro pid


def test_reinicio_mata_el_propio_y_arranca_otro(monkeypatch, taskkills):
    propio, nuevo = PopenFalso(111), PopenFalso(222)
    monkeypatch.setattr(sd, "PROCESO_ACTUAL", propio)
    monkeypatch.setattr(sd, "puerto_abierto", lambda p, host="localhost": propio.vivo)  # se libera al morir

    def arrancar(puerto, timeout_s):
        sd.PROCESO_ACTUAL, sd.RUTA_LOG_ACTUAL = nuevo, SimpleNamespace(name="showdown_server_X.log")
        return nuevo
    monkeypatch.setattr(sd, "_arrancar", arrancar)
    info = sd.reiniciar_showdown(8000)
    assert info["pid_viejo"] == 111 and info["pid_nuevo"] == 222 and sd.PROCESO_ACTUAL is nuevo
    assert all("111" in [str(x) for x in c] for c in taskkills)  # solo se mato el 111
    assert not any("222" in [str(x) for x in c] for c in taskkills)
