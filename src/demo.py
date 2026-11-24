"""Demo: recomienda los 3 mejores brings con el agente RL entrenado y juega una batalla con el mejor.

Uso (desde la raiz del repo, con el .venv):
    .venv\\Scripts\\python src/demo.py demo/mi_equipo.txt demo/rival.txt

- Mi equipo: archivo con el texto de "Export" de Pokemon Showdown (6 Pokemon completos).
- Rival: archivo con los 6 nombres de especie, uno por linea (o separados por comas).
  El resto de cada Pokemon rival (objeto, habilidad, movimientos) se completa con el set
  mas comun de esa especie en las batallas de train. Naturaleza y stat points se completan
  con la regla de rol del proyecto (utils/spreads.py), igual que en todas las evaluaciones.
- Modelo: el checkpoint del agente RL de la semilla elegida (--semilla, 0 por defecto),
  el mismo que se evaluo en test.
- Simulacion: una batalla en el Showdown local (puerto 8000) con el mejor bring contra
  el rival eligiendo con Usage estocastico (el rival de todas las evaluaciones). Ambos
  lados juegan con la misma politica de batalla heuristica. El replay queda en HTML en
  demo/replays/ (se abre con el navegador).
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from poke_env import AccountConfiguration, ServerConfiguration   # noqa: E402
from poke_env.teambuilder import Teambuilder                     # noqa: E402

from baselines.usage import UsageBring                           # noqa: E402
from config.config import FORMAT, load_config                    # noqa: E402
from dataPrep.cargar import load_codificador, load_records      # noqa: E402
from dataPrep.parser import parse_packed_team, to_id            # noqa: E402
from models.bc import PoliticaLineal, greedy                     # noqa: E402
from sim.jugador import BringPlayer                              # noqa: E402
from sim.showdown import puerto_abierto, servidor_showdown       # noqa: E402
from utils.brings import BRINGS                                  # noqa: E402
from utils.spreads import MAX_PER_STAT, impute_packed            # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
CAMPOS_EVS = 6      # posicion de los stat points en el formato packed de Showdown


# ---------------------------------------------------------------------------
# Equipos
# ---------------------------------------------------------------------------
def leer_mi_equipo(ruta: Path) -> str:
    """Texto de Export de Showdown -> equipo packed (lo que entiende el servidor)."""
    mons = Teambuilder.parse_showdown_team(ruta.read_text(encoding="utf-8"))
    if len(mons) != 6:
        raise SystemExit(f"Mi equipo tiene {len(mons)} Pokemon; hacen falta 6.")
    packed = Teambuilder.join_team(mons)
    # El formato Champions usa stat points (maximo 32 por stat), no EVs clasicos (hasta 252).
    # Si el export trae EVs clasicos, se borran y se completan con la regla de rol.
    chunks = []
    for chunk in packed.split("]"):
        f = chunk.split("|")
        evs = [int(x) for x in f[CAMPOS_EVS].split(",") if x.strip().isdigit()] if len(f) > CAMPOS_EVS else []
        if any(x > MAX_PER_STAT for x in evs):
            print(f"[demo] {f[1] or f[0]}: EVs mayores a {MAX_PER_STAT}; se reemplazan por la regla de rol")
            f[CAMPOS_EVS] = ""
        chunks.append("|".join(f))
    return "]".join(chunks)


def sets_de_train() -> dict[str, list[tuple[str, str]]]:
    """Especie -> sets de train, del mas comun al menos comun: [(objeto, chunk packed real), ...]."""
    conteo: dict[str, Counter] = {}
    for rec in load_records("train"):
        for lado in (0, 1):
            for mon, chunk in zip(rec["teams"][lado], rec["packed"][lado].split("]")):
                clave = (mon["item"], mon["ability"], tuple(sorted(mon["moves"])))
                conteo.setdefault(mon["species"], Counter())[(clave, chunk)] += 1
    # por especie: sets ordenados por frecuencia; de cada set, un chunk packed real para copiarlo
    ranking = {}
    for especie, c in conteo.items():
        por_set = Counter()
        for (clave, _), n in c.items():
            por_set[clave] += n
        ranking[especie] = [(clave[0], next(ch for (cl, ch) in c if cl == clave)) for clave, _ in por_set.most_common()]
    return ranking


def completar_rival(ruta: Path) -> str:
    """Nombres de especie -> equipo packed completo con los sets mas comunes de train."""
    nombres = [n.strip() for linea in ruta.read_text(encoding="utf-8").splitlines() for n in linea.split(",")]
    nombres = [n for n in nombres if n]
    if len(nombres) != 6:
        raise SystemExit(f"El rival tiene {len(nombres)} nombres; hacen falta 6.")
    sets = sets_de_train()
    chunks, usados = [], set()
    for n in nombres:
        if to_id(n) not in sets:
            raise SystemExit(f"'{n}' no aparece en las batallas de train: no hay un set para completarlo. "
                             "Escribe ese Pokemon completo (export) o usa otra especie.")
        # Item Clause del formato: un objeto no se puede repetir. Se toma el set mas comun
        # cuyo objeto todavia no se uso.
        libres = [(item, ch) for item, ch in sets[to_id(n)] if not item or item not in usados]
        if not libres:
            raise SystemExit(f"'{n}': todos sus sets de train repiten un objeto ya usado (Item Clause).")
        item, ch = libres[0]
        usados.add(item)
        chunks.append(ch)
    return "]".join(chunks)


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def checkpoint_agente(semilla: int) -> Path:
    """El mismo checkpoint que se evaluo en test: corrida RL completa con la config de rl.yaml."""
    from main import corrida_hecha, ultimo_checkpoint
    from training.entrenar_rl import config_de_corrida
    cfg = load_config("rl")
    d = corrida_hecha("rl", semilla, False, config_de_corrida(cfg, ultimo_checkpoint("bc", semilla, "model.npz", False)))
    if d is None:
        raise SystemExit(f"No hay una corrida RL completa para la semilla {semilla} en src/output/.")
    return d / "ckpt_best_val.npz"


def nombre_bring(k: int, equipo: list[dict]) -> str:
    L, R = BRINGS[k]
    n = [m["species_name"] for m in equipo]
    return f"lideres {n[L[0]]} + {n[L[1]]} | reservas {n[R[0]]} + {n[R[1]]}"


# ---------------------------------------------------------------------------
# Batalla
# ---------------------------------------------------------------------------
async def jugar(packed_mio: str, packed_rival: str, bring_mio: int, bring_rival: int, carpeta: Path) -> dict:
    acfg = load_config("arena")
    server = ServerConfiguration(f"ws://localhost:{acfg['port']}/showdown/websocket",
                                 "https://play.pokemonshowdown.com/action.php?")
    tag = os.getpid() % 100000
    kw = dict(battle_format=FORMAT, max_concurrent_battles=1, auto_mega=acfg["auto_mega"],
              server_configuration=server)
    yo = BringPlayer(account_configuration=AccountConfiguration(f"Agente{tag}", None),
                     save_replays=str(carpeta), team=impute_packed(packed_mio, acfg["a8_rule"]), **kw)
    rival = BringPlayer(account_configuration=AccountConfiguration(f"Rival{tag}", None),
                        team=impute_packed(packed_rival, acfg["a8_rule"]), **kw)
    yo.chooser, rival.chooser = (lambda: bring_mio), (lambda: bring_rival)
    try:
        # una batalla dura segundos; si Showdown rechaza un equipo, nadie juega y se espera hasta aqui
        await asyncio.wait_for(yo.battle_against(rival, n_battles=1), 180)
    except asyncio.TimeoutError:
        raise SystemExit("La batalla no termino en 3 minutos: revisa arriba si Showdown rechazo un equipo "
                         "(mensaje 'Your team was rejected').")
    b = next(iter(yo.battles.values()))
    # cerrar las conexiones antes de apagar Showdown (si no, poke-env imprime errores de websocket)
    for j in (yo, rival):
        await j.ps_client.stop_listening()
    return {"tag": b.battle_tag, "gano": b.won, "turnos": b.turn}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mi_equipo", type=Path, help="export de Showdown con mis 6 Pokemon")
    ap.add_argument("rival", type=Path, help="6 nombres de especie del rival")
    ap.add_argument("--semilla", type=int, default=0, help="semilla del agente RL (0, 1 o 2)")
    ap.add_argument("--sin-batalla", action="store_true", help="solo recomendar, no simular")
    a = ap.parse_args()

    packed_mio, packed_rival = leer_mi_equipo(a.mi_equipo), completar_rival(a.rival)
    mio, rival = parse_packed_team(packed_mio), parse_packed_team(packed_rival)
    objetos = [m["item"] for m in mio if m["item"]]
    if len(objetos) != len(set(objetos)):
        raise SystemExit("Mi equipo repite un objeto: el formato no lo permite (Item Clause).")

    cod = load_codificador()
    ck = checkpoint_agente(a.semilla)
    agente = PoliticaLineal.cargar(ck, cod, name="agent")
    phi = agente.estado(mio, rival)
    p = agente.probs_desde_estado(phi)
    # 1 = lo que juega el agente evaluado en test: argmax autorregresivo (mejor par lider y, dado
    # ese par, mejores reservas). Puede no ser el bring de mayor probabilidad conjunta.
    elegido = int(greedy(agente.theta_L, agente.theta_R, phi[None])[0])
    top = [elegido] + [int(k) for k in np.argsort(-p) if k != elegido][:2]

    print(f"\nAgente RL, semilla {a.semilla} ({ck.parent.name})")
    print("Rival completado con los sets mas comunes de train:")
    for m in rival:
        print(f"  {m['species_name']:<18} @ {m['item'] or '-'} | {m['ability']} | {', '.join(m['moves'])}")
    print("\nMejores 3 brings (1 = el que elige el agente, como en test; 2 y 3 = los siguientes por probabilidad):")
    for i, k in enumerate(top, 1):
        print(f"  {i}. {nombre_bring(k, mio)}   (probabilidad {p[k]:.3f})")

    if a.sin_batalla:
        return
    if puerto_abierto(load_config("arena")["port"]):
        raise SystemExit("El puerto de Showdown ya esta ocupado: cierra el otro Showdown y vuelve a correr.")

    # el rival elige como en todas las evaluaciones: Usage estocastico
    rng = np.random.default_rng(a.semilla)
    p_rival = UsageBring(cod.usage).probs(rival, mio)
    bring_rival = int(rng.choice(90, p=p_rival))
    carpeta = RAIZ / "demo" / "replays"
    carpeta.mkdir(parents=True, exist_ok=True)
    acfg = load_config("arena")
    print(f"\nRival trae: {nombre_bring(bring_rival, rival)}")
    print("Jugando una batalla en el Showdown local...")
    with servidor_showdown(acfg["port"], acfg["startup_timeout_s"]):
        r = asyncio.run(jugar(packed_mio, packed_rival, int(top[0]), bring_rival, carpeta))
    resultado = "gano el agente" if r["gano"] else "gano el rival" if r["gano"] is False else "empate"
    replays = sorted(carpeta.glob("*.html"), key=lambda f: f.stat().st_mtime)
    print(f"\nResultado: {resultado} en {r['turnos']} turnos ({r['tag']})")
    print(f"Replay: {replays[-1] if replays else '(no se guardo)'}  -> abrir con el navegador")


if __name__ == "__main__":
    main()
