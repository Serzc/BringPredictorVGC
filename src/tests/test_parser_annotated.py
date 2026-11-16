"""Compara el parser con las anotaciones manuales de 20 replays (plan v2 §6, F5)."""
import json
from pathlib import Path

import pytest

from dataPrep.cargar import clean_available, load_records
from utils.brings import bring_index, consistent_brings

FIX = Path(__file__).parent / "fixtures" / "annotated_replays.json"
CASES = json.loads(FIX.read_text(encoding="utf-8"))["replays"]


def _indice(team, nombre):
    """Posicion de 'nombre' en el equipo, tolerando como se escribio a mano.

    Primero busca el nombre exacto (sin espacios sobrantes). Si no esta, compara la
    especie base (lo que va antes del primer '-'): asi 'Garchomp-Mega', 'Maushold-Three',
    'Basculegion-M' o 'Meowstic-M' cuentan como el Pokemon de la hoja del equipo.
    Ditto e Illusion no necesitan regla: se anota el Pokemon real que entro.
    """
    nombre = nombre.strip()
    if nombre in team:
        return team.index(nombre)
    base = nombre.split("-")[0].lower()
    candidatos = [i for i, t in enumerate(team) if t.split("-")[0].lower() == base]
    assert len(candidatos) == 1, f"'{nombre}' no esta en el equipo {team}"
    return candidatos[0]


@pytest.mark.parametrize("case", CASES, ids=[c["battle_id"] for c in CASES])
def test_parser_matches_hand_annotation(case):
    if not clean_available():
        pytest.skip("falta src/dataset/clean")
    rec = next(r for r in load_records("train") if r["battle_id"] == case["battle_id"])
    for side, s in enumerate(("p1", "p2")):
        team = [m["species_name"] for m in rec["teams"][side]]
        # sorted(): el orden dentro de cada lista no importa
        L = sorted(_indice(team, x) for x in case[s]["leads"])
        V = sorted(_indice(team, x) for x in case[s]["revealed"])
        assert sorted(rec["obs"][side]["leads"]) == L, f"{s} lideres"
        assert sorted(rec["obs"][side]["revealed"]) == V, f"{s} reservas"
        # C(o) siempre contiene el bring verdadero cuando la anotacion lo da completo
        if len(V) == 2:
            assert bring_index(L, V) in consistent_brings(rec["obs"][side]["leads"],
                                                          rec["obs"][side]["revealed"])
