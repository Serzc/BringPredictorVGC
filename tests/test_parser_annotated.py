"""Test del parser contra 20 replays anotados A MANO (plan v2 §6, F5).

Estado: PENDIENTE. Las fixtures (tests/fixtures/annotated_replays.json) tienen
los battle_id elegidos pero los valores esperados estan en "TODO": los tienen
que llenar los estudiantes mirando cada replay. Nunca generar los valores
esperados con el parser. Mientras haya TODO, cada caso se marca como skip y el
parser NO se considera verificado.
"""
import json
from pathlib import Path

import pytest

from bringvgc.brings import consistent_brings, bring_index
from bringvgc.data.dataset import load_records, processed_available

FIX = Path(__file__).parent / "fixtures" / "annotated_replays.json"
CASES = json.loads(FIX.read_text(encoding="utf-8"))["replays"]


def _todo(case):
    return case["checked_by"] == "TODO" or any(
        case[s][k] == "TODO" for s in ("p1", "p2") for k in ("leads", "revealed"))


@pytest.mark.parametrize("case", CASES, ids=[c["battle_id"] for c in CASES])
def test_parser_matches_hand_annotation(case):
    if _todo(case):
        pytest.skip("anotacion a mano pendiente (estudiantes)")
    if not processed_available():
        pytest.skip("falta data/processed")
    rec = next(r for r in load_records("train") if r["battle_id"] == case["battle_id"])
    for side, s in enumerate(("p1", "p2")):
        team = [m["species_name"] for m in rec["teams"][side]]
        L = sorted(team.index(x) for x in case[s]["leads"])
        V = sorted(team.index(x) for x in case[s]["revealed"])
        assert rec["obs"][side]["leads"] == L
        assert rec["obs"][side]["revealed"] == V
        # C(o) siempre contiene el bring verdadero cuando la anotacion lo da completo
        if len(V) == 2:
            assert bring_index(L, V) in consistent_brings(rec["obs"][side]["leads"],
                                                          rec["obs"][side]["revealed"])
