"""Tests de la Fase 1: espacio de acciones, C(o), A8 y leakage (F3)."""
import csv
import json
from collections import defaultdict
from itertools import combinations

import numpy as np
import pytest

from bringvgc.brings import BRINGS, N_BRINGS, bring_index, bring_to_team_order, consistent_brings
from bringvgc.config import resolve
from bringvgc.data.dataset import load_records, processed_available
from bringvgc.data.splits import load_pairs
from bringvgc.data.usage import UsageStats, battle_contrib
from bringvgc.features.spreads import ROLE_SPREADS, impute_packed, infer_role, is_legal
from bringvgc.parsing.showdown import parse_battle, parse_packed_team, team_key

needs_data = pytest.mark.skipif(not processed_available(),
                                reason="falta data/processed (correr scripts/build_dataset.py)")


# ---------- espacio de acciones ----------
def test_exactly_90_distinct_legal_brings():
    assert N_BRINGS == 90
    seen = set()
    for L, R in BRINGS:
        assert len(set(L) | set(R)) == 4 and not set(L) & set(R)
        seen.add((frozenset(L), frozenset(R)))
    assert len(seen) == 90
    all_brings = {(frozenset(L), frozenset(R)) for L in combinations(range(6), 2)
                  for R in combinations([i for i in range(6) if i not in L], 2)}
    assert seen == all_brings


def test_bring_index_roundtrip_and_team_order():
    for k, (L, R) in enumerate(BRINGS):
        assert bring_index(L[::-1], R[::-1]) == k
        s = bring_to_team_order(k)
        assert len(s) == 4 and len(set(s)) == 4 and set(s) <= set("123456")


@pytest.mark.parametrize("v,size", [(2, 1), (1, 3), (0, 6)])
def test_consistent_sizes(v, size):
    revealed = [2, 3][:v]
    C = consistent_brings([0, 1], revealed)
    assert len(C) == size
    for k in C:
        L, R = BRINGS[k]
        assert set(L) == {0, 1} and set(revealed) <= set(R)


def test_consistent_rejects_invalid():
    with pytest.raises(ValueError):
        consistent_brings([0, 0], [])
    with pytest.raises(ValueError):
        consistent_brings([0, 1], [1])


# ---------- parser sobre un log sintetico ----------
TEAM1 = ("Zoroark||FocusSash|Illusion|NightDaze,Protect,FlameThrower,SludgeBomb||||||50|]"
         "Charizard||CharizarditeY|SolarPower|HeatWave,Protect,SolarBeam,WeatherBall||||||50|]"
         "Rotom-Wash||SitrusBerry|Levitate|HydroPump,VoltSwitch,Protect,WillOWisp||||||50|]"
         "Garchomp||ChoiceScarf|RoughSkin|Earthquake,RockSlide,DragonClaw,IronHead||||||50|]"
         "Incineroar||SitrusBerry|Intimidate|FakeOut,KnockOff,PartingShot,FlareBlitz||||||50|]"
         "Floette-Eternal||Floettite|FlowerVeil|Moonblast,Protect,LightofRuin,DazzlingGleam||||||50|")
TEAM2 = ("Kingambit||BlackGlasses|Defiant|KowtowCleave,SuckerPunch,Protect,IronHead||||||50|]"
         "Sneasler||FocusSash|Unburden|CloseCombat,DireClaw,FakeOut,Protect||||||50|]"
         "Basculegion||ChoiceScarf|SwiftSwim|WaveCrash,LastRespects,AquaJet,FlipTurn||||||50|]"
         "Amoonguss||RockyHelmet|Regenerator|Spore,RagePowder,Protect,PollenPuff||||||50|]"
         "Tyranitar||Tyranitarite|SandStream|RockSlide,KnockOff,LowKick,Protect||||||50|]"
         "Gholdengo||ChoiceSpecs|GoodasGold|MakeItRain,ShadowBall,NastyPlot,Protect||||||50|")


def synth_log(extra_p1="", winner="alice"):
    return "\n".join([
        "|player|p1|alice|1|", "|player|p2|bob|2|", f"|showteam|p1|{TEAM1}", f"|showteam|p2|{TEAM2}",
        "|teamsize|p1|4", "|teamsize|p2|4", "|start",
        "|switch|p1a: Zorua|Charizard, L50|100/100",          # Zoroark disfrazado
        "|switch|p1b: Rotom|Rotom-Wash, L50|100/100",
        "|switch|p2a: Kingambit|Kingambit, L50|100/100",
        "|switch|p2b: Sneasler|Sneasler, L50|100/100", "|turn|1",
        "|replace|p1a: Zoroark|Zoroark, L50",
        "|detailschange|p1b: Rotom|Rotom-Wash, L50",
        extra_p1,
        "|switch|p2a: Amoonguss|Amoonguss, L50|100/100", "|turn|2",
        f"|win|{winner}"])


def test_parser_illusion_forms_nicknames():
    log = synth_log("|switch|p1a: Floette|Floette-Mega, L50|100/100")
    r = parse_battle("x-1", 0, log)
    assert r["valid"], r["reason"]
    assert r["obs"][0]["leads"] == [0, 2]          # Zoroark (real) + Rotom-Wash
    assert r["obs"][0]["revealed"] == [5]          # Floette-Eternal via Floette-Mega
    assert r["obs"][1] == {"leads": [0, 1], "revealed": [3], "v": 1}
    assert r["winner"] == 0 and r["turns"] == 2
    assert r["illusion_unresolved"] == [False, False]


def test_parser_rejects_teamsize_5():
    log = synth_log().replace("|teamsize|p1|4", "|teamsize|p1|5")
    assert not parse_battle("x-2", 0, log)["valid"]


def test_team_key_ignores_order():
    t = parse_packed_team(TEAM1)
    assert team_key(t) == team_key(t[::-1])


# ---------- A8 ----------
def test_role_spreads_legal():
    for nature, spread in ROLE_SPREADS.values():
        assert is_legal(spread) and sum(spread) == 66


def test_infer_role():
    assert infer_role(["earthquake", "rockslide", "dragonclaw", "protect"]) == "physical"
    assert infer_role(["heatwave", "solarbeam", "protect", "weatherball"]) == "special"
    assert infer_role(["spore", "ragepowder", "protect", "trickroom"]) == "support"


@pytest.mark.parametrize("rule", ["role_spread", "random"])
def test_impute_fills_legal(rule):
    out = impute_packed(TEAM1, rule, seed=3)
    for chunk in out.split("]"):
        f = chunk.split("|")
        assert f[5] and f[6]
        assert is_legal([int(x) for x in f[6].split(",")])


def test_impute_same_rule_both_sides_is_deterministic():
    assert impute_packed(TEAM2, "role_spread", 0) == impute_packed(TEAM2, "role_spread", 99)


# ---------- usage LOO (sintetico) ----------
def test_loo_excludes_own_battle():
    r1 = parse_battle("b1", 0, synth_log())
    r2 = parse_battle("b2", 1, synth_log("|switch|p1a: Floette|Floette-Eternal, L50|100/100"))
    both = UsageStats.fit([r1, r2])
    only2 = UsageStats.fit([r2])
    ex = battle_contrib(r1)
    for sp in ex:
        np.testing.assert_allclose(both.features(sp, exclude=ex)[:3], only2.features(sp)[:3])


# ---------- leakage sobre los datos reales (F3) ----------
@needs_data
def test_no_series_split_across_partitions():
    units = defaultdict(set)
    with open(resolve("splits/battles.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            units[row["unit"]].add(row["partition"])
    assert all(len(p) == 1 for p in units.values())


@needs_data
def test_no_val_or_test_team_in_train():
    train_keys = {team_key(t) for r in load_records("train") for t in r["teams"]}
    val = load_pairs("val", resolve("splits"))
    test = load_pairs("test", resolve("splits"), allow_test=True)  # solo claves, sin modelos
    for p in val + test:
        assert p["key_A"] not in train_keys and p["key_B"] not in train_keys
    val_keys = {k for p in val for k in (p["key_A"], p["key_B"])}
    assert not val_keys & {k for p in test for k in (p["key_A"], p["key_B"])}


@needs_data
def test_train_pairs_come_from_train():
    train_ids = {r["battle_id"] for r in load_records("train")}
    assert all(p["battle_id"] in train_ids for p in load_pairs("train", resolve("splits")))


@needs_data
def test_usage_stats_from_train_ids_only():
    d = json.loads(resolve("data/processed/usage.json").read_text(encoding="utf-8"))
    train_ids = {r["battle_id"] for r in load_records("train")}
    assert set(d["train_ids"]) == train_ids
    with open(resolve("splits/battles.csv"), encoding="utf-8") as f:
        part = {row["battle_id"]: row["partition"] for row in csv.DictReader(f)}
    assert all(part[b] == "train" for b in d["train_ids"])


@needs_data
def test_test_partition_is_gated():
    with pytest.raises(PermissionError):
        load_records("test")
    with pytest.raises(PermissionError):
        load_pairs("test", resolve("splits"))
