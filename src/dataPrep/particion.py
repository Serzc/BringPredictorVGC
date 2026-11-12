"""Particion temporal por serie Bo3 (A7) y pares congelados (plan v2 §6).

BORRADOR asistido por IA. Los pares de src/dataset/splits/ ya estan congelados: select_pairs()
solo se usa si faltan (nunca se regeneran a proposito; un test compara su SHA-256).

Unidad = serie Bo3 (o batalla suelta). La serie se asigna por el timestamp de
su primer juego; nunca se parte una serie entre particiones.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from config.config import SPLITS_DIR
from dataPrep.parser import team_key

PARTITIONS = ("train", "val", "test")


def unit_of(rec: dict) -> str:
    return rec["series_id"] or rec["battle_id"]


def compute_cutoffs(records: list[dict], fracs=(0.70, 0.85)) -> tuple[int, int]:
    """Cortes (timestamps) por cuantiles de batallas, ordenando unidades por su primer juego."""
    first, size = {}, Counter()
    for r in records:
        u = unit_of(r)
        first[u] = min(first.get(u, r["ts"]), r["ts"])
        size[u] += 1
    units = sorted(first, key=lambda u: first[u])
    total, acc, cuts = sum(size.values()), 0, []
    targets = list(fracs)
    for u in units:
        acc += size[u]
        while targets and acc / total >= targets[0]:
            cuts.append(first[u] + 1)
            targets.pop(0)
    return cuts[0], cuts[1]


def assign_partitions(records: list[dict], cut_val: int, cut_test: int) -> dict[str, str]:
    """battle_id -> particion. La unidad entera hereda la particion de su primer juego."""
    first = {}
    for r in records:
        u = unit_of(r)
        first[u] = min(first.get(u, r["ts"]), r["ts"])
    out = {}
    for r in records:
        t = first[unit_of(r)]
        out[r["battle_id"]] = "train" if t < cut_val else ("val" if t < cut_test else "test")
    return out


def write_battle_split(records, part, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["battle_id", "unit", "ts", "partition", "valid"])
        for r in sorted(records, key=lambda r: (r["ts"], r["battle_id"])):
            w.writerow([r["battle_id"], unit_of(r), r["ts"], part[r["battle_id"]], int(r["valid"])])


def read_battle_split(path: Path) -> dict[str, str]:
    with open(path, encoding="utf-8") as f:
        return {row["battle_id"]: row["partition"] for row in csv.DictReader(f)}


def _pair_entry(r: dict) -> dict:
    return {"battle_id": r["battle_id"], "packed_A": r["packed"][0], "packed_B": r["packed"][1],
            "key_A": team_key(r["teams"][0]), "key_B": team_key(r["teams"][1])}


def select_pairs(records: list[dict], part: dict[str, str], n_train: int, n_val: int, n_test: int,
                 seed: int, legal: set[str] | None = None) -> tuple[dict, dict]:
    """Pares de equipos para simular. Val y test: ningun equipo visto en train
    (test ademas sin equipos de val). Si faltan, se relaja a 'par no visto' (R5).
    `legal`: equipos packed que pasan el validador de Showdown; los pares con un
    equipo ilegal no se eligen (se colgarian en la arena)."""
    rng = random.Random(seed)
    # registros validos por particion (solo pares con los dos equipos legales)
    by = defaultdict(list)
    for r in records:
        if r["valid"] and (legal is None or (r["packed"][0] in legal and r["packed"][1] in legal)):
            by[part[r["battle_id"]]].append(r)
    # El filtro de leakage usa TODOS los registros de train (tambien los de equipos ilegales)
    all_train = [r for r in records if r["valid"] and part[r["battle_id"]] == "train"]
    train_keys = {team_key(t) for r in all_train for t in r["teams"]}
    train_pairs_seen = {frozenset((team_key(r["teams"][0]), team_key(r["teams"][1])))
                        for r in all_train}
    report = {}

    # elige n pares de un pool sin equipos vistos (banned_keys); si no alcanzan, relaja a "par no visto"
    def pick(pool, n, banned_keys, name):
        cand, seen_pairs = [], set()
        counts = Counter()
        for r in pool:
            ka, kb = team_key(r["teams"][0]), team_key(r["teams"][1])
            pk = frozenset((ka, kb))
            if ka == kb or pk in seen_pairs:
                continue
            seen_pairs.add(pk)
            n_seen = (ka in banned_keys) + (kb in banned_keys)
            counts[n_seen] += 1
            cand.append((n_seen, pk in train_pairs_seen, r))
        strict = [c[2] for c in cand if c[0] == 0]
        rng.shuffle(strict)
        chosen = strict[:n]
        relaxed = False
        if len(chosen) < n:  # R5: relajar a "par no visto"
            relaxed = True
            extra = [c[2] for c in sorted(cand, key=lambda c: c[0]) if c[0] > 0 and not c[1]]
            chosen += extra[: n - len(chosen)]
        report[name] = {"candidate_pairs": len(cand), "pairs_with_0_seen_teams": counts[0],
                        "pairs_with_1_seen_team": counts[1], "pairs_with_2_seen_teams": counts[2],
                        "chosen": len(chosen), "relaxed_to_unseen_pair": relaxed,
                        "chosen_seen_teams": [((team_key(r["teams"][0]) in banned_keys)
                                               + (team_key(r["teams"][1]) in banned_keys))
                                              for r in chosen]}
        return [_pair_entry(r) for r in chosen]

    # train: pares distintos al azar; val y test: con el filtro de equipos vistos
    pool_train = list(by["train"])
    rng.shuffle(pool_train)
    train_list, seen = [], set()
    for r in pool_train:
        pk = frozenset((team_key(r["teams"][0]), team_key(r["teams"][1])))
        if pk in seen or len(pk) < 2:
            continue
        seen.add(pk)
        train_list.append(_pair_entry(r))
        if len(train_list) >= n_train:
            break
    val_list = pick(sorted(by["val"], key=lambda r: r["battle_id"]), n_val, train_keys, "val")
    val_keys = {k for p in val_list for k in (p["key_A"], p["key_B"])}
    test_list = pick(sorted(by["test"], key=lambda r: r["battle_id"]), n_test,
                     train_keys | val_keys, "test")
    return {"train": train_list, "val": val_list, "test": test_list}, report


def save_pairs(pairs: dict, outdir: Path) -> None:
    for name, lst in pairs.items():
        (outdir / f"{name}_pairs.json").write_text(json.dumps(lst, indent=1), encoding="utf-8")


def load_pairs(name: str, splits_dir: Path = SPLITS_DIR, allow_test: bool = False) -> list[dict]:
    """Lee los pares congelados. Test esta bloqueado salvo en la corrida final (allow_test=True)."""
    if name == "test" and not allow_test:
        raise PermissionError("Test congelado: solo la corrida preregistrada (main.py test) puede "
                              "cargar test_pairs.json (allow_test=True).")
    return json.loads((splits_dir / f"{name}_pairs.json").read_text(encoding="utf-8"))
