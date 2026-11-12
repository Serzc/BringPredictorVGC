"""Paso 'data': lee los logs, limpia, particiona por serie y calcula usage stats y vocabulario.

BORRADOR asistido por IA. Se llama con: python src/main.py data   (o dentro de 'all').
Entradas: src/dataset/raw/<logs>.json (lo descarga 'setup').
Salidas:
  src/dataset/clean/   records.pkl, usage.json, vocab.json          (no versionado, se regenera)
  src/dataset/splits/  battles.csv y pares congelados               (versionado; NO se regeneran)
  src/output/dataset/  dataset_stats.json y dataset_stats.md         (tabla de estadisticas)
"""
from __future__ import annotations

import json
import pickle
from collections import Counter
from datetime import datetime, timezone

import pandas as pd

from config.config import CLEAN_DIR, OUTPUT_DIR, RAW_DIR, SPLITS_DIR, Run
from dataPrep.parser import parse_battle
from dataPrep.particion import (assign_partitions, read_battle_split, save_pairs, select_pairs,
                                write_battle_split)
from utils.estadistica import tabla_markdown
from utils.usage import UsageStats
from utils.vocab import Vocab

PAIR_FILES = [SPLITS_DIR / f"{n}_pairs.json" for n in ("train", "val", "test")]


def iso(ts) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def construir(cfg: dict, smoke: bool = False) -> dict:
    run = Run("dataset", cfg, smoke=smoke)
    logs = RAW_DIR / cfg["logs_file"]
    if not logs.exists():
        raise SystemExit(f"No esta el log set en {logs}. Correr primero: python src/main.py setup")

    # --- 1) Parseo de todas las batallas ---
    data = json.loads(logs.read_text(encoding="utf-8"))
    records = [parse_battle(k, v[0], v[1]) for k, v in data.items()]
    n_ok = sum(r["valid"] for r in records)
    tasa = n_ok / len(records)
    motivos = Counter(r["reason"].split(":")[0] for r in records if not r["valid"])
    print(f"parseo: {n_ok}/{len(records)} validas ({tasa:.4%}); excluidas: {dict(motivos)}")
    if tasa < cfg["min_parse_success"]:
        raise SystemExit(f"Tasa de parseo {tasa:.4f} < {cfg['min_parse_success']}: revisar el parser")

    # --- 2) Particion temporal por serie Bo3 con los cortes fijos ---
    part = assign_partitions(records, cfg["cut_val"], cfg["cut_test"])
    battles_csv = SPLITS_DIR / "battles.csv"
    if battles_csv.exists():
        # la particion versionada manda: se comprueba que el parser actual da lo mismo
        congelada = read_battle_split(battles_csv)
        distintas = [b for b in part if congelada.get(b) != part[b]]
        if distintas:
            raise SystemExit(f"{len(distintas)} batallas cambian de particion respecto de splits/battles.csv")
    else:
        write_battle_split(records, part, battles_csv)
    for r in records:
        r["partition"] = part[r["battle_id"]]

    # --- 3) Usage stats y vocabulario: SOLO con train ---
    validas = [r for r in records if r["valid"]]
    por_part = {p: [r for r in validas if r["partition"] == p] for p in ("train", "val", "test")}
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    usage = UsageStats.fit(por_part["train"], prior=cfg["usage_prior"])
    usage.save(CLEAN_DIR / "usage.json")
    vocab = Vocab.build([t for r in por_part["train"] for t in r["teams"]])
    vocab.save(CLEAN_DIR / "vocab.json")
    with open(CLEAN_DIR / "records.pkl", "wb") as f:
        pickle.dump(records, f)

    # --- 4) Pares congelados: se reutilizan; solo se eligen si faltan ---
    if all(p.exists() for p in PAIR_FILES):
        print("pares congelados encontrados en src/dataset/splits/: no se regeneran")
        report = json.loads((SPLITS_DIR / "pairs_report.json").read_text(encoding="utf-8"))
    else:
        from dataPrep.validar import validate_packed
        packed_all = sorted({p for r in validas for p in r["packed"]})
        veredictos = validate_packed(packed_all, cfg["a8_rule"])
        legales = {p for p, v in zip(packed_all, veredictos) if v == "OK"}
        pc = cfg["pairs"]
        pairs, report = select_pairs(records, part, pc["n_train"], pc["n_val"], pc["n_test"], pc["seed"],
                                     legal=legales)
        save_pairs(pairs, SPLITS_DIR)
        (SPLITS_DIR / "pairs_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    # --- 5) Tabla de estadisticas por particion ---
    filas = []
    for p, rs in por_part.items():
        obs = [o for r in rs for o in r["obs"]]
        vc = Counter(o["v"] for o in obs)
        filas.append({
            "particion": p, "batallas": len(rs),
            "unidades": len({r["series_id"] or r["battle_id"] for r in rs}),
            "series_bo3": len({r["series_id"] for r in rs if r["series_id"]}),
            "player_brings": len(obs), "desde": iso(min(r["ts"] for r in rs)), "hasta": iso(max(r["ts"] for r in rs)),
            "v2_%": round(100 * vc[2] / len(obs), 1), "v1_%": round(100 * vc[1] / len(obs), 1),
            "v0_%": round(100 * vc[0] / len(obs), 1),
            "mediana_turnos": int(pd.Series([r["turns"] for r in rs]).median()),
        })
    tabla = pd.DataFrame(filas)

    # revelacion completa vs resultado: se reporta, NUNCA se usa como feature
    lados = pd.DataFrame([{"gano": int(s == r["winner"]), "completo": int(r["obs"][s]["v"] == 2)}
                          for r in validas if r["winner"] is not None for s in (0, 1)])
    rev = lados.groupby("gano")["completo"].mean()
    stats = {"parse_success_rate": round(tasa, 6), "n_battles": len(records), "n_valid": n_ok,
             "excluded": dict(motivos), "excluded_ids": [r["battle_id"] for r in records if not r["valid"]],
             "illusion_unresolved_battles": sum(any(r["illusion_unresolved"]) for r in validas),
             "ties": sum(r["winner"] is None for r in validas),
             "full_reveal_rate_winners": round(float(rev.get(1, float("nan"))), 4),
             "full_reveal_rate_losers": round(float(rev.get(0, float("nan"))), 4),
             "vocab_sizes": {c: vocab.size(c) for c in vocab.maps},
             "vocab_coverage": {p: vocab.coverage([t for r in por_part[p] for t in r["teams"]]) for p in ("val", "test")},
             "partitions": filas, "pairs_report": {k: v for k, v in report.items() if k != "validator"}}
    out = OUTPUT_DIR / "dataset"
    out.mkdir(parents=True, exist_ok=True)
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    md = ["# Estadisticas del dataset (generado por src/dataPrep/construir.py)", "",
          f"Parseo: {n_ok}/{len(records)} validas ({tasa:.2%}). Excluidas: {dict(motivos)}.", "",
          tabla_markdown(tabla), "",
          f"Revelacion completa: ganadores {stats['full_reveal_rate_winners']:.1%}, "
          f"perdedores {stats['full_reveal_rate_losers']:.1%} (nunca se usa como feature).", ""]
    (out / "dataset_stats.md").write_text("\n".join(md), encoding="utf-8")
    run.save_meta(stats_file=str(out / "dataset_stats.json"), n_valid=n_ok)
    print("\n".join(md))
    return stats
