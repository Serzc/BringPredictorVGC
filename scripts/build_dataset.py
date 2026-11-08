"""Fase 1: parsea los logs, particiona por serie, congela pares, usage stats y vocab.

BORRADOR asistido por IA.
Uso: python scripts/build_dataset.py --config configs/data.yaml
Salidas: splits/ (versionado), data/processed/ (no versionado), results/dataset/.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from bringvgc.config import Run, load_config, resolve  # noqa: E402
from bringvgc.data.splits import (assign_partitions, compute_cutoffs, save_pairs,  # noqa: E402
                                  select_pairs, write_battle_split)
from bringvgc.data.usage import UsageStats  # noqa: E402
from bringvgc.data.vocab import Vocab  # noqa: E402
from bringvgc.parsing.showdown import parse_battle  # noqa: E402
from bringvgc.sim.validate import validate_packed  # noqa: E402


def iso(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    logs = Path(os.environ.get("BRINGVGC_LOGS") or resolve(cfg["logs"]))
    if not logs.exists():
        raise SystemExit(f"No se encuentra el log set: {logs} (usar BRINGVGC_LOGS)")

    data = json.loads(logs.read_text(encoding="utf-8"))
    records = [parse_battle(k, v[0], v[1]) for k, v in data.items()]
    n_ok = sum(r["valid"] for r in records)
    rate = n_ok / len(records)
    reasons = Counter(r["reason"].split(":")[0] for r in records if not r["valid"])
    print(f"parse: {n_ok}/{len(records)} validos ({rate:.4%}); excluidos: {dict(reasons)}")
    if rate < cfg["min_parse_success"]:
        raise SystemExit(f"Tasa de parseo {rate:.4f} < {cfg['min_parse_success']}: revisar el parser")

    if cfg.get("cut_val") is None or cfg.get("cut_test") is None:
        cv, ct = compute_cutoffs(records)
        print(f"Cortes calculados: cut_val={cv} ({iso(cv)}), cut_test={ct} ({iso(ct)}). "
              "Fijarlos en configs/data.yaml y volver a correr.")
        return
    part = assign_partitions(records, cfg["cut_val"], cfg["cut_test"])
    run = Run("dataset", cfg, root=cfg["results_dir"].split("/")[0])

    splits_dir, proc = resolve(cfg["splits_dir"]), resolve(cfg["processed_dir"])
    splits_dir.mkdir(exist_ok=True)
    proc.mkdir(parents=True, exist_ok=True)
    write_battle_split(records, part, splits_dir / "battles.csv")
    for r in records:
        r["partition"] = part[r["battle_id"]]

    valid = [r for r in records if r["valid"]]
    by = {p: [r for r in valid if r["partition"] == p] for p in ("train", "val", "test")}
    usage = UsageStats.fit(by["train"], prior=cfg["usage_prior"])
    usage.save(proc / "usage.json")
    vocab = Vocab.build([t for r in by["train"] for t in r["teams"]])
    vocab.save(proc / "vocab.json")
    with open(proc / "records.pkl", "wb") as f:
        pickle.dump(records, f)

    # Equipos legales segun el validador del Showdown fijado (con la imputacion A8)
    packed_all = sorted({p for r in valid for p in r["packed"]})
    verdicts = validate_packed(packed_all, cfg.get("a8_rule", "role_spread"))
    legal = {p for p, v in zip(packed_all, verdicts) if v == "OK"}
    illegal_battles = [r["battle_id"] for r in valid
                       if r["packed"][0] not in legal or r["packed"][1] not in legal]
    print(f"validador: {len(legal)}/{len(packed_all)} equipos legales; "
          f"{len(illegal_battles)} batallas con algun equipo ilegal (no se usan como pares)")
    pc = cfg["pairs"]
    pairs, report = select_pairs(records, part, pc["n_train"], pc["n_val"], pc["n_test"], pc["seed"],
                                 legal=legal)
    report["validator"] = {"teams_checked": len(packed_all), "teams_legal": len(legal),
                           "battles_with_illegal_team": len(illegal_battles),
                           "battles_with_illegal_team_ids": illegal_battles}
    save_pairs(pairs, splits_dir)
    (splits_dir / "pairs_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")

    # Tabla de estadisticas (F6)
    out = resolve(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p, rs in by.items():
        ts = [r["ts"] for r in rs]
        obs = [o for r in rs for o in r["obs"]]
        vc = Counter(o["v"] for o in obs)
        units = {r["series_id"] or r["battle_id"] for r in rs}
        rows.append({
            "particion": p, "batallas": len(rs), "unidades": len(units),
            "series_bo3": len({r["series_id"] for r in rs if r["series_id"]}),
            "player_brings": len(obs), "desde": iso(min(ts)), "hasta": iso(max(ts)),
            "v2_%": round(100 * vc[2] / len(obs), 1), "v1_%": round(100 * vc[1] / len(obs), 1),
            "v0_%": round(100 * vc[0] / len(obs), 1),
            "mediana_turnos": sorted(r["turns"] for r in rs)[len(rs) // 2],
            "cobertura_vocab": vocab.coverage([t for r in rs for t in r["teams"]]) if p != "train" else "-",
        })
    # revelacion vs resultado (se reporta; nunca se usa como feature)
    fw = fl = nw = nl = 0
    for r in valid:
        if r["winner"] is None:
            continue
        for s in (0, 1):
            full = r["obs"][s]["v"] == 2
            if s == r["winner"]:
                nw, fw = nw + 1, fw + full
            else:
                nl, fl = nl + 1, fl + full
    stats = {"parse_success_rate": round(rate, 6), "n_battles": len(records), "n_valid": n_ok,
             "excluded": dict(reasons),
             "excluded_ids": [r["battle_id"] for r in records if not r["valid"]],
             "illusion_unresolved_battles": sum(any(r["illusion_unresolved"]) for r in valid),
             "ties": sum(r["winner"] is None for r in valid),
             "full_reveal_rate_winners": round(fw / nw, 4), "full_reveal_rate_losers": round(fl / nl, 4),
             "cut_val": iso(cfg["cut_val"]), "cut_test": iso(cfg["cut_test"]),
             "vocab_sizes": {c: vocab.size(c) for c in vocab.maps},
             "partitions": rows, "pairs_report": report}
    (out / "dataset_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False),
                                            encoding="utf-8")
    cols = list(rows[0])
    md = ["# Estadisticas del dataset (generado por scripts/build_dataset.py)", "",
          f"Parseo: {n_ok}/{len(records)} validos ({rate:.2%}). Excluidos: {dict(reasons)}.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    md += ["", f"Revelacion completa: ganadores {fw / nw:.1%}, perdedores {fl / nl:.1%} "
           "(nunca se usa como feature).", "",
           "Pares congelados (ver splits/pairs_report.json):", "```",
           json.dumps(report, indent=1), "```"]
    (out / "dataset_stats.md").write_text("\n".join(md), encoding="utf-8")
    run.save_meta(stats_file=str(out / "dataset_stats.json"))
    print("\n".join(md[:8 + len(rows)]))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
