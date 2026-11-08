"""Tablas multi-semilla a partir de results/arena_*/per_pair.csv (plan v2 §9.2).

BORRADOR asistido por IA.
Por metodo: win rate de cada par promediado sobre semillas -> media e intervalo t
sobre pares, desviacion estandar entre semillas (de la media sobre pares) y
diferencia pareada contra una referencia. Aplica la regla preregistrada del 55 %.
Uso: python -m bringvgc.eval.aggregate --pairs val [--reference usage_argmax] [--out results/tabla_val.md]
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from ..config import resolve
from .stats import decision, paired_difference


def collect(pairs: str, opponent: str = "usage") -> dict[str, dict[int, dict[int, float]]]:
    """metodo -> semilla -> {pair_id: win_rate}. Usa la corrida mas reciente por (metodo, semilla)."""
    latest = {}
    for d in sorted(resolve("results").glob(f"arena_*_vs_{opponent}_{pairs}_*")):
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if not (d / "per_pair.csv").exists() or "summary" not in meta:
            continue
        latest[(meta["summary"]["method"], meta["seed"])] = d
    out = defaultdict(dict)
    for (m, s), d in latest.items():
        with open(d / "per_pair.csv", encoding="utf-8") as f:
            out[m][s] = {int(r["pair_id"]): float(r["win_rate"]) for r in csv.DictReader(f)}
    return out


def table(pairs: str, reference: str | None, opponent: str = "usage") -> list[dict]:
    data = collect(pairs, opponent)
    rows = []
    ref = None
    if reference in data:
        ref_seeds = data[reference]
        ids = sorted(set.intersection(*[set(v) for v in ref_seeds.values()]))
        ref = {i: np.mean([ref_seeds[s][i] for s in ref_seeds]) for i in ids}
    for m, seeds in sorted(data.items()):
        ids = sorted(set.intersection(*[set(v) for v in seeds.values()]))
        per_pair = [np.mean([seeds[s][i] for s in seeds]) for i in ids]
        dec = decision(per_pair)
        seed_means = [np.mean([seeds[s][i] for i in ids]) for s in seeds]
        row = {"method": m, "seeds": len(seeds), "pairs": len(ids),
               "mean": round(dec["mean"], 4), "t95_low": round(dec["t_low"], 4),
               "t95_high": round(dec["t_high"], 4),
               "sd_seeds": round(float(np.std(seed_means, ddof=1)), 4) if len(seeds) > 1 else "n/a (1 corrida)",
               "regla_55": dec["verdict"]}
        if ref is not None and m != reference:
            common = [i for i in ids if i in ref]
            d, se = paired_difference([np.mean([seeds[s][i] for s in seeds]) for i in common],
                                      [ref[i] for i in common])
            row[f"diff_vs_{reference}"] = f"{d:+.4f} ± {se:.4f}"
        rows.append(row)
    return rows


def to_md(rows: list[dict]) -> str:
    cols = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="val")
    ap.add_argument("--opponent", default="usage")
    ap.add_argument("--reference", default="usage_argmax")
    ap.add_argument("--out")
    a = ap.parse_args()
    md = to_md(table(a.pairs, a.reference, a.opponent))
    print(md)
    if a.out:
        resolve(a.out).write_text(md + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
