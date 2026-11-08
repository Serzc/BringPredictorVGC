"""Evalua un metodo de bring contra un rival en pares congelados (plan v2 §8, §9).

BORRADOR asistido por IA.
Uso:
  python -m bringvgc.eval.run_arena --method usage --opponent usage --pairs val --n-per-pair 200
  python -m bringvgc.eval.run_arena --method bc --bc-ckpt results/bc_.../model.pt --pairs val
  python -m bringvgc.eval.run_arena --method usage_argmax --mirror --pairs val   # mismo equipo y bring en ambos lados
--pairs test exige --allow-test (solo la corrida preregistrada de la Fase 5).
Salidas: results/arena_<metodo>_vs_<rival>_<pares>_<fecha>_s<seed>/{battles.csv, per_pair.csv, meta.json}
"""
from __future__ import annotations

import argparse
import asyncio
import csv

import numpy as np

from ..config import Run, load_config, resolve, set_seed
from ..data.dataset import load_artifacts
from ..data.splits import load_pairs
from ..baselines.methods import build_method
from ..sim.arena import Arena, evaluate_pairs
from .stats import t_interval, wilson


def per_pair_table(rows: list[dict]) -> list[dict]:
    out = []
    for pid in sorted({r["pair_id"] for r in rows}):
        rs = [r for r in rows if r["pair_id"] == pid]
        w = sum(r["won"] for r in rs)
        p, lo, hi = wilson(w, len(rs))
        out.append({"pair_id": pid, "pair_battle": rs[0]["pair_battle"], "n": len(rs), "wins": w,
                    "draws": sum(r["reward"] == 0 for r in rs), "win_rate": round(p, 4),
                    "wilson_low": round(lo, 4), "wilson_high": round(hi, 4),
                    "distinct_brings_method": len({r["bring_m"] for r in rs}),
                    "median_turns": int(np.median([r["turns"] for r in rs]))})
    return out


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


async def amain(a):
    cfg = load_config(a.config)
    set_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    vocab, usage = load_artifacts()
    pairs = load_pairs(a.pairs, resolve("splits"), allow_test=a.allow_test)
    if a.max_pairs:
        pairs = pairs[: a.max_pairs]
    if a.mirror:
        pairs = [{**p, "packed_B": p["packed_A"], "key_B": p["key_A"]} for p in pairs]
    ck = dict(bc_ckpt=a.bc_ckpt or cfg.get("bc_ckpt"), vhat_ckpt=a.vhat_ckpt or cfg.get("vhat_ckpt"),
              agent_ckpt=a.agent_ckpt or cfg.get("agent_ckpt"))
    method = build_method(a.method, usage, vocab, **ck)
    opponent = build_method(a.method if a.mirror else a.opponent, usage, vocab, **ck)
    n = a.n_per_pair or cfg["n_per_pair_val"]
    tag = f"arena_{a.method}_vs_{opponent.name}{'_mirror' if a.mirror else ''}_{a.pairs}"
    run = Run(tag, {**cfg, **vars(a), "n_per_pair": n}, a.seed)
    arena = Arena(cfg["concurrency"], cfg["auto_mega"], cfg["a8_rule"], cfg["max_turns"], cfg["timeout_s"])
    rows = await evaluate_pairs(arena, method, opponent, pairs, n, rng, greedy=True, block=cfg["block"])
    for r in rows:
        r["seed"] = a.seed
    write_csv(run.dir / "battles.csv", rows)
    table = per_pair_table(rows)
    write_csv(run.dir / "per_pair.csv", table)
    m, lo, hi, sd = t_interval([t["win_rate"] for t in table]) if table else (float("nan"),) * 4
    summary = {"method": a.method, "opponent": opponent.name, "pairs": a.pairs, "n_pairs": len(table),
               "battles": len(rows), "mean_win_rate_over_pairs": m, "t95_low": lo, "t95_high": hi,
               "sd_over_pairs": sd, "draws": sum(r["reward"] == 0 for r in rows),
               "capped": sum(r["capped"] for r in rows), "timeouts": arena.n_timeouts,
               "T_battles_per_s": round(arena.throughput, 3)}
    run.save_meta(summary=summary)
    print(summary)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/arena.yaml")
    ap.add_argument("--method", required=True)
    ap.add_argument("--opponent", default="usage")
    ap.add_argument("--pairs", default="val", choices=["train", "val", "test"])
    ap.add_argument("--n-per-pair", type=int, default=None)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mirror", action="store_true")
    ap.add_argument("--allow-test", action="store_true")
    ap.add_argument("--bc-ckpt")
    ap.add_argument("--vhat-ckpt")
    ap.add_argument("--agent-ckpt")
    asyncio.run(amain(ap.parse_args()))


if __name__ == "__main__":
    main()
