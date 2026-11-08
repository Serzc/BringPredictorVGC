"""rho_BC: imitacion con etiquetas parciales, perdida (7) (plan v2 §7.3).

BORRADOR asistido por IA; responsable sugerido: Sergio.
Uso: python -m bringvgc.train.bc --config configs/bc.yaml --seed 0
Early stopping: NLL parcial en val con |C|=1 (se reportan tambien |C|=3 y |C|=6).
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from ..config import Run, load_config, set_seed
from ..data.dataset import load_artifacts, load_records
from ..models.policy import BringPolicy, partial_label_loss
from .common import batches, bc_examples, to_torch


def evaluate(model, data, bs=1024) -> dict:
    model.eval()
    nll, lead_ok, perm_ok = [], [], []
    with torch.no_grad():
        for idx in batches(len(data["csize"]), bs, None):
            b = to_torch(data, idx)
            lp = model.log_probs(b)
            nll.append(partial_label_loss(lp, b["cmask"]).numpy())
            g = model.greedy(b).numpy()
            perm_ok.append(data["cmask"][idx, g])
            lead_ok.append(g // 6 == np.array([_pair_id(l) for l in data["lead"][idx]]))
    nll, lead_ok, perm_ok = map(np.concatenate, (nll, lead_ok, perm_ok))
    out = {"nll_all": float(nll.mean()), "lead_pair_acc": float(lead_ok.mean()),
           "full_bring_permissive_acc": float(perm_ok.mean())}
    for c in (1, 3, 6):
        sel = data["csize"] == c
        out[f"nll_C{c}"] = float(nll[sel].mean()) if sel.any() else float("nan")
        out[f"n_C{c}"] = int(sel.sum())
    return out


def _pair_id(lead) -> int:
    from ..brings import LEAD_PAIRS
    return LEAD_PAIRS.index(tuple(sorted(lead)))


def train(cfg: dict, seed: int) -> tuple[BringPolicy, dict, Run]:
    set_seed(seed)
    run = Run("bc", cfg, seed)
    vocab, usage = load_artifacts()
    tr = bc_examples(load_records("train"), vocab, usage, loo=True)
    va = bc_examples(load_records("val"), vocab, usage, loo=False)
    sizes = {c: vocab.size(c) for c in vocab.maps}
    model = BringPolicy(sizes, cfg["d"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(seed)
    best, best_ep, bad, rows = float("inf"), -1, 0, []
    for ep in range(cfg["max_epochs"]):
        model.train()
        losses = []
        for idx in batches(len(tr["csize"]), cfg["batch_size"], rng):
            b = to_torch(tr, idx)
            loss = partial_label_loss(model.log_probs(b), b["cmask"]).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        m = evaluate(model, va)
        row = {"epoch": ep, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in m.items()}}
        rows.append(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        crit = m[cfg["early_stop_metric"]]
        if crit < best - 1e-4:
            best, best_ep, bad = crit, ep, 0
            torch.save({"state_dict": model.state_dict(), "sizes": sizes, "cfg": cfg, "seed": seed},
                       run.dir / "model.pt")
        else:
            bad += 1
            if bad >= cfg["patience"]:
                break
    with open(run.dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    best_row = rows[best_ep]
    run.save_meta(best_epoch=best_ep, best_val=best_row, n_train=len(tr["csize"]),
                  n_val=len(va["csize"]), note="Piloto (smoke test, 1 semilla, no citable)")
    return model, best_row, run


def load_policy(path) -> BringPolicy:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = BringPolicy(ck["sizes"], ck["cfg"]["d"], ck["cfg"]["dropout"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/bc.yaml")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    _, best, run = train(load_config(a.config), a.seed)
    print("mejor epoca (val):", best, "->", run.dir)


if __name__ == "__main__":
    main()
