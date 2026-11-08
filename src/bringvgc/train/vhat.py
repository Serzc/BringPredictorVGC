"""Entrenamiento de v-hat con la verosimilitud marginalizada (11) (plan v2 §7.4).

BORRADOR asistido por IA; responsable sugerido: Jose (revisar y hacer commit).
Uso: python -m bringvgc.train.vhat --config configs/vhat.yaml --seed 0

Metricas en val (test solo en la Fase 5):
- team_*: prediccion a nivel de equipo, q uniforme sobre los 90 brings de cada lado.
  No usa nada de lo observado en la batalla -> sin fuga por revelacion. Es la
  comparable con Oblander (equipo vs equipo) y la que se usa para early stopping.
- obs_*: Pr(w|o1,o2) de (11) con los C(o) observados. Es el objetivo de
  entrenamiento, pero |C(o)| se correlaciona con el resultado (los ganadores
  revelan menos), asi que como metrica de evaluacion esta contaminada. Solo diagnostico.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
import torch

from ..config import Run, load_config, set_seed
from ..data.dataset import load_artifacts, load_records
from ..models.vhat import VHat, uniform_q
from .common import auc, batches, to_torch, vhat_examples


def evaluate(model, data, bs=256) -> dict:
    model.eval()
    p_team, p_obs = [], []
    with torch.no_grad():
        for idx in batches(len(data["y"]), bs, None):
            b = to_torch(data, idx)
            ones = torch.ones_like(b["m1"], dtype=torch.float32) / 90
            p_team.append(model.marginal(b["tok1"], b["tok2"], ones, ones).numpy())
            p_obs.append(model.marginal(b["tok1"], b["tok2"], uniform_q(b["m1"]),
                                        uniform_q(b["m2"])).numpy())
    y = data["y"]
    out = {}
    for name, p in (("team", np.concatenate(p_team)), ("obs", np.concatenate(p_obs))):
        out[f"{name}_nll"] = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        out[f"{name}_acc"] = float(np.mean((p > 0.5) == (y == 1)))
        out[f"{name}_auc"] = auc(y, p)
    return out


def train(cfg: dict, seed: int):
    set_seed(seed)
    run = Run("vhat", cfg, seed)
    vocab, usage = load_artifacts()
    tr = vhat_examples(load_records("train"), vocab, usage)
    va = vhat_examples(load_records("val"), vocab, usage)
    sizes = {c: vocab.size(c) for c in vocab.maps}
    model = VHat(sizes, cfg["d"], cfg["hidden"], cfg["K"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(seed)
    best, best_ep, bad, rows = float("inf"), -1, 0, []
    for ep in range(cfg["max_epochs"]):
        model.train()
        losses = []
        for idx in batches(len(tr["y"]), cfg["batch_size"], rng):
            b = to_torch(tr, idx)
            p = model.marginal(b["tok1"], b["tok2"], uniform_q(b["m1"]), uniform_q(b["m2"]))
            loss = torch.nn.functional.binary_cross_entropy(p, b["y"].float())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        m = evaluate(model, va)
        row = {"epoch": ep, "train_loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in m.items()}}
        rows.append(row)
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
        if m["team_nll"] < best - 1e-4:
            best, best_ep, bad = m["team_nll"], ep, 0
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
    run.save_meta(best_epoch=best_ep, best_val=rows[best_ep], n_train=len(tr["y"]), n_val=len(va["y"]),
                  val_win_rate_side1=float(va["y"].mean()),
                  note="Piloto (1 semilla; v-hat no es el modelo de la RQ). No citable.")
    return model, rows[best_ep], run


def load_vhat(path) -> VHat:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck["cfg"]
    m = VHat(ck["sizes"], c["d"], c["hidden"], c["K"], c["dropout"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/vhat.yaml")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    _, best, run = train(load_config(a.config), a.seed)
    print("mejor epoca (val):", best, "->", run.dir)


if __name__ == "__main__":
    main()
