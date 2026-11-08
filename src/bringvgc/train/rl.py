"""Agente RL: REINFORCE + baseline de media del batch + entropia + KL a rho_BC (plan v2 §7.5).

BORRADOR asistido por IA; responsable sugerido: Jose (bucle de entrenamiento).

L(theta) = -E[(R - mean_batch R) log pi(b|s)] - beta_H H[pi] + beta_KL KL(pi || rho_BC)

- Episodio = una batalla real en Showdown (H=2: lideres y reservas, gamma=1).
- Sin replay buffer ni red objetivo: on-policy con retorno de Monte Carlo exacto.
- La politica se usa en modo eval (sin dropout) para muestrear y para el gradiente,
  asi la distribucion con que se muestrea es la misma que se optimiza.
- Pares de train con lado aleatorio; un bloque = un par y una orientacion con
  `block` batallas concurrentes (los episodios de un bloque comparten estado).
- Features de uso con las tasas completas de train (sin LOO: aqui no hay etiqueta
  del log que se pueda filtrar; la recompensa sale de la batalla).
- Cada `eval_every` episodios: eval greedy en val contra Usage y checkpoint.
  El checkpoint final se elige por win rate en val (nunca test).
Uso: python -m bringvgc.train.rl --config configs/rl.yaml --seed 0 [--resume results/rl_.../]
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from ..baselines.methods import PolicyMethod, UsageMethod
from ..config import Run, load_config, resolve, set_seed
from ..data.dataset import load_artifacts
from ..data.splits import load_pairs
from ..features.encode import encode_state, stack
from ..parsing.showdown import parse_packed_team
from ..sim.arena import Arena, evaluate_pairs
from .bc import load_policy


def _tb(states: list[dict]) -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v) for k, v in stack(states).items()}


class RLTrainer:
    def __init__(self, cfg: dict, seed: int, resume: str | None = None):
        self.cfg, self.seed = cfg, seed
        set_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.vocab, self.usage = load_artifacts()
        self.policy = load_policy(resolve(cfg["bc_ckpt"]))
        self.ref = load_policy(resolve(cfg["bc_ckpt"]))
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=cfg["lr"])
        self.opponent = UsageMethod(self.usage)
        self.train_pairs = load_pairs("train", resolve("splits"))
        self.val_pairs = load_pairs("val", resolve("splits"))[: cfg.get("val_pairs", 10)]
        # estados precalculados: (par, orientacion)
        self.slots = []
        for pid, p in enumerate(self.train_pairs):
            for o, (pm, po) in (("A", (p["packed_A"], p["packed_B"])), ("B", (p["packed_B"], p["packed_A"]))):
                tm, to = parse_packed_team(pm), parse_packed_team(po)
                self.slots.append({"pid": pid, "orient": o, "pm": pm, "po": po,
                                   "state": encode_state(tm, to, self.vocab, self.usage),
                                   "opp_p": self.opponent.probs(to, tm)})
        self.episode, self.step, self.best_val, self.curves = 0, 0, -1.0, []
        self.train_wall = 0.0
        if resume:
            self.run_dir = Path(resume)
            self._load(self.run_dir / "ckpt_latest.pt")
            self.meta = json.loads((self.run_dir / "meta.json").read_text(encoding="utf-8"))
            self.run = None
        else:
            self.run = Run("rl", cfg, seed)
            self.run_dir = self.run.dir
        self.arena = Arena(cfg["concurrency"], cfg["auto_mega"], cfg["a8_rule"], cfg["max_turns"],
                           cfg["timeout_s"])

    # ---------- checkpoints ----------
    def _save(self, name: str):
        torch.save({"state_dict": self.policy.state_dict(), "opt": self.opt.state_dict(),
                    "sizes": {c: self.vocab.size(c) for c in self.vocab.maps},
                    "cfg": {"d": self.policy.enc.d, "dropout": 0.1, **self.cfg},
                    "episode": self.episode, "step": self.step, "best_val": self.best_val,
                    "curves": self.curves, "train_wall": self.train_wall,
                    "np_rng": self.rng.bit_generator.state, "py_rng": random.getstate(),
                    "torch_rng": torch.get_rng_state(), "seed": self.seed},
                   self.run_dir / name)

    def _load(self, path: Path):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.policy.load_state_dict(ck["state_dict"])
        self.opt.load_state_dict(ck["opt"])
        self.episode, self.step, self.best_val = ck["episode"], ck["step"], ck["best_val"]
        self.curves, self.train_wall = ck["curves"], ck["train_wall"]
        self.rng.bit_generator.state = ck["np_rng"]
        random.setstate(ck["py_rng"])
        torch.set_rng_state(ck["torch_rng"])
        print(f"reanudando desde episodio {self.episode}")

    # ---------- entrenamiento ----------
    async def collect(self) -> list[dict]:
        cfg, batch = self.cfg, []
        self.policy.eval()
        while len(batch) < cfg["batch_size"]:
            si = int(self.rng.integers(len(self.slots)))
            s = self.slots[si]
            with torch.no_grad():
                p = self.policy.log_probs(_tb([s["state"]]))[0].exp().double().numpy()
            p /= p.sum()
            rows = await self.arena.play(
                s["pm"], s["po"], cfg["block"],
                lambda: int(self.rng.choice(90, p=p)),
                lambda: int(self.rng.choice(90, p=s["opp_p"])))
            for r in rows:
                r.update(slot=si, pair_id=s["pid"], orientation=s["orient"])
            batch += rows
        return batch

    def update(self, batch: list[dict]) -> dict:
        cfg = self.cfg
        states = _tb([self.slots[r["slot"]]["state"] for r in batch])
        acts = torch.as_tensor([r["bring_m"] for r in batch])
        R = torch.as_tensor([float(r["reward"]) for r in batch])
        lp_all = self.policy.log_probs(states)
        lp = lp_all.gather(1, acts[:, None]).squeeze(1)
        adv = R - R.mean()
        p = lp_all.exp()
        H = -(p * lp_all).sum(-1).mean()
        with torch.no_grad():
            ref_lp = self.ref.log_probs(states)
        KL = (p * (lp_all - ref_lp)).sum(-1).mean()
        loss = -(adv * lp).mean() - cfg["beta_H"] * H + cfg["beta_KL"] * KL
        self.opt.zero_grad()
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.policy.parameters(), cfg["grad_clip"])
        self.opt.step()
        self.step += 1
        won = [r for r in batch if r["reward"] == 1]
        lost = [r for r in batch if r["reward"] == -1]
        return {"loss": loss.item(), "entropy": H.item(), "kl_to_bc": KL.item(), "grad_norm": float(gn),
                "mean_reward": float(R.mean()), "win_rate_batch": float((R == 1).float().mean()),
                "draws": int((R == 0).sum()),
                "turns_won": float(np.mean([r["turns"] for r in won])) if won else float("nan"),
                "turns_lost": float(np.mean([r["turns"] for r in lost])) if lost else float("nan")}

    async def eval_val(self) -> dict:
        method = PolicyMethod(self.policy, self.vocab, self.usage, name="agent", greedy=True)
        t0 = time.perf_counter()
        rows = await evaluate_pairs(self.arena, method, self.opponent, self.val_pairs,
                                    self.cfg["val_n_per_pair"], self.rng, greedy=True,
                                    block=self.cfg["block"])
        wr = [np.mean([r["won"] for r in rows if r["pair_id"] == i]) for i in range(len(self.val_pairs))]
        return {"val_win_rate": float(np.mean(wr)), "val_sd_pairs": float(np.std(wr, ddof=1)),
                "val_battles": len(rows), "val_wall_s": round(time.perf_counter() - t0, 1),
                "val_distinct_brings": len({(r["pair_id"], r["orientation"], r["bring_m"]) for r in rows})}

    async def run_loop(self, max_wall_min: float | None = None):
        cfg = self.cfg
        t_start = time.perf_counter()
        ep_file = open(self.run_dir / "episodes.csv", "a", newline="", encoding="utf-8")
        ep_w = csv.writer(ep_file)
        if self.episode == 0:
            ep_w.writerow(["episode", "pair_id", "orientation", "bring", "bring_opp", "reward", "turns"])
            ev = await self.eval_val()
            self.curves.append({"episode": 0, "step": 0, **ev})
            print("eval inicial (rho_BC):", ev)
        next_eval = (self.episode // cfg["eval_every"] + 1) * cfg["eval_every"]
        stop_reason = "budget"
        while self.episode < cfg["episodes"]:
            if max_wall_min and (time.perf_counter() - t_start) / 60 > max_wall_min:
                stop_reason = "wall_clock_cap"
                break
            t0 = time.perf_counter()
            batch = await self.collect()
            info = self.update(batch)
            dt = time.perf_counter() - t0
            self.train_wall += dt
            for r in batch:
                self.episode += 1
                ep_w.writerow([self.episode, r["pair_id"], r["orientation"], r["bring_m"], r["bring_o"],
                               r["reward"], r["turns"]])
            row = {"episode": self.episode, "step": self.step, **info,
                   "T_train_eps_per_s": round(len(batch) / dt, 3)}
            if self.episode >= next_eval:
                row.update(await self.eval_val())
                next_eval += cfg["eval_every"]
                if row["val_win_rate"] > self.best_val:
                    self.best_val = row["val_win_rate"]
                    self._save("ckpt_best_val.pt")
                self._save("ckpt_latest.pt")
                ep_file.flush()
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
            self.curves.append(row)
        self._save("ckpt_latest.pt")
        ep_file.close()
        keys = sorted({k for c in self.curves for k in c}, key=lambda k: (k not in ("episode", "step"), k))
        with open(self.run_dir / "curves.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(self.curves)
        T_train = self.episode / self.train_wall if self.train_wall else 0
        extra = dict(episodes=self.episode, updates=self.step, stop_reason=stop_reason,
                     best_val_win_rate=self.best_val, T_train_eps_per_s=round(T_train, 3),
                     T_arena_battles_per_s=round(self.arena.throughput, 3),
                     arena_timeouts=self.arena.n_timeouts, train_wall_s=round(self.train_wall, 1),
                     note="Piloto (smoke test, 1 semilla, no citable)")
        if self.run:
            self.run.save_meta(**extra)
        else:
            self.meta.update(extra, resumed=True)
            (self.run_dir / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        print(extra)
        return extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/rl.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume")
    ap.add_argument("--episodes", type=int)
    ap.add_argument("--max-wall-min", type=float)
    ap.add_argument("--beta-H", type=float)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.episodes:
        cfg["episodes"] = a.episodes
    if a.beta_H is not None:
        cfg["beta_H"] = a.beta_H
    tr = RLTrainer(cfg, a.seed, a.resume)
    asyncio.run(tr.run_loop(a.max_wall_min))


if __name__ == "__main__":
    main()
