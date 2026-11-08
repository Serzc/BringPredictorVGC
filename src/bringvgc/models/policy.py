"""Politica autorregresiva pi(b|s) = pi(a0|s) pi(a1|s,a0) (plan v2 §7.2).

Cabeza de lideres: score por par MLP(h_i + h_j), softmax sobre 15.
Cabeza de reservas: MLP([h_k + h_l, h_i + h_j]), softmax sobre los 6 pares restantes.
Mascara de legalidad por Pokemon (mon_mask): un par con un Pokemon no disponible
recibe -inf. Con los 6 disponibles hay exactamente 90 brings alcanzables.
"""
from __future__ import annotations

import torch
from torch import nn

from ..brings import LEAD_IDX, RES_IDX
from .encoder import DeepSetsEncoder, mlp

NEG = -1e9


class BringPolicy(nn.Module):
    def __init__(self, sizes, d: int = 64, dropout: float = 0.1):
        super().__init__()
        self.enc = DeepSetsEncoder(sizes, d, dropout)
        self.lead_head = mlp(d, d, 1, dropout)
        self.res_head = mlp(2 * d, d, 1, dropout)
        self.register_buffer("lead_idx", torch.as_tensor(LEAD_IDX))   # (15,2)
        self.register_buffer("res_idx", torch.as_tensor(RES_IDX))     # (15,6,2)

    def logits(self, batch: dict, mon_mask: torch.Tensor | None = None):
        h = self.enc(batch["tok_self"], batch["feat_self"], batch["tok_opp"], batch["feat_opp"])
        B = h.shape[0]
        pair_h = h[:, self.lead_idx].sum(2)                           # (B,15,d)
        lead = self.lead_head(pair_h).squeeze(-1)                     # (B,15)
        res_h = h[:, self.res_idx.view(-1, 2)].sum(2).view(B, 15, 6, -1)
        cond = pair_h.unsqueeze(2).expand_as(res_h)
        res = self.res_head(torch.cat([res_h, cond], -1)).squeeze(-1)  # (B,15,6)
        if mon_mask is not None:
            ok = mon_mask.bool()
            lead = lead.masked_fill(~ok[:, self.lead_idx].all(-1), NEG)
            ok_r = ok[:, self.res_idx.view(-1, 2)].all(-1).view(B, 15, 6)
            res = res.masked_fill(~ok_r, NEG)
        return lead, res

    def log_probs(self, batch, mon_mask=None) -> torch.Tensor:
        """log pi(b|s) para los 90 brings: (B,90)."""
        lead, res = self.logits(batch, mon_mask)
        lp = torch.log_softmax(lead, -1).unsqueeze(-1) + torch.log_softmax(res, -1)
        return lp.reshape(lead.shape[0], 90)

    @torch.no_grad()
    def greedy(self, batch, mon_mask=None) -> torch.Tensor:
        """Argmax autorregresivo: mejor par lider y luego mejores reservas dado ese par."""
        lead, res = self.logits(batch, mon_mask)
        p = lead.argmax(-1)
        r = res[torch.arange(len(p)), p].argmax(-1)
        return p * 6 + r


def partial_label_loss(logp: torch.Tensor, cmask: torch.Tensor) -> torch.Tensor:
    """Perdida (7): -log sum_{b in C(o)} pi(b|s), por ejemplo (B,)."""
    return -torch.logsumexp(logp.masked_fill(~cmask, NEG), -1)
