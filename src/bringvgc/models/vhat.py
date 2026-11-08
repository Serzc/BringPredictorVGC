"""v-hat: evaluador de matchup bilineal antisimetrico, eqs. (8)-(11) del survey.

Adaptado de Oblander al nivel del bring (plan v2 §7.4). BORRADOR asistido por IA;
responsable sugerido: Jose.

h(b) = ReLU(beta_h + W_L sum_{x in L} phi_x + W_R sum_{x in R} phi_x)
f(b) = sigmoid(beta_z + W_z h(b))                     (K = 16)
v(b1,b2) = sigmoid(f(b1)^T M f(b2)),  M = A - A^T  ->  v(b1,b2) + v(b2,b1) = 1
Pr(w=1|o1,o2) = sum_{C1 x C2} q(b1) q(b2) v(b1,b2), q uniforme sobre cada C(o)   (11)

phi_x usa especie, movimientos, objeto y habilidad (sin features de uso, como el survey).
"""
from __future__ import annotations

import torch
from torch import nn

from ..brings import LEAD_MASK, RES_MASK
from .encoder import MonEmbed


class VHat(nn.Module):
    def __init__(self, sizes, d: int = 64, hidden: int = 64, K: int = 16, dropout: float = 0.1):
        super().__init__()
        self.embed = MonEmbed(sizes, d, use_feats=False, dropout=dropout)
        self.W_L = nn.Linear(d, hidden)              # incluye beta_h
        self.W_R = nn.Linear(d, hidden, bias=False)
        self.W_z = nn.Linear(hidden, K)              # incluye beta_z
        self.A = nn.Parameter(torch.randn(K, K) * 0.1)
        self.register_buffer("lead_m", torch.as_tensor(LEAD_MASK))   # (90,6)
        self.register_buffer("res_m", torch.as_tensor(RES_MASK))

    @property
    def M(self) -> torch.Tensor:
        return self.A - self.A.T

    def f_all(self, tok: torch.Tensor) -> torch.Tensor:
        """Embeddings f(b) de los 90 brings de un equipo: (B,90,K)."""
        phi = self.embed(tok)                                     # (B,6,d)
        sL = torch.einsum("bm,nmd->nbd", self.lead_m, phi)        # (B,90,d)
        sR = torch.einsum("bm,nmd->nbd", self.res_m, phi)
        h = torch.relu(self.W_L(sL) + self.W_R(sR))
        return torch.sigmoid(self.W_z(h))

    def score_matrix(self, f1, f2) -> torch.Tensor:
        """Logits bilineales (B,90,90): f1 M f2^T."""
        return torch.einsum("nik,kl,njl->nij", f1, self.M, f2)

    def v_matrix(self, tok1, tok2) -> torch.Tensor:
        return torch.sigmoid(self.score_matrix(self.f_all(tok1), self.f_all(tok2)))

    def marginal(self, tok1, tok2, q1, q2) -> torch.Tensor:
        """Pr(gana el lado 1) con q1,q2 (B,90) distribuciones sobre brings."""
        V = self.v_matrix(tok1, tok2)
        return torch.einsum("ni,nij,nj->n", q1, V, q2).clamp(1e-6, 1 - 1e-6)


def uniform_q(mask: torch.Tensor) -> torch.Tensor:
    m = mask.float()
    return m / m.sum(-1, keepdim=True)
