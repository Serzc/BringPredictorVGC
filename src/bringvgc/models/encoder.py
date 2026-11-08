"""Codificador DeepSets (plan v2 §7.1). BORRADOR asistido por IA.

phi_x = MLP([E_especie(32), E_objeto(16), E_habilidad(16), media E_mov(16), m_x(4)]) -> 64
h_i   = MLP([phi_i, media(phi_propios), media(phi_rivales)]) -> 64

El plan pide concatenar el contexto rival; aqui se agrega tambien la media del
propio equipo (sinergia) porque la eleccion de reservas depende de los
companeros. Sigue siendo invariante a permutaciones dentro de cada equipo.
"""
from __future__ import annotations

import torch
from torch import nn


def mlp(i, h, o, dropout=0.0):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Dropout(dropout), nn.Linear(h, o))


class MonEmbed(nn.Module):
    def __init__(self, sizes: dict[str, int], d: int = 64, n_feat: int = 4, use_feats: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        self.sp = nn.Embedding(sizes["species"], 32, padding_idx=0)
        self.it = nn.Embedding(sizes["item"], 16, padding_idx=0)
        self.ab = nn.Embedding(sizes["ability"], 16, padding_idx=0)
        self.mv = nn.Embedding(sizes["move"], 16, padding_idx=0)
        self.use_feats = use_feats
        self.out = mlp(32 + 16 + 16 + 16 + (n_feat if use_feats else 0), d, d, dropout)

    def forward(self, tok: torch.Tensor, feat: torch.Tensor | None = None) -> torch.Tensor:
        """tok (B,6,7) int, feat (B,6,4) -> (B,6,d)"""
        mv = self.mv(tok[..., 3:])                          # (B,6,4,16)
        nz = (tok[..., 3:] > 0).float().unsqueeze(-1)
        mv = (mv * nz).sum(-2) / nz.sum(-2).clamp(min=1)
        parts = [self.sp(tok[..., 0]), self.it(tok[..., 1]), self.ab(tok[..., 2]), mv]
        if self.use_feats:
            parts.append(feat)
        return self.out(torch.cat(parts, -1))


class DeepSetsEncoder(nn.Module):
    def __init__(self, sizes, d: int = 64, dropout: float = 0.1):
        super().__init__()
        self.embed = MonEmbed(sizes, d, dropout=dropout)
        self.mix = mlp(3 * d, d, d, dropout)
        self.d = d

    def forward(self, tok_self, feat_self, tok_opp, feat_opp) -> torch.Tensor:
        ps = self.embed(tok_self, feat_self)
        po = self.embed(tok_opp, feat_opp)
        ctx_s = ps.mean(1, keepdim=True).expand_as(ps)
        ctx_o = po.mean(1, keepdim=True).expand_as(ps)
        return self.mix(torch.cat([ps, ctx_s, ctx_o], -1))   # (B,6,d)
