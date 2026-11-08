"""Tests de la Fase 2: politica (90 brings legales), perdida (7) y v-hat (antisimetria)."""
import torch

from bringvgc.brings import BRINGS, bring_index, consistent_mask
from bringvgc.models.policy import BringPolicy, partial_label_loss
from bringvgc.models.vhat import VHat, uniform_q

SIZES = {"species": 20, "item": 15, "ability": 15, "move": 40}


def fake_batch(B=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    tok = lambda: torch.randint(1, 15, (B, 6, 7), generator=g)
    return {"tok_self": tok(), "feat_self": torch.rand(B, 6, 4, generator=g),
            "tok_opp": tok(), "feat_opp": torch.rand(B, 6, 4, generator=g)}


def test_policy_90_reachable_legal_and_normalised():
    torch.manual_seed(0)
    m = BringPolicy(SIZES).eval()
    lp = m.log_probs(fake_batch())
    assert lp.shape == (3, 90)
    assert torch.isfinite(lp).all()
    torch.testing.assert_close(lp.exp().sum(-1), torch.ones(3))
    assert ((lp.exp() > 0).sum(-1) == 90).all()


def test_policy_mask_removes_brings_with_unavailable_mon():
    torch.manual_seed(0)
    m = BringPolicy(SIZES).eval()
    mask = torch.ones(1, 6, dtype=torch.bool)
    mask[0, 4] = False
    p = m.log_probs(fake_batch(1), mask).exp()[0]
    legal = [k for k, (L, R) in enumerate(BRINGS) if 4 not in L + R]
    assert len(legal) == 30
    assert p[legal].sum() > 0.999
    assert int(m.greedy(fake_batch(1), mask)[0]) in legal


def test_partial_loss_is_ce_for_singletons_and_marginal_otherwise():
    torch.manual_seed(0)
    lp = torch.log_softmax(torch.randn(2, 90), -1)
    c1 = torch.as_tensor(consistent_mask([0, 1], [2, 3]))
    c6 = torch.as_tensor(consistent_mask([0, 1], []))
    k = bring_index([0, 1], [2, 3])
    loss = partial_label_loss(lp, torch.stack([c1, c6]))
    torch.testing.assert_close(loss[0], -lp[0, k])
    torch.testing.assert_close(loss[1], -torch.logsumexp(lp[1][c6], 0))


def test_vhat_antisymmetry_exact():
    torch.manual_seed(0)
    v = VHat(SIZES).eval()
    b = fake_batch(2)
    V12 = v.v_matrix(b["tok_self"], b["tok_opp"])
    V21 = v.v_matrix(b["tok_opp"], b["tok_self"])
    torch.testing.assert_close(V12 + V21.transpose(1, 2), torch.ones_like(V12))
    q = torch.full((2, 90), 1 / 90)
    torch.testing.assert_close(v.marginal(b["tok_self"], b["tok_opp"], q, q)
                               + v.marginal(b["tok_opp"], b["tok_self"], q, q), torch.ones(2))


def test_vhat_order_invariance_within_pairs():
    torch.manual_seed(0)
    v = VHat(SIZES).eval()
    b = fake_batch(1)
    perm = [1, 0, 3, 2, 5, 4]   # cambia el orden dentro de pares de la hoja
    f = v.f_all(b["tok_self"])
    fp = v.f_all(b["tok_self"][:, perm])
    for k, (L, R) in enumerate(BRINGS):
        kp = bring_index([perm.index(i) for i in L], [perm.index(i) for i in R])
        torch.testing.assert_close(f[0, k], fp[0, kp])


def test_vhat_leads_and_reserves_are_distinguished():
    torch.manual_seed(0)
    v = VHat(SIZES).eval()
    f = v.f_all(fake_batch(1)["tok_self"])[0]
    swapped = bring_index([2, 3], [0, 1])
    assert not torch.allclose(f[bring_index([0, 1], [2, 3])], f[swapped])


def test_uniform_q():
    m = torch.as_tensor(consistent_mask([0, 1], [2])).unsqueeze(0)
    q = uniform_q(m)
    torch.testing.assert_close(q.sum(-1), torch.ones(1))
    assert (q[m] - 1 / 3).abs().max() < 1e-6
