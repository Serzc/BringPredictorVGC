"""Tests de los modelos numpy: politica (90 brings legales), perdida (7), v-hat (antisimetria) y la clase padre."""
import numpy as np
import pytest

from baselines.planner import PlannerBring
from baselines.random import RandomBring
from baselines.tipos import TiposBring
from baselines.usage import UsageBring
from dataPrep.parser import parse_packed_team
from models.base import MetodoBring
from models.bc import PoliticaLineal, log_probs, greedy, perdida_parcial
from models.rl_agent import AgenteRL
from models.vhat import VHat, _torre_adelante, perdida_marginal
from tests.test_data import TEAM1, TEAM2
from utils.brings import BRINGS, bring_index, consistent_mask
from utils.features import Codificador
from utils.usage import UsageStats
from utils.vocab import Vocab


def codificador_falso():
    """Codificador con el vocabulario de los dos equipos sinteticos y sin datos de uso (prior puro)."""
    t1, t2 = parse_packed_team(TEAM1), parse_packed_team(TEAM2)
    return Codificador(Vocab.build([t1[:4], t2[:4]]), UsageStats({}, prior=5.0)), t1, t2


def test_policy_90_reachable_legal_and_normalised():
    cod, t1, t2 = codificador_falso()
    rng = np.random.default_rng(0)
    phi = np.stack([cod.mon_features(t1, t2), cod.mon_features(t2, t1)])
    lp = log_probs(rng.normal(size=cod.F), rng.normal(size=cod.F), phi)
    assert lp.shape == (2, 90) and np.isfinite(lp).all()
    np.testing.assert_allclose(np.exp(lp).sum(1), 1.0)
    assert ((np.exp(lp) > 0).sum(1) == 90).all()


def test_policy_mask_removes_brings_with_unavailable_mon():
    cod, t1, t2 = codificador_falso()
    rng = np.random.default_rng(1)
    phi = cod.mon_features(t1, t2)[None]
    mask = np.ones((1, 6), bool)
    mask[0, 4] = False
    tL, tR = rng.normal(size=cod.F), rng.normal(size=cod.F)
    p = np.exp(log_probs(tL, tR, phi, mask))[0]
    legal = [k for k, (L, R) in enumerate(BRINGS) if 4 not in L + R]
    assert len(legal) == 30
    assert p[legal].sum() > 0.999999
    assert int(greedy(tL, tR, phi, mask)[0]) in legal


def test_partial_loss_is_ce_for_singletons_and_marginal_otherwise():
    cod, t1, t2 = codificador_falso()
    rng = np.random.default_rng(2)
    phi = np.stack([cod.mon_features(t1, t2)] * 2)
    tL, tR = rng.normal(size=cod.F), rng.normal(size=cod.F)
    lp = log_probs(tL, tR, phi)
    c1, c6 = consistent_mask([0, 1], [2, 3]), consistent_mask([0, 1], [])
    _, por_ejemplo, _, _ = perdida_parcial(tL, tR, phi, np.stack([c1, c6]))
    k = bring_index([0, 1], [2, 3])
    assert por_ejemplo[0] == pytest.approx(-lp[0, k])
    assert por_ejemplo[1] == pytest.approx(-np.log(np.exp(lp[1][c6]).sum()))


# ---------------------------------------------------------------------------
# v-hat bilineal (eqs. 8-11 del survey): antisimetria, invarianzas, roles, <UNK>, checkpoint, interfaz del Planificador
# ---------------------------------------------------------------------------
def vhat_aleatorio(cod, seed=3, **kw):
    """VHat con parametros aleatorios y A grande (para que la antisimetria no sea trivial: v lejos de 0,5)."""
    vh = VHat(cod, seed=seed, **kw)
    vh.p["A"] = np.random.default_rng(seed).normal(0, 1.0, vh.p["A"].shape)
    return vh


def test_vhat_antisymmetry_exact():
    cod, t1, t2 = codificador_falso()
    vh = vhat_aleatorio(cod)
    V12, V21 = vh.v_matrix(t1, t2), vh.v_matrix(t2, t1)
    assert np.abs(V12 - 0.5).max() > 0.05                       # la prueba no es trivial
    np.testing.assert_allclose(V12 + V21.T, 1.0, atol=1e-12)    # v(b1,b2) + v(b2,b1) = 1 (eq:bilinear)
    # lo mismo para la verosimilitud marginalizada (11) con q uniforme sobre los 90: Pr(w=1) + Pr(w=2) = 1
    tok1, tok2 = vh.tokens(t1)[None], vh.tokens(t2)[None]
    q = np.full((1, 90), 1 / 90)
    y = np.array([1.0])
    p12 = perdida_marginal(vh.p, tok1, tok2, q, q, y, con_grad=False)[3]
    p21 = perdida_marginal(vh.p, tok2, tok1, q, q, y, con_grad=False)[3]
    np.testing.assert_allclose(p12 + p21, 1.0, atol=1e-12)


def test_vhat_order_invariance_within_pairs():
    """f(b) no cambia al reordenar los dos lideres ni las dos reservas (suma dentro de cada subconjunto)."""
    cod, t1, t2 = codificador_falso()
    vh = vhat_aleatorio(cod, seed=5)
    perm = [1, 0, 3, 2, 5, 4]                                    # intercambia el orden dentro de pares de la hoja
    f, _ = _torre_adelante(vh.p, vh.tokens(t1)[None])
    fp, _ = _torre_adelante(vh.p, vh.tokens([t1[i] for i in perm])[None])
    for k, (L, R) in enumerate(BRINGS):
        kp = bring_index([perm.index(i) for i in L], [perm.index(i) for i in R])
        np.testing.assert_allclose(f[0, k], fp[0, kp], atol=1e-12)
    # y dentro de un mismo bring: dar los lideres o las reservas en otro orden da el mismo indice de accion
    assert bring_index([1, 0], [3, 2]) == bring_index([0, 1], [2, 3])


def test_vhat_leads_and_reserves_are_distinguished_only_by_WL_neq_WR():
    cod, t1, t2 = codificador_falso()
    vh = vhat_aleatorio(cod, seed=4)
    f, _ = _torre_adelante(vh.p, vh.tokens(t1)[None])
    a, b = bring_index([0, 1], [2, 3]), bring_index([2, 3], [0, 1])    # mismos 4 Pokemon, roles intercambiados
    assert not np.allclose(f[0, a], f[0, b])                           # W_L != W_R: el rol importa
    vh.p["W_R"] = vh.p["W_L"].copy()                                   # con W_R = W_L los roles serian intercambiables
    f2, _ = _torre_adelante(vh.p, vh.tokens(t1)[None])
    np.testing.assert_allclose(f2[0, a], f2[0, b], atol=1e-12)


def test_vhat_unseen_tokens_map_to_unk_and_padding_to_zero():
    cod, t1, t2 = codificador_falso()                                  # el vocabulario solo vio los primeros 4 Pokemon de cada equipo
    vh = VHat(cod)
    t = [dict(m) for m in t1]
    t[0]["moves"] = t[0]["moves"][:2]                                  # un Pokemon con solo 2 movimientos
    tok = vh.tokens(t)
    assert tok.shape == (6, 7) and tok.dtype.kind == "i"
    assert (tok[4:, 0] == 1).all()                                     # especies no vistas en train -> <UNK> = 1
    assert (tok[0, 5:] == 0).all() and (tok[0, 3:5] > 0).all()         # movimientos faltantes -> <PAD> = 0
    assert vh.v_matrix(t, t2).shape == (90, 90)


def test_vhat_planner_interface_unchanged():
    cod, t1, t2 = codificador_falso()
    vh = vhat_aleatorio(cod, seed=6)
    plan = PlannerBring(vh, UsageBring(cod.usage))
    V = vh.v_matrix(t1, t2)
    assert V.shape == (90, 90) and ((V > 0) & (V < 1)).all()
    np.testing.assert_allclose(V + vh.v_matrix(t2, t1).T, 1.0, atol=1e-12)
    assert plan.expected(t1, t2).shape == (90,)
    p = plan.probs(t1, t2)
    assert p.shape == (90,) and p.sum() == pytest.approx(1) and int(np.argmax(p)) == int(np.argmax(plan.expected(t1, t2)))


def test_vhat_checkpoint_roundtrip_and_rejects_old_logistic_format(tmp_path):
    cod, t1, t2 = codificador_falso()
    vh = vhat_aleatorio(cod, seed=7, d=8, hidden=8, K=4)
    vh.guardar(tmp_path / "vhat_bilineal_s0.npz")
    otro = VHat.cargar(tmp_path / "vhat_bilineal_s0.npz", cod)
    np.testing.assert_array_equal(vh.v_matrix(t1, t2), otro.v_matrix(t1, t2))
    np.savez(tmp_path / "model.npz", w=np.zeros(5), nombres=np.array(["a"]))      # formato logistico viejo
    with pytest.raises(ValueError, match="logistico antiguo"):
        VHat.cargar(tmp_path / "model.npz", cod)


def test_all_methods_inherit_metodobring_and_return_distributions():
    cod, t1, t2 = codificador_falso()
    bc = PoliticaLineal(cod, name="bc")
    metodos = [RandomBring(), UsageBring(cod.usage), UsageBring(cod.usage, argmax=True), TiposBring(),
               PlannerBring(VHat(cod), UsageBring(cod.usage)), bc, AgenteRL(cod, bc, 0.01, 0.01, 0.1)]
    rng = np.random.default_rng(0)
    for m in metodos:
        assert isinstance(m, MetodoBring)
        p = m.probs(t1, t2)
        assert p.shape == (90,) and p.sum() == pytest.approx(1) and (p >= 0).all()
        assert 0 <= m.elegir(t1, t2, rng) < 90


def test_checkpoint_roundtrip(tmp_path):
    cod, t1, t2 = codificador_falso()
    rng = np.random.default_rng(5)
    pol = PoliticaLineal(cod, rng.normal(size=cod.F), rng.normal(size=cod.F))
    pol.guardar(tmp_path / "m.npz")
    otra = PoliticaLineal.cargar(tmp_path / "m.npz", cod)
    np.testing.assert_array_equal(pol.params, otra.params)
