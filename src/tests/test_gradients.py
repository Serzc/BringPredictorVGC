"""Gradientes a mano contra diferencias finitas centrales: BC (7), v-hat bilineal (11) y la actualizacion de RL.

Si alguno falla, el entrenamiento estaria optimizando otra cosa sin dar error. Se usan
float64, pesos aleatorios y una tolerancia relativa de 1e-5.
"""
import numpy as np

from models.bc import log_probs, perdida_parcial
from models.rl_agent import perdida_rl
from models.vhat import VHat, nombres_parametros, perdida_marginal
from tests.test_models import codificador_falso
from utils.brings import consistent_mask

EPS = 1e-6


def diferencias_finitas(f, x):
    """Gradiente numerico de f en x (central)."""
    g = np.zeros_like(x)
    for i in range(len(x)):
        d = np.zeros_like(x)
        d[i] = EPS
        g[i] = (f(x + d) - f(x - d)) / (2 * EPS)
    return g


def assert_close(analitico, numerico):
    err = np.linalg.norm(analitico - numerico) / max(1e-12, np.linalg.norm(analitico) + np.linalg.norm(numerico))
    assert err < 1e-5, f"error relativo {err:.2e}"


def estados(n=4, seed=0):
    cod, t1, t2 = codificador_falso()
    rng = np.random.default_rng(seed)
    base = np.stack([cod.mon_features(t1, t2), cod.mon_features(t2, t1)])
    phi = base[rng.integers(0, 2, n)] + rng.normal(0, 0.1, (n, 6, cod.F))    # estados distintos entre si
    return cod, phi, rng


def test_bc_partial_label_gradient():
    cod, phi, rng = estados()
    F = cod.F
    cmask = np.stack([consistent_mask([0, 1], [2, 3]), consistent_mask([0, 2], [3]),
                      consistent_mask([1, 4], []), consistent_mask([3, 5], [0])])
    theta = rng.normal(0, 0.3, 2 * F)
    f = lambda th: perdida_parcial(th[:F], th[F:], phi, cmask, l2=0.01)[0]
    _, _, gL, gR = perdida_parcial(theta[:F], theta[F:], phi, cmask, l2=0.01)
    assert_close(np.concatenate([gL, gR]), diferencias_finitas(f, theta))


def _rl_check(mon_mask):
    cod, phi, rng = estados(n=5, seed=1)
    F = cod.F
    theta = rng.normal(0, 0.3, 2 * F)
    ref = rng.normal(0, 0.3, 2 * F)
    logp_ref = log_probs(ref[:F], ref[F:], phi, mon_mask)
    p = np.exp(log_probs(theta[:F], theta[F:], phi, mon_mask))
    acciones = np.array([rng.choice(90, p=pi / pi.sum()) for pi in p])     # acciones legales
    recompensas = np.array([1.0, -1.0, 1.0, 0.0, -1.0])
    args = (phi, acciones, recompensas, logp_ref, 0.05, 0.2, mon_mask)
    f = lambda th: perdida_rl(th[:F], th[F:], *args)[0]
    _, _, gL, gR = perdida_rl(theta[:F], theta[F:], *args)
    assert_close(np.concatenate([gL, gR]), diferencias_finitas(f, theta))


def test_rl_update_gradient():
    _rl_check(None)


def test_rl_update_gradient_with_masked_second_step():
    # Pokemon 2 y 5 no disponibles en algunos estados: los pares que los contienen quedan con prob. 0
    mask = np.ones((5, 6), bool)
    mask[0, 2] = mask[1, 5] = mask[3, 2] = False
    _rl_check(mask)


def _datos_vhat(vh):
    """Lote de 3 ejemplos con |C| = 3, 1 y 6 (|C| > 1 en ambos lados), un Pokemon con 2 movimientos (<PAD>) y especies <UNK>."""
    cod, t1, t2 = codificador_falso()
    corto = [dict(m) for m in t1]
    corto[0]["moves"] = corto[0]["moves"][:2]
    tok1 = np.stack([vh.tokens(corto), vh.tokens(t2), vh.tokens(t1)])
    tok2 = np.stack([vh.tokens(t2), vh.tokens(t1), vh.tokens(corto)])
    m1 = np.stack([consistent_mask([0, 1], [2]), consistent_mask([2, 3], [0, 1]), consistent_mask([1, 4], [])]).astype(float)
    m2 = np.stack([consistent_mask([4, 5], [0, 1]), consistent_mask([0, 5], [1]), consistent_mask([2, 3], [])]).astype(float)
    assert set(m1.sum(1)) == {3.0, 1.0, 6.0}
    return tok1, tok2, m1 / m1.sum(1, keepdims=True), m2 / m2.sum(1, keepdims=True), np.array([1.0, 0.0, 1.0])


def _revisar_grupos(vh, coordenadas_por_grupo=None, l2=0.01):
    """Compara el gradiente a mano de la perdida (11) con diferencias finitas, grupo por grupo (error relativo < 1e-5)."""
    tok1, tok2, q1, q2, y = _datos_vhat(vh)
    rng = np.random.default_rng(11)
    vh.p["A"] = rng.normal(0, 0.5, vh.p["A"].shape)                 # A aleatoria: con A = 0 el gradiente del resto seria 0
    _, _, grad, _ = perdida_marginal(vh.p, tok1, tok2, q1, q2, y, l2=l2)
    for nombre in nombres_parametros():
        base = vh.p[nombre]
        n = base.size
        idx = np.arange(n) if coordenadas_por_grupo is None else np.sort(rng.choice(n, min(n, coordenadas_por_grupo), replace=False))

        def f(valores):
            q = dict(vh.p)
            q[nombre] = valores.reshape(base.shape)
            return perdida_marginal(q, tok1, tok2, q1, q2, y, l2=l2, con_grad=False)[0]

        numerico = np.zeros(len(idx))
        for j, i in enumerate(idx):
            d = np.zeros(n)
            d[i] = EPS
            numerico[j] = (f(base.ravel() + d) - f(base.ravel() - d)) / (2 * EPS)
        try:
            assert_close(grad[nombre].ravel()[idx], numerico)
        except AssertionError as e:
            raise AssertionError(f"grupo {nombre}: {e}") from None


def test_vhat_marginal_likelihood_gradient_all_params_small_dims():
    """Todas las coordenadas de los 14 grupos, con d = hidden = 8 y K = 4 (rapido)."""
    cod, _, _ = codificador_falso()
    _revisar_grupos(VHat(cod, d=8, hidden=8, K=4, seed=1))


def test_vhat_marginal_likelihood_gradient_default_dims_sampled():
    """Dimensiones reales (d = hidden = 64, K = 16): 25 coordenadas al azar por grupo."""
    cod, _, _ = codificador_falso()
    _revisar_grupos(VHat(cod, seed=2), coordenadas_por_grupo=25)
