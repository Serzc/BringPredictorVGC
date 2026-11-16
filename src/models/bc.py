"""Politica lineal softmax autorregresiva y rho_BC (imitacion con etiquetas parciales, perdida 7).

Implementacion NumPy descrita en el plan v2 §7.2-7.3; se sustituyo DeepSets.

Modelo (phi = features por Pokemon de utils/features.py, forma (6, F)):
  puntaje_lider(x)   = theta_L . phi_x          puntaje_reserva(x) = theta_R . phi_x
  paso 1: logit del par lider p        z_p     = sum_{x en p} puntaje_lider(x)        (15 opciones)
  paso 2: logit del par de reservas r  u_{p,r} = sum_{x en r} puntaje_reserva(x)      (6 opciones,
          solo entre los 4 que quedan: el paso 2 esta enmascarado por el paso 1)
  pi(b=(p,r)|s) = softmax(z)_p * softmax(u_p)_r

Mascara opcional por Pokemon (mon_mask, (6,) bool): un par con un Pokemon no
disponible recibe logit NEG (probabilidad 0). Con los 6 disponibles hay 90 brings.

Gradiente: para cualquier perdida de la forma  sum_b c_b * log pi(b)  (con c fijo)
el gradiente se calcula en gradiente_ponderado(); BC, entropia, KL y REINFORCE se
reducen a elegir los pesos c (ver models/rl_agent.py). Se verifica por diferencias
finitas en src/tests/test_gradients.py.
"""
from __future__ import annotations

import numpy as np

from models.base import MetodoBring, one_hot
from utils.brings import LEAD_MON, RES_MON

NEG = -1e9   # logit de una opcion prohibida (probabilidad 0 sin producir NaN)


# ---------------------------------------------------------------------------
# Funciones de la politica sobre un lote de estados phi: (N, 6, F)
# ---------------------------------------------------------------------------
def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = x.max(axis=axis, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=axis, keepdims=True))


def logits(theta_L, theta_R, phi, mon_mask=None):
    """Logits del paso 1 (N, 15) y del paso 2 (N, 15, 6)."""
    s_lider = phi @ theta_L                                  # (N, 6) puntaje de cada Pokemon como lider
    s_reserva = phi @ theta_R                                # (N, 6) puntaje como reserva
    z = s_lider @ LEAD_MON.T                                 # (N, 15)
    u = np.einsum("nm,prm->npr", s_reserva, RES_MON)         # (N, 15, 6)
    if mon_mask is not None:
        ok = np.asarray(mon_mask, bool)                      # (N, 6)
        # un par es legal solo si sus 2 Pokemon estan disponibles
        lider_ok = (ok[:, None, :] | (LEAD_MON[None] == 0)).all(-1)              # (N, 15)
        reserva_ok = (ok[:, None, None, :] | (RES_MON[None] == 0)).all(-1)       # (N, 15, 6)
        z = np.where(lider_ok, z, NEG)
        u = np.where(reserva_ok, u, NEG)
    return z, u


def log_probs(theta_L, theta_R, phi, mon_mask=None) -> np.ndarray:
    """log pi(b|s) para los 90 brings: (N, 90)."""
    z, u = logits(theta_L, theta_R, phi, mon_mask)
    return (_log_softmax(z)[:, :, None] + _log_softmax(u)).reshape(len(phi), 90)


def gradiente_ponderado(theta_L, theta_R, phi, c, mon_mask=None):
    """Gradiente de  sum_n sum_b c[n, b] * log pi(b | s_n)  respecto de theta_L y theta_R.

    d log pi(p,r) / d z_q   = 1[q=p] - softmax(z)_q
    d log pi(p,r) / d u_p,s = 1[s=r] - softmax(u_p)_s
    y z, u son lineales en theta (sumas de puntajes de los Pokemon del par).
    """
    z, u = logits(theta_L, theta_R, phi, mon_mask)
    pz = np.exp(_log_softmax(z))                             # (N, 15) probabilidades del paso 1
    pu = np.exp(_log_softmax(u))                             # (N, 15, 6) probabilidades del paso 2
    C = c.reshape(len(phi), 15, 6)
    peso_lider = C.sum(2)                                    # (N, 15) masa de c en cada par lider
    total = C.sum((1, 2))                                    # (N,)
    g_z = peso_lider - total[:, None] * pz                   # dL/dz
    g_u = C - peso_lider[:, :, None] * pu                    # dL/du
    g_mon_lider = g_z @ LEAD_MON                             # (N, 6) de vuelta a cada Pokemon
    g_mon_reserva = np.einsum("npr,prm->nm", g_u, RES_MON)
    grad_L = np.einsum("nmf,nm->f", phi, g_mon_lider)
    grad_R = np.einsum("nmf,nm->f", phi, g_mon_reserva)
    return grad_L, grad_R


def greedy(theta_L, theta_R, phi, mon_mask=None) -> np.ndarray:
    """Argmax autorregresivo: mejor par lider y, dado ese par, mejores reservas. (N,) bring_ids."""
    z, u = logits(theta_L, theta_R, phi, mon_mask)
    p = z.argmax(1)
    r = u[np.arange(len(p)), p].argmax(1)
    return p * 6 + r


# ---------------------------------------------------------------------------
# Perdida (7) de etiquetas parciales
# ---------------------------------------------------------------------------
def perdida_parcial(theta_L, theta_R, phi, cmask, l2=0.0):
    """Perdida (7): media de -log sum_{b en C(o)} pi(b|s) + L2, y su gradiente.

    cmask (N, 90): brings consistentes con lo observado. Si |C|=1 es entropia cruzada normal.
    El gradiente usa pesos c_b = -pi(b)/sum_C pi  para b en C (posterior dentro de C(o)).
    """
    N = len(phi)
    lp = log_probs(theta_L, theta_R, phi)
    lp_C = np.where(cmask, lp, -np.inf)
    m = lp_C.max(1, keepdims=True)
    log_masa_C = (m + np.log(np.exp(lp_C - m).sum(1, keepdims=True)))[:, 0]   # log sum_C pi
    perdida_por_ejemplo = -log_masa_C
    posterior = np.where(cmask, np.exp(lp - log_masa_C[:, None]), 0.0)        # pi(b)/sum_C pi
    gL, gR = gradiente_ponderado(theta_L, theta_R, phi, -posterior / N)
    perdida = perdida_por_ejemplo.mean() + 0.5 * l2 * (theta_L @ theta_L + theta_R @ theta_R)
    return perdida, perdida_por_ejemplo, gL + l2 * theta_L, gR + l2 * theta_R


# ---------------------------------------------------------------------------
# Optimizador Adam (a mano, para no depender de torch)
# ---------------------------------------------------------------------------
class Adam:
    def __init__(self, n_params: int, lr: float, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = np.zeros(n_params)      # media movil del gradiente
        self.v = np.zeros(n_params)      # media movil del gradiente al cuadrado
        self.t = 0                       # numero de pasos dados

    def paso(self, params: np.ndarray, grad: np.ndarray) -> np.ndarray:
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        m_hat = self.m / (1 - self.b1 ** self.t)
        v_hat = self.v / (1 - self.b2 ** self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ---------------------------------------------------------------------------
# La politica como MetodoBring (sirve para rho_BC y, por herencia, para el agente RL)
# ---------------------------------------------------------------------------
class PoliticaLineal(MetodoBring):
    """Politica lineal softmax autorregresiva. name='bc' para rho_BC."""

    def __init__(self, codificador, theta_L=None, theta_R=None, name="bc", greedy_mode=True):
        self.cod = codificador
        F = codificador.F
        self.theta_L = np.zeros(F) if theta_L is None else np.asarray(theta_L, float)
        self.theta_R = np.zeros(F) if theta_R is None else np.asarray(theta_R, float)
        self.name, self.deterministic = name, greedy_mode

    # vector unico de parametros (para Adam y para guardar)
    @property
    def params(self) -> np.ndarray:
        return np.concatenate([self.theta_L, self.theta_R])

    @params.setter
    def params(self, v: np.ndarray) -> None:
        F = self.cod.F
        self.theta_L, self.theta_R = v[:F].copy(), v[F:].copy()

    def estado(self, team_self, team_opp) -> np.ndarray:
        return self.cod.mon_features(team_self, team_opp)

    def probs_desde_estado(self, phi: np.ndarray) -> np.ndarray:
        """(90,) probabilidades para un estado ya codificado (6, F)."""
        p = np.exp(log_probs(self.theta_L, self.theta_R, phi[None])[0])
        return p / p.sum()

    def probs(self, team_self, team_opp):
        phi = self.estado(team_self, team_opp)
        if self.deterministic:
            return one_hot(int(greedy(self.theta_L, self.theta_R, phi[None])[0]))
        return self.probs_desde_estado(phi)

    def guardar(self, path) -> None:
        np.savez(path, theta_L=self.theta_L, theta_R=self.theta_R, nombres=np.array(self.cod.nombres))

    @classmethod
    def cargar(cls, path, codificador, name="bc"):
        d = np.load(path)
        if list(d["nombres"]) != codificador.nombres:
            raise ValueError(f"{path}: las features del checkpoint no coinciden con el codificador actual")
        return cls(codificador, d["theta_L"], d["theta_R"], name=name)
