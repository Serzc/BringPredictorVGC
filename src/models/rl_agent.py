"""Agente RL: REINFORCE con baseline + bonus de entropia + KL a rho_BC (plan v2 §7.5), en numpy.

BORRADOR asistido por IA; responsable sugerido: Jose (revisar y hacer commit).

Perdida que se minimiza en cada actualizacion (N episodios = N batallas reales):
  L(theta) = - (1/N) sum_i (R_i - retorno_medio) * log pi(a_i | s_i)      <- REINFORCE con baseline
             - beta_H  * (1/N) sum_i H[pi(.|s_i)]                           <- explorar (no colapsar)
             + beta_KL * (1/N) sum_i KL(pi(.|s_i) || rho_BC(.|s_i))        <- no alejarse de lo humano
  R_i            = +1 si gano la batalla i, -1 si perdio, 0 empate/tope de turnos
  retorno_medio  = media de R en el lote; se resta para bajar la varianza (no cambia el valor esperado)
  H[pi]          = entropia de la politica sobre los 90 brings
Episodio de 2 pasos (lideres, reservas), gamma = 1, sin shaping: el retorno es R.

Los tres terminos son "sum_b c_b log pi(b)" con pesos c distintos, asi que se usa
el mismo gradiente_ponderado() de models/bc.py:
  REINFORCE: c[i, a_i] = -(R_i - retorno_medio)/N
  entropia:  c[i, b]   = +beta_H  * pi_b * log pi_b / N
  KL:        c[i, b]   = +beta_KL * pi_b * (log pi_b - log rho_b) / N
(se usa que sum_b pi_b * grad log pi_b = 0). Verificado en src/tests/test_gradients.py.
"""
from __future__ import annotations

import numpy as np

from models.bc import Adam, PoliticaLineal, gradiente_ponderado, log_probs


def perdida_rl(theta_L, theta_R, phi, acciones, recompensas, logp_ref, beta_H, beta_KL, mon_mask=None):
    """Perdida L(theta) y su gradiente. phi (N,6,F); acciones (N,); recompensas (N,); logp_ref (N,90)."""
    N = len(phi)
    recompensas = np.asarray(recompensas, float)
    lp = log_probs(theta_L, theta_R, phi, mon_mask)                   # (N, 90) log pi
    pi = np.exp(lp)
    retorno_medio = recompensas.mean()                                # baseline del lote
    ventaja = recompensas - retorno_medio                             # (N,) R_i - retorno_medio
    lp_accion = lp[np.arange(N), acciones]                            # log pi(a_i|s_i)
    pi_log_pi = np.where(pi > 0, pi * lp, 0.0)                        # evita 0 * (-1e9)
    entropia = -pi_log_pi.sum(1)                                      # (N,) H[pi(.|s_i)]
    pi_log_ratio = np.where(pi > 0, pi * (lp - logp_ref), 0.0)
    kl = pi_log_ratio.sum(1)                                          # (N,) KL(pi || rho_BC)
    perdida = -(ventaja * lp_accion).mean() - beta_H * entropia.mean() + beta_KL * kl.mean()

    # pesos c para el gradiente de cada termino
    c = np.zeros((N, 90))
    c[np.arange(N), acciones] -= ventaja / N
    c += beta_H * pi_log_pi / N
    c += beta_KL * pi_log_ratio / N
    gL, gR = gradiente_ponderado(theta_L, theta_R, phi, c, mon_mask)
    info = {"loss": float(perdida), "entropy": float(entropia.mean()), "kl_to_bc": float(kl.mean()),
            "mean_reward": float(retorno_medio)}
    return perdida, info, gL, gR


class AgenteRL(PoliticaLineal):
    """Politica lineal que arranca en rho_BC y se mejora con batallas jugadas."""

    def __init__(self, codificador, bc: PoliticaLineal, lr: float, beta_H: float, beta_KL: float,
                 grad_clip: float = 1.0):
        super().__init__(codificador, bc.theta_L.copy(), bc.theta_R.copy(), name="agent", greedy_mode=True)
        self.bc = bc                                   # referencia fija para el termino KL
        self.beta_H, self.beta_KL, self.grad_clip = beta_H, beta_KL, grad_clip
        self.opt = Adam(2 * codificador.F, lr)
        self.pasos = 0                                 # actualizaciones hechas

    def muestrear(self, phi: np.ndarray, rng: np.random.Generator) -> int:
        """Muestrea un bring de pi(.|s) (en entrenamiento se explora; en evaluacion se usa greedy)."""
        return int(rng.choice(90, p=self.probs_desde_estado(phi)))

    def actualizar(self, phis: np.ndarray, acciones, recompensas) -> dict:
        """Un paso de Adam con la perdida L(theta) sobre un lote de episodios."""
        logp_ref = log_probs(self.bc.theta_L, self.bc.theta_R, phis)     # log rho_BC, fijo
        _, info, gL, gR = perdida_rl(self.theta_L, self.theta_R, phis, np.asarray(acciones),
                                     recompensas, logp_ref, self.beta_H, self.beta_KL)
        grad = np.concatenate([gL, gR])
        norma = float(np.linalg.norm(grad))
        if norma > self.grad_clip:                                          # recorte de la norma
            grad *= self.grad_clip / norma
        self.params = self.opt.paso(self.params, grad)
        self.pasos += 1
        info["grad_norm"] = norma
        return info
