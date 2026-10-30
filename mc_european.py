
"""
Etape 2 - Monte-Carlo standard pour le call europeen (banc d'essai).

Principe : C = exp(-rT) E[(S_T - K)^+]  est estime par la moyenne empirique de
N payoffs actualises independants. L'erreur standard est sigma_N / sqrt(N) et
l'intervalle de confiance a 95 % est  C_MC +/- 1,96 sigma_N / sqrt(N).
"""
from dataclasses import dataclass

import numpy as np

from config import K, S0, T, Z95, r, sigma


@dataclass(frozen=True)
class MCResult:
    """Resultat d'une estimation Monte-Carlo."""
    N: int
    price: float
    std_error: float

    @property
    def half_width(self) -> float:
        return Z95 * self.std_error

    @property
    def ci_low(self) -> float:
        return self.price - self.half_width

    @property
    def ci_high(self) -> float:
        return self.price + self.half_width

    def contains(self, value: float) -> bool:
        return self.ci_low <= value <= self.ci_high


# ----------------------------------------------------------------------
# Briques de simulation
# ----------------------------------------------------------------------
def terminal_price(Z, S0=S0, T=T, r=r, sigma=sigma):
    """Schema exact : S_T = S0 exp((r - sigma^2/2) T + sigma sqrt(T) Z), Z ~ N(0,1)."""
    return S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)


def call_payoff(ST, K=K):
    """Payoff du call europeen (S_T - K)^+."""
    return np.maximum(ST - K, 0.0)


def simulate_paths(n_paths, M, rng, S0=S0, T=T, r=r, sigma=sigma):
    """
    Trajectoires du GBM sur la grille t_i = i T / M, schema EXACT.
    Retourne un tableau (n_paths, M + 1) avec S_{t_0} = S0 en premiere colonne.
    """
    dt = T / M
    Z = rng.standard_normal((n_paths, M))
    increments = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(increments, axis=1)
    paths = S0 * np.exp(log_paths)
    return np.hstack([np.full((n_paths, 1), S0), paths])


# ----------------------------------------------------------------------
# Estimateurs
# ----------------------------------------------------------------------
def mc_call(N, rng, S0=S0, K=K, T=T, r=r, sigma=sigma) -> MCResult:
    """Estimateur Monte-Carlo standard du call europeen (schema exact, un seul pas)."""
    Z = rng.standard_normal(N)
    Y = np.exp(-r * T) * call_payoff(terminal_price(Z, S0, T, r, sigma), K)
    return MCResult(N, float(Y.mean()), float(Y.std(ddof=1) / np.sqrt(N)))


def mc_call_euler(N, M, rng, S0=S0, K=K, T=T, r=r, sigma=sigma) -> MCResult:
    """
    Meme estimateur mais S_T simule avec le schema d'Euler a M pas :
        S_{i+1} = S_i (1 + r dt + sigma sqrt(dt) Z_i).
    Sert a montrer l'erreur de discretisation (le schema exact n'en a pas).
    """
    dt = T / M
    S = np.full(N, S0)
    for _ in range(M):
        S = S + r * S * dt + sigma * S * np.sqrt(dt) * rng.standard_normal(N)
    Y = np.exp(-r * T) * call_payoff(S, K)
    return MCResult(N, float(Y.mean()), float(Y.std(ddof=1) / np.sqrt(N)))


# ----------------------------------------------------------------------
# Etudes de validation
# ----------------------------------------------------------------------
def convergence_study(Ns, rng):
    """Une estimation independante pour chaque N de la liste."""
    return [mc_call(int(N), rng) for N in Ns]


def coverage_study(N, R, rng, exact, chunk=500) -> float:
    """
    Taux de couverture empirique de l'IC a 95 % : proportion de R experiences
    independantes (de N simulations chacune) dont l'IC contient le prix exact.
    Doit etre proche de 0,95.
    """
    hits = 0
    done = 0
    while done < R:
        n = min(chunk, R - done)
        Z = rng.standard_normal((n, N))
        Y = np.exp(-r * T) * call_payoff(terminal_price(Z))
        m = Y.mean(axis=1)
        se = Y.std(axis=1, ddof=1) / np.sqrt(N)
        hits += int(np.sum(np.abs(m - exact) <= Z95 * se))
        done += n
    return hits / R


def moments_check(N, rng):
    """Moments empiriques de S_T compares aux valeurs theoriques (loi log-normale)."""
    ST = terminal_price(rng.standard_normal(N))
    return {
        "mean": float(ST.mean()),
        "var": float(ST.var(ddof=1)),
        "mean_th": float(S0 * np.exp(r * T)),
        "var_th": float(S0**2 * np.exp(2 * r * T) * (np.exp(sigma**2 * T) - 1)),
        "mean_se": float(ST.std(ddof=1) / np.sqrt(N)),
    }
