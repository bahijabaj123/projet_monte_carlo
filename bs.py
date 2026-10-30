"""
Etape 1 - Formule analytique de Black-Scholes (reference exacte).
"""
import numpy as np
from scipy.stats import norm

from config import K, S0, T, r, sigma


def d1_d2(S0=S0, K=K, T=T, r=r, sigma=sigma):
    """Retourne (d1, d2)."""
    nu = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / nu
    return d1, d1 - nu


def bs_call(S0=S0, K=K, T=T, r=r, sigma=sigma):
    """Prix exact du call europeen."""
    d1, d2 = d1_d2(S0, K, T, r, sigma)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put(S0=S0, K=K, T=T, r=r, sigma=sigma):
    """Prix exact du put europeen."""
    d1, d2 = d1_d2(S0, K, T, r, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


if __name__ == "__main__":
    print(f"C_BS = {bs_call():.6f}   P_BS = {bs_put():.6f}")
    print(f"Parite call-put : C - P = {bs_call() - bs_put():.6f}")
    print(f"S0 - K exp(-rT) = {S0 - K * np.exp(-r * T):.6f}")
