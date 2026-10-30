"""Validation : le MC europeen doit tomber dans l'IC de Black-Scholes."""
import numpy as np
from config import make_rng, SEED
import bs
import mc_european as mc


def test_call_mc_dans_ic_bs():
    rng = make_rng(SEED)
    exact = float(bs.bs_call())
    for N in (10**4, 10**5, 10**6):
        res = mc.mc_call(N, rng)
        assert res.contains(exact), (
            f"N={N} : C_MC={res.price:.4f} hors IC "
            f"[{res.ci_low:.4f},{res.ci_high:.4f}] vs C_BS={exact:.4f}"
        )


def test_put_call_parity():
    from config import S0, K, T, r
    diff = float(bs.bs_call()) - float(bs.bs_put())
    assert np.isclose(diff, S0 - K * np.exp(-r * T), atol=1e-10)


def test_call_bs_valeur_reference():
    assert abs(float(bs.bs_call()) - 10.4506) < 1e-4
