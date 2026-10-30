"""Moments de S_T : moyenne et variance theoriques (loi log-normale)."""
from config import make_rng, SEED
import mc_european as mc


def test_moments_ST():
    rng = make_rng(SEED + 7)
    N = 10**6
    m = mc.moments_check(N, rng)
    se_mean = m["mean_se"]
    assert abs(m["mean"] - m["mean_th"]) <= 4 * se_mean
    assert abs(m["var"] - m["var_th"]) / m["var_th"] < 0.05
