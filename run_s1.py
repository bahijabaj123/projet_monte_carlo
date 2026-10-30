"""
Semaine 1 - Etapes 1 et 2 : Black-Scholes + Monte-Carlo standard.

    python run_s1.py

Produit :
  - figures/*.pdf|png             (figures du rapport)
  - report/generated/*.tex        (parametres, resultats, tableaux -> \\input dans le rapport)
  - report/generated/results_s1.json  (resultats bruts)
Le rapport LaTeX lit ces fichiers : texte et code ne peuvent donc pas diverger.
"""
import json

import numpy as np

import bs
import mc_european as mc
import plots
from config import (GEN_DIR, K, N_MAX_EXP, N_MIN_EXP, N_POINTS, N_REF, SEED, S0, T,
                    make_rng, r, sigma)

COV_N, COV_R = 2000, 5000            # etude de couverture de l'IC
EULER_M = [1, 2, 5, 10, 25, 50, 100]
EULER_N = 10**6


def fr(x, nd=4):
    """Nombre au format francais (virgule decimale)."""
    return f"{x:.{nd}f}".replace(".", ",")


def fr_int(n):
    """Entier avec espace fine comme separateur de milliers (LaTeX)."""
    return f"{int(n):,}".replace(",", r"\,")


def write(name, text):
    (GEN_DIR / name).write_text(text, encoding="utf-8")


def main():
    plots.set_style()

    # ---------------- Etape 1 : valeur exacte ----------------
    C, P = float(bs.bs_call()), float(bs.bs_put())
    print(f"[Etape 1] C_BS = {C:.6f}   P_BS = {P:.6f}")

    # ---------------- Etape 2 : convergence du MC standard ----------------
    Ns = np.unique(np.logspace(N_MIN_EXP, N_MAX_EXP, N_POINTS).astype(int))
    results = mc.convergence_study(Ns, make_rng(SEED))
    rows, outside = [], 0
    for res in results:
        inside = res.contains(C)
        outside += int(not inside)
        rows.append(
            f"{fr_int(res.N)} & {fr(res.price)} & [{fr(res.ci_low)} ; {fr(res.ci_high)}] "
            f"& {fr(abs(res.price - C))} & {fr(res.std_error)} & {'oui' if inside else 'non'} \\\\"
        )
        print(f"  N={res.N:>8d}  C_MC={res.price:8.4f}  IC=[{res.ci_low:8.4f}, {res.ci_high:8.4f}]"
              f"  |erreur|={abs(res.price - C):.4f}  C_BS dans IC : {inside}")
    write("table_convergence.tex", "\n".join(rows) + "\n")

    # ---------------- Couverture de l'IC ----------------
    cov = mc.coverage_study(COV_N, COV_R, make_rng(SEED + 1), C)
    print(f"[Couverture] N={COV_N}, R={COV_R} : {100 * cov:.2f} % (attendu ~ 95 %)")

    # ---------------- Schema d'Euler vs schema exact ----------------
    rng_e = make_rng(SEED + 2)
    euler = [mc.mc_call_euler(EULER_N, M, rng_e) for M in EULER_M]
    erows = []
    for M, res in zip(EULER_M, euler):
        erows.append(f"{M} & {fr(res.price)} & [{fr(res.ci_low)} ; {fr(res.ci_high)}] "
                     f"& {fr(res.price - C)} \\\\")
        print(f"  Euler M={M:>3d}  C={res.price:8.4f}  biais estime={res.price - C:+.4f}")
    write("table_euler.tex", "\n".join(erows) + "\n")

    # ---------------- Moments de S_T ----------------
    mom = mc.moments_check(N_REF, make_rng(SEED + 3))
    print(f"[Moments] E[S_T] : MC={mom['mean']:.4f} theorie={mom['mean_th']:.4f} ; "
          f"Var[S_T] : MC={mom['var']:.3f} theorie={mom['var_th']:.3f}")

    # ---------------- Figures ----------------
    plots.fig_trajectoires(make_rng(SEED + 4))
    ST = mc.terminal_price(make_rng(SEED + 5).standard_normal(200_000))
    plots.fig_distribution_ST(ST)
    plots.fig_convergence_ic(results, C)
    plots.fig_vitesse(results)
    plots.fig_euler(EULER_M, euler, C)

    # ---------------- Macros LaTeX ----------------
    write("params.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in {
        "paramSzero": fr(S0, 0), "paramK": fr(K, 0), "paramT": fr(T, 0),
        "paramr": fr(100 * r, 0) + r"\,\%", "paramsigma": fr(100 * sigma, 0) + r"\,\%",
        "paramSeed": str(SEED), "paramNref": fr_int(N_REF),
    }.items()))
    ref = results[-1]
    write("results.tex", "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in {
        "valCall": fr(C), "valPut": fr(P),
        "valMCref": fr(ref.price), "valMCrefLow": fr(ref.ci_low), "valMCrefHigh": fr(ref.ci_high),
        "valMCrefSE": fr(ref.std_error, 5),
        "nbPointsIC": str(len(results)), "nbOutsideIC": str(outside),
        "covN": fr_int(COV_N), "covR": fr_int(COV_R), "covRate": fr(100 * cov, 1) + r"\,\%",
        "valMeanST": fr(mom["mean"], 3), "valMeanTheory": fr(mom["mean_th"], 3),
        "valVarST": fr(mom["var"], 1), "valVarTheory": fr(mom["var_th"], 1),
    }.items()))

    write("results_s1.json", json.dumps({
        "C_BS": C, "P_BS": P,
        "convergence": [{"N": x.N, "price": x.price, "se": x.std_error} for x in results],
        "coverage": {"N": COV_N, "R": COV_R, "rate": cov},
        "euler": [{"M": M, "price": x.price, "se": x.std_error} for M, x in zip(EULER_M, euler)],
        "moments": mom,
    }, indent=2))
    print("\nS1 termine : figures/ et report/generated/ mis a jour.")


if __name__ == "__main__":
    main()
