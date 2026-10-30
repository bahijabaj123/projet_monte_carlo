"""
Figures du projet - style commun a toute l'equipe.
Chaque figure est enregistree en PDF (pour LaTeX) et en PNG (pour les slides).
Les autres membres doivent REUTILISER set_style() et save() pour garder une
presentation homogene.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import lognorm

from config import FIG_DIR, K, S0, T, r, sigma
import mc_european as mc

PALETTE = {"main": "#1f4e79", "accent": "#c0392b", "grey": "#7f7f7f", "green": "#2e8b57"}


def set_style():
    plt.rcParams.update({
        "figure.figsize": (7.5, 4.3),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.frameon": False,
        "font.size": 10,
    })


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{name}.pdf")
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200)
    plt.close(fig)


def fig_trajectoires(rng, n=10, M=252):
    paths = mc.simulate_paths(n, M, rng)
    t = np.linspace(0, T, M + 1)
    fig, ax = plt.subplots()
    ax.plot(t, paths.T, lw=0.9)
    ax.axhline(K, color="k", ls="--", lw=1, label="Strike $K$")
    ax.set_xlabel("$t$ (annees)")
    ax.set_ylabel("$S_t$")
    ax.set_title("Trajectoires du mouvement brownien geometrique")
    ax.legend()
    save(fig, "01_trajectoires")


def fig_distribution_ST(ST):
    """Histogramme de S_T simule contre la densite log-normale theorique."""
    nu = sigma * np.sqrt(T)
    m = (r - 0.5 * sigma**2) * T
    dist = lognorm(s=nu, scale=S0 * np.exp(m))
    x = np.linspace(ST.min(), np.quantile(ST, 0.9995), 400)
    fig, ax = plt.subplots()
    ax.hist(ST, bins=120, density=True, range=(x[0], x[-1]), color=PALETTE["main"],
            alpha=0.55, label="$S_T$ simule")
    ax.plot(x, dist.pdf(x), color=PALETTE["accent"], lw=2, label="densite log-normale")
    ax.axvline(K, color="k", ls="--", lw=1, label="Strike $K$")
    ax.set_xlabel("$S_T$")
    ax.set_ylabel("densite")
    ax.set_title("Loi de $S_T$ : simulation et densite log-normale")
    ax.legend()
    save(fig, "02_distribution_ST")


def fig_convergence_ic(results, exact):
    Ns = np.array([x.N for x in results])
    prix = np.array([x.price for x in results])
    hw = np.array([x.half_width for x in results])
    fig, ax = plt.subplots()
    ax.errorbar(Ns, prix, yerr=hw, fmt="o-", capsize=3, color=PALETTE["main"],
                label="Monte-Carlo (IC a 95 %)")
    ax.axhline(exact, color=PALETTE["accent"], ls="--", lw=1.5,
               label=f"Black-Scholes = {exact:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("$N$ (nombre de simulations)")
    ax.set_ylabel("prix du call")
    ax.set_title("Convergence de Monte-Carlo vers Black-Scholes")
    ax.legend()
    save(fig, "03_convergence_IC")


def fig_vitesse(results):
    Ns = np.array([x.N for x in results], dtype=float)
    se = np.array([x.std_error for x in results])
    fig, ax = plt.subplots()
    ax.loglog(Ns, se, "o-", color=PALETTE["main"], label="erreur standard (simulee)")
    ax.loglog(Ns, se[0] * np.sqrt(Ns[0] / Ns), "k--", lw=1,
              label=r"reference en $1/\sqrt{N}$")
    ax.set_xlabel("$N$")
    ax.set_ylabel("erreur standard")
    ax.set_title(r"Vitesse de convergence en $O(1/\sqrt{N})$")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    save(fig, "04_vitesse_convergence")


def fig_euler(Ms, results, exact):
    prix = np.array([x.price for x in results])
    hw = np.array([x.half_width for x in results])
    fig, ax = plt.subplots()
    ax.errorbar(Ms, prix, yerr=hw, fmt="o-", capsize=3, color=PALETTE["green"],
                label="schema d'Euler (IC a 95 %)")
    ax.axhline(exact, color=PALETTE["accent"], ls="--", lw=1.5,
               label="Black-Scholes (schema exact)")
    ax.set_xscale("log")
    ax.set_xlabel("$M$ (nombre de pas de temps)")
    ax.set_ylabel("prix du call")
    ax.set_title("Erreur de discretisation du schema d'Euler")
    ax.legend()
    save(fig, "05_euler_vs_exact")
