"""
Parametres communs du projet.

REGLE D'EQUIPE : aucune valeur numerique du modele ne doit etre ecrite en dur
ailleurs que dans ce fichier. Tous les modules importent leurs parametres ici.
"""
from pathlib import Path

import numpy as np

# Chemins
ROOT = Path(__file__).resolve().parent
FIG_DIR = ROOT / "figures"
GEN_DIR = ROOT / "report" / "generated"
for _d in (FIG_DIR, GEN_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Modele de Black-Scholes (mesure risque-neutre)
S0 = 100.0      # prix initial du sous-jacent
K = 100.0       # strike
T = 1.0         # maturite (en annees)
r = 0.05        # taux sans risque
sigma = 0.20    # volatilite

# Simulation
SEED = 2026
Z95 = 1.959963984540054
N_MIN_EXP, N_MAX_EXP = 3, 6
N_POINTS = 13
N_REF = 10**6


def make_rng(seed: int = SEED) -> np.random.Generator:
    """Generateur aleatoire reproductible. Toujours le passer explicitement."""
    return np.random.default_rng(seed)
