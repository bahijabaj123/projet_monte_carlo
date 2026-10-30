
# Fiche de notations — à utiliser PARTOUT (code, rapport, slides)

## Marché
| Symbole | Sens | Type |
|---|---|---|
| S_0 | prix initial du sous-jacent | constante |
| S_t | prix à l'instant t | processus |
| K | strike | constante |
| T | maturité (années) | constante |
| r | taux sans risque | constante |
| σ | volatilité | constante |
| W_t | mouvement brownien standard | processus |
| Z | N(0,1) | variable |

## Options
| Symbole | Sens |
|---|---|
| (x)^+ | max(x, 0) |
| C_BS | prix exact du call européen (Black-Scholes) |
| P_BS | prix exact du put européen |
| C_MC | estimateur Monte-Carlo du call européen |
| C_as | prix du call asiatique arithmétique |
| C_geo | prix exact du call asiatique géométrique |

## Simulation
| Symbole | Sens | Convention code |
|---|---|---|
| N | nombre de trajectoires | `N` |
| M | nombre de pas de temps | `M` |
| dt | T/M | `dt` |
| s | écart-type empirique des payoffs | `std(ddof=1)` |
| IC95 | intervalle de confiance à 95 % | `Z95=1.95996…` |
| MSE | erreur quadratique moyenne | |

## Règles
1. Aucune valeur numérique en dur hors `config.py`.
2. Tous les RNG sont explicitement passés (`make_rng(SEED+k)`).
3. Notation LaTeX : `\hat{C}_{\mathrm{MC}}`, `\sigma`, `\Phi` (fonction de répartition N(0,1)).
4. Figures : PDF (rapport) + PNG (slides), mêmes couleurs (`PALETTE` dans `plots.py`).
5. En français : virgule décimale dans le texte, point dans le code.
