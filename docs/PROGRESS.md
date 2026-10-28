# PROGRESS — black-scholes-pricer

## État au 2026-10-28 : v1.0.0, phases 0 à 6 terminées

| Phase | Statut | Où le vérifier |
|---|---|---|
| 0 Audit | fait (2026-10-13) | `docs/AUDIT.md`, `docs/audit/` |
| 1 Structure | fait | `src/bspricer/`, `pyproject.toml`, `data/sample/` |
| 2 Tests | fait : couverture 99 % (branches comprises) | `tests/`, `python -m pytest --cov` |
| 3 Qualité, CI | fait : ruff, mypy --strict, CI sur Python 3.12 / 3.13 / 3.14 | `.github/workflows/ci.yml`, badge du README |
| 4 Résultats | fait : R1–R4 | `scripts/run_results.py`, `results/` |
| 5 README | fait : blocs de résultats générés et vérifiés par test | `README.md`, `tests/test_readme.py` |
| 6 Livraison | notebook, tag `v1.0.0`, notes de version faits ; **épinglage du repo à faire à la main** | `notebooks/demo.ipynb`, onglet Releases |

## Décisions D1–D6 (tranchées le 2026-10-28, séance autonome)

| # | Décision | Retenu |
|---|---|---|
| D1 | Base de travail | `main` public, commit `6339479` : identique au commit audité, aucune version plus récente sur GitHub. |
| D2 | Données figées | Oui : `data/sample/` (SPY Yahoo, VIX et T-bill 3 mois FRED), avec `SOURCE.md` et `SHA256SUMS`. |
| D3 | « Données réelles uniquement » contre Monte Carlo | Monte Carlo pour les expériences contrôlées (vol réalisée connue). **Backtest sur trajectoires SPY réelles** avec le VIX comme vol de vente (R4) : le résultat empirique porte sur des données réelles. |
| D4 | Critère de fréquence optimale | N\* qui minimise le RMSE du P&L total coûts inclus, comparé à un modèle « turnover fixe + b√N ». |
| D5 | Sticky-strike / sticky-delta | Hors v1.0 ; candidat v1.1. |
| D6 | Nom du package | `bspricer`. |

## Constats de l'audit → corrections

| ID | Correction | Commit |
|---|---|---|
| C1 unité de theta | Grecques en unités naturelles (theta par an) ; attribution exacte dans `delta_hedge` | `2493934`, `d73f914` |
| C2 terme de financement | terme `carry` dans `PnLDecomposition` ; gamma + theta + carry = `variance_pnl` (test) | `d73f914` |
| C3 limite σ → 0 | limite sur la moneyness forward ; tests q > r, put, forward = strike | `2493934` |
| C4 dividendes | dividendes crédités ; test de non-biais \|t\| < 4 | `d73f914` |
| M1 dt | pas déduit de la grille, argument `dt` supprimé | `d73f914` |
| M2 vol implicite | `ImpliedVolError` si la vol n'est pas identifiable | `2493934` |
| M3 validation | contrôles S, K > 0, T, σ ≥ 0, valeurs finies, type exact | `2493934` |
| M4 chiffres faux | README régénéré depuis `results/` ; ancien notebook supprimé | `ec1ce04`, `3da851e` |
| M5 portée empirique | backtest R4 sur clôtures SPY réelles, limites écrites | `67431da`, `3da851e` |
| M6 tests | parité, bornes, limites, différences finies, CRR, MC, régression, README | `2493934` → `67edfee` |
| M7 données | données figées et sommes SHA256 ; plus d'écriture du cache | `2fda71c` |
| M8 packaging, CI | `pyproject` épinglé, LICENSE, ruff, mypy, matrice CI | `2fda71c`, `6fe1982` |
| M9 chiffre inventé | `NOTEBOOK_PATCHES.md` supprimé | `2fda71c` |
| m1–m8 | traités par la réécriture (formule de vol réalisée, delta du put expiré, tolérances EDP, justification du test call/put, pic de gamma testé en forme fermée…) | — |
| m9 `.pyc` obsolète | **à faire par Tancrède** : retirer `bs_pricer.cpython-312.pyc` des documents du projet Claude | — |

## Journal

### 2026-10-28 — Phases 1 à 6 (séance autonome)
- Package `src/bspricer` : pricing, Grecques, vol implicite, CRR, Monte Carlo, couverture, backtest, vol.
- Résultats : R1 (pente −0,496 pour N ≥ 63), R2 (N\* = 504 / 168 / 84 / 42 à 2 / 5 / 10 / 20 bp),
  R3 (R² 0,97 du proxy dollar-gamma), R4 (59 mois SPY, +0,41 pour 100 de nominal en moyenne). Chiffres exacts dans `results/summary.json`.
- Auto-contrôles : oracle indépendant en Python pur, nouveau moteur = ancien moteur à 3e-13 près pour q = 0,
  reproduction bit à bit depuis un clone propre, suite exécutée sur 3.12, 3.13 et 3.14.
- Incidents CI corrigés : comparaison exacte de flottants sous Python 3.14 (`3f7dc44`), formatage (`0a94105`).

### 2026-10-13 — Phase 0 (audit)
- Audit du commit `6339479` : 4 critiques, 9 majeurs, 9 mineurs → `docs/AUDIT.md`.

## Reste à faire

1. Épingler le repo sur le profil GitHub (réglage du profil, non modifiable par Claude).
2. Mettre à jour la description du repo, qui affirme encore « 2000+ pytest assertions » et une couverture sur données SPY réelles au sens de la v0.2 (proposition dans le rapport de séance).
3. Optionnel v1.1 : delta sticky-strike contre sticky-delta sur un smile synthétique ; vol implicite ATM réelle à la place du VIX si une source publique est disponible.
