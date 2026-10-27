# PROGRESS — black-scholes-pricer

## Décisions D1–D6 (tranchées le 2026-10-28, séance autonome)

| # | Décision | Retenu |
|---|---|---|
| D1 | Base de travail | `main` public, commit `6339479` : identique au commit audité, aucune copie plus récente sur GitHub. |
| D2 | Données figées | Oui : `data/sample/` (SPY Yahoo, VIX et T-bill 3 mois FRED), avec `SOURCE.md` et `SHA256SUMS`. |
| D3 | « Données réelles uniquement » contre Monte Carlo | Monte Carlo pour les expériences contrôlées (vol réalisée connue). **Backtest sur trajectoires SPY réelles** avec le VIX comme vol de vente : le résultat empirique porte sur des données réelles. |
| D4 | Critère de fréquence optimale | N\* qui minimise le RMSE du P&L total coûts inclus, plus la frontière (coût moyen ; écart-type). |
| D5 | Sticky-strike / sticky-delta | Hors v1.0 ; à traiter en v1.1. |
| D6 | Nom du package | `bspricer`. |

## Journal

### 2026-10-13 — Phase 0 (audit) — terminée
- Audit du commit `6339479` : 4 critiques, 9 majeurs, 9 mineurs → `docs/AUDIT.md`.
- Preuves : `docs/audit/audit_phase0.py` (A01–A21, F01).
