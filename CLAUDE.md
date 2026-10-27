# CLAUDE.md — black-scholes-pricer

## Standard « repo recruteur » (commun à tous les repos)

Objectif : un recruteur quant/trading comprend en 90 s ce que fait le repo et ce qu'il prouve,
et peut tout relancer en 2 commandes.

1. **Audit** — `docs/AUDIT.md` (problèmes classés par gravité) + plan des phases 1–6.
2. **Structure** — `src/bspricer/`, `tests/`, `notebooks/`, `data/sample/`, `scripts/`, `docs/` ;
   `pyproject.toml` épinglé ; `pip install -e .` ; fonctions pures, type hints, docstrings
   (formule + référence), aucun chemin absolu, seeds fixées.
3. **Tests** — cohérence financière (identités, cas limites, convergence), régression sur
   `data/sample/`, couverture ≥ 80 % sur le cœur de calcul.
4. **Qualité** — ruff (lint + format), mypy sur `src/`, GitHub Actions à chaque push, badge.
5. **Résultats** — `scripts/run_results.py` régénère toutes les figures et tables du README ;
   2 à 4 résultats chiffrés, chacun interprété en une phrase.
6. **README** (anglais, ordre imposé) — une phrase ; résultats clés (tableau + 1–2 figures) ;
   méthodologie et limites ; reproduire en 2 commandes ; structure ; références.
7. **Livraison** — notebook de démo < 2 min, tag + notes de version, brouillon LinkedIn.

Règles : relire `docs/AUDIT.md` et `docs/PROGRESS.md` avant chaque séance, mettre à jour
`docs/PROGRESS.md` après. Commits atomiques. **Ne jamais inventer un résultat** : chaque chiffre
du README sort d'un script exécuté (le bloc de résultats du README est généré, et
`tests/test_readme.py` vérifie qu'il correspond à `results/`).

## Commandes

```bash
pip install -e ".[dev]"              # installation (Python ≥ 3.12)
python -m pytest -m "not slow"       # tests rapides (< 30 s)
python -m pytest                     # suite complète, convergence comprise
ruff check . && ruff format --check . && mypy
python scripts/run_results.py        # régénère results/ et le bloc de résultats du README
```

## Conventions du code

* Unités naturelles partout : `T` en années, `sigma` par unité (0,20 = 20 %), `r` et `q` continus.
  Les Grecques sont des dérivées partielles brutes : `theta = ∂V/∂t` **par an**, `vega = ∂V/∂σ` par
  unité de vol, `rho = ∂V/∂r` par unité de taux. La mise à l'échelle (par jour, par point de vol)
  se fait uniquement à l'affichage.
* Convention de couverture : le teneur de livre est **vendeur** de l'option et long Δ actions.
  P&L > 0 = gain du vendeur couvert. Le pas de temps est déduit de la grille : `dt = T / n_steps`.
* Aléatoire : `np.random.default_rng(seed)` uniquement, seed explicite à chaque appel.
* Données : `data/sample/` est figé (voir `SOURCE.md`, `SHA256SUMS`) ; le code ne l'écrase jamais.
* Fichiers générés (PNG, notebook) : métadonnées auteur « Tancrède Geli » et titre explicite.

## Git

* Auteur des commits : `Tancrède Geli <282603140+tancredeg@users.noreply.github.com>`.
* Lancer les tests avant tout push ; s'ils échouent, pousser sur `fix/<date>` et le signaler.
* Jamais de force push, de réécriture d'historique ni de modification des réglages du dépôt.
* Aucune donnée liée au stage en cours : données publiques uniquement.
