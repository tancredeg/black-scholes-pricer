# AUDIT — Phase 0 · `black-scholes-pricer`

Audit du dépôt public `tancredeg/black-scholes-pricer`, commit **`6339479`** (2026-06-02, branche `main`, aucun tag).
Chaque chiffre de ce document est produit par `docs/audit/audit_phase0.py` (identifiants `A01`…`A21`, `F01`) ; sortie brute dans `docs/audit/audit_phase0_output.txt`.

> Le script cible l'API **de la version auditée** (`bs_pricer.py`, commit `6339479`) et ne s'exécute pas contre le package actuel. Pour le rejouer :
>
> ```bash
> git worktree add ../bsp-6339479 6339479
> python docs/audit/audit_phase0.py --repo ../bsp-6339479 --spy data/sample/SPY_2020_2024.csv   # ~15 s
> ```
>
> A21 exigeait un second téléchargement Yahoo (`--spy-fresh`) ; `data/sample/SPY_2020_2024.csv` **est** ce second téléchargement. Les chiffres de volatilité (A16) sont identiques avec les deux fichiers : seuls les niveaux de prix diffèrent.
>
> Suivi des constats et décisions D1–D6 : `docs/PROGRESS.md`.

Environnement de vérification : Python 3.12.3, numpy 2.5.4, scipy 1.18.1, pandas 3.0.6, pytest 9.1.1.

---

## 0. Périmètre, écarts de version, échelle de gravité

**Lu intégralement** : `bs_pricer.py` (748 l.), `tests/test_bs_pricer.py`, `black_scholes.ipynb` (46 cellules, code + markdown + sorties), `README.md`, `CHANGELOG.md`, `NOTEBOOK_PATCHES.md`, CI, `requirements.txt`, `conftest.py`, `.gitignore`, `.gitattributes`.

**Exécuté** : doctests, pytest + couverture, ruff, mypy, ré-exécution complète du notebook (11,6 s), 21 vérifications numériques indépendantes.

**Écart de version à trancher avant la Phase 1.** Le dépôt public est figé au 2026-06-02 (8 commits). Le correctif dividendes et les corrections README/LaTeX produits lors d'un audit ultérieur ne sont **pas** sur GitHub. Le fichier `bs_pricer.cpython-312.pyc` stocké dans le projet Claude correspond à une version encore antérieure (sans `q`, sans solveur d'IV). Cet audit porte sur `6339479`.

| Gravité | Définition |
|---|---|
| **Critique** | Résultat faux produit silencieusement par une fonction cœur, ou démonstration centrale du repo invalide. |
| **Majeur** | Échec silencieux sur des entrées plausibles, chiffre faux visible par un recruteur, ou trou de reproductibilité. |
| **Mineur** | Incohérence documentaire, style, typage, formulation. |

---

## 1. Synthèse

| ID | Gravité | Sujet | Preuve | Phase |
|---|---|---|---|---|
| C1 | Critique | Convention theta mal documentée → P&L theta 252× trop petit (Section 7) | A03 | 1 |
| C2 | Critique | Décomposition gamma/theta sans terme de financement ; figure EDP de signe faux | A04, A05 | 1 |
| C3 | Critique | Limite σ → 0 fausse, prix **négatifs** | A01 | 1 |
| C4 | Critique | Moteur de couverture ignore les dividendes pour `q > 0` (biais t = −31,7) | A02 | 1 |
| M1 | Majeur | Aucune garde `N_steps·dt = T_total` (biais silencieux +1,10 USD) | A14 | 1 |
| M2 | Majeur | Solveur d'IV : renvoie σ = 1e-6 pour un prix généré à σ = 0,19 | A12 | 1 |
| M3 | Majeur | Aucune validation d'entrées (σ < 0, T < 0, S = 0, σ = 0 vectoriel) | A13 | 1 |
| M4 | Majeur | Chiffres faux dans README / notebook (Grecques q, N(d2), vega, dates) | A06, A08, A16, A18 | 5 |
| M5 | Majeur | Sur-affirmation « empirical analysis on real SPY data » ; scénario Section 8 irréaliste | lecture | 4–5 |
| M6 | Majeur | Tests du cahier des charges absents ; test d'échelle non discriminant | A10, cov. | 2 |
| M7 | Majeur | Données non figées, cache écrasé à chaque run, traces sandbox dans le notebook | A21 | 1 |
| M8 | Majeur | Packaging/CI : pas de `pyproject`, deps non épinglées, ni lint ni mypy en CI, pas de licence | outils | 1, 3 |
| M9 | Majeur | `NOTEBOOK_PATCHES.md` contient un résultat inventé (put MC 7,2151 ; réel 7,1480) | A20 | 1 |
| m1–m9 | Mineur | Formule LaTeX vol réalisée, delta put expiré, tolérances, justification d'un test, etc. | A07, A15, A19… | 1–5 |

---

## 2. Constats détaillés

### C1 — Unité de theta : le P&L theta de la Section 7 est divisé par 252 une fois de trop

- **Où** : docstring module `bs_pricer.py` l. 13–14 et `bs_greeks` l. 311–313 (« `theta_pnl = theta * dt` with `dt = 1/252` ») ; notebook Section 7 (cellules d'index 36–39).
- **Constat** : `theta` est stocké par jour de bourse (Θ annuel ÷ 252, valeur correcte). Le P&L theta sur un pas d'un jour vaut donc `theta × 1`, ou `theta × (dt·252)`, et non `theta × dt`. Le notebook applique la règle documentée.
- **Preuve (A03, 20 000 trajectoires, call ATM 30 j)** : somme gamma par trajectoire 4,013 ; somme theta selon la convention du notebook **−0,020** ; somme theta correcte **−5,033**.
- **Conséquence** : la figure « Gamma Gain vs Theta Decay » et le paragraphe « Desk Answer » démontrent le contraire de l'arbitrage gamma/theta. C'est la question d'entretien que le repo prétend illustrer.
- **Correction** : code inchangé ; corriger les docstrings et le notebook (`theta_pnl = theta_jour × n_jours`). Ajouter un test qui vérifie la somme sur trajectoires (voir C2).

### C2 — Décomposition incomplète : le terme de financement manque ; figure EDP de signe faux

- **Constat 1 (A04)** : même en unités correctes, gamma + theta ne s'annulent pas en espérance. Pour un call long couvert en delta et financé au taux r, l'identité exacte est :

```math
\delta\Pi_t \;=\; \Theta\,\delta t \;+\; \tfrac12\,\Gamma\,(\delta S)^2 \;-\; r\,(V-\Delta S)\,\delta t
\;\;\overset{\text{EDP}}{=}\;\; \tfrac12\,\Gamma S^2\!\left[\Big(\tfrac{\delta S}{S}\Big)^{2}-\sigma^2\,\delta t\right]
```

  (Θ annuel = ∂V/∂t, δt en années.) Sur 20 000 trajectoires : gamma + theta = **−1,020** (≈ 11 % de la prime) ; avec le terme de financement (+1,019) le total vaut **−0,0016 ± 0,0087**.
- **Constat 2 (A05)** : la cellule « PDE Relationship Visualization » (index 13) trace `theta + (−½σ²S²Γ)`, c'est-à-dire Θ − ½σ²S²Γ, légendé « residual ~ 0 ». À l'ATM cette courbe vaut **−75,6 USD/an**. La somme correcte Θ + ½σ²S²Γ vaut −8,06 = r(C − SΔ) : non nulle (19 % de Θ). La cellule suivante annonce « Theta ≈ −½σ²S²Γ holds to 1.5e-14 », alors que 1,5e-14 est le résidu de l'EDP complète.
- **Correction** : réécrire la Section 7 avec les trois termes ; tracer Θ et −½σ²S²Γ − r(SΔ − V) ; le texte du markdown index 11 (« rSΔ ≈ rC pour un ATM ») est faux et doit disparaître.

### C3 — Limite σ → 0 : mauvais critère de moneyness, prix négatifs

- **Où** : `_d1_d2` l. 101–105 (scalaire) et l. 107–114 (vectoriel, `denom > 1e-14`).
- **Constat** : la branche dégénérée choisit ±∞ selon `S ≥ K`. La bonne limite dépend du forward, `S·e^{−qT}` contre `K·e^{−rT}` :

```math
\lim_{\sigma\to 0^+} C \;=\; \max\!\big(S e^{-qT}-K e^{-rT},\,0\big),
\qquad
\lim_{\sigma\to 0^+} P \;=\; \max\!\big(K e^{-rT}-S e^{-qT},\,0\big)
```

- **Preuve (A01, T = 1)** :

| Cas | Limite exacte | σ = 1e-8 | σ = 1e-10 | σ = 0 |
|---|---|---|---|---|
| Call S = 305, K = 310, r = 5 % | 10,119 | 10,119 | **0** | **0** |
| Call S = K = 310, r = 0, q = 3 % | 0 | 0 | **−9,162** | **−9,162** |
| Put S = 305, K = 310, r = 5 % | 0 | 0 | **−10,119** | **−10,119** |

  Discontinuité entre 1e-8 et 1e-10, et violation des bornes d'arbitrage. Le « Test 3 » du notebook passe parce qu'il utilise σ = 1e-6 (> seuil) à S = K.
- **Correction** : critère `ln(S/K) + (r − q)T ≥ 0`, ou renvoyer directement la valeur intrinsèque actualisée sur le forward. Les tests σ → 0 du cahier des charges (Phase 2) échoueraient sur le code actuel.

### C4 — Dividendes non crédités dans la couverture (`q > 0`)

- **Où** : `delta_hedge_single_path` l. 584–594, `run_hedging_experiment_vec` l. 682–693.
- **Constat** : le prix et le delta intègrent `q`, mais la jambe actions ne reçoit pas `q·Δ·S·dt`. Le portefeuille n'est plus autofinançant au sens du modèle.
- **Preuve (A02, 20 000 trajectoires, q = 1,3 %)** : erreur moyenne du moteur **−0,286 ± 0,009** (t = −31,7, ≈ 3,3 % de la prime à q = 1,3 %, soit 8,774). Avec dividendes crédités (moteur de référence du script, identique au moteur du repo à 0 près pour q = 0) : **−0,009 ± 0,009** (t = −1,06).
- **Correction** : `cash += Δ_{k−1}·q·dt·½(S_{k−1}+S_k)` à chaque pas, scalaire et vectoriel ; test de non-régression |t| < 4.

### M1 — Pas de garde de cohérence temporelle

- **Où** : signature `dt=1/252` par défaut, l. 543–594 et 641–700.
- **Preuve (A14)** : 30 pas sur T = 60/252 avec `dt` par défaut → moyenne **+1,096**, RMSE 2,79, sans erreur. Avec `dt = 2/252` : moyenne +0,005, RMSE 1,82.
- **Correction** : supprimer `dt` et le déduire de `T_total / N_steps`, ou lever `ValueError` si `|N·dt − T| > 1e-12`.

### M2 — Solveur d'IV : échec silencieux, comportement incohérent aux bornes

- **Preuve (A12)** : call S = 420, K = 310, T = 5 j, prix généré à σ = 0,19 → IV renvoyée **1e-6**, sans avertissement. Prix = borne inférieure + 1e-9 → IV 0,351 (problème mal conditionné). Prix = borne − 1e-11 → accepté par le test d'arbitrage (tolérance 1e-10) puis « Root not bracketed ». Le README présente le renvoi de la borne basse comme un comportement normal (l. 236–238).
- **Correction** : lever une erreur dédiée (ou renvoyer `nan` avec drapeau) quand la valeur temps est inférieure à `vega × xtol` ; harmoniser la tolérance des bornes.

### M3 — Validation d'entrées absente

- **Preuve (A13)** : σ = −0,2 → prix 0 ; T = −1 → valeur intrinsèque ; S = 0 → gamma `nan` ; `bs_greeks_vec` avec un σ nul dans le vecteur → gamma `nan`. Les limites S → 0 (put → K·e^{−rT}) et S → ∞ sont correctes.
- **Correction** : `ValueError` pour S ≤ 0, K ≤ 0, σ < 0, T < 0 ; traitement explicite de σ = 0 dans la voie vectorielle.

### M4 — Chiffres faux visibles

| Emplacement | Affiché | Valeur calculée | Preuve |
|---|---|---|---|
| README l. 69, Grecques `q = 0,013` | delta 0,5446 · theta −0,1666 · rho 19,19 · vega 42,30 · gamma 0,01946 | 0,5390 · −0,1570 · 18,85 · 42,39 · 0,019503 | A08 |
| Notebook Section 4, texte | N(d2) sous Q ≈ 48 %, sous P ≈ 57 % | 52,3 % et 58,8 % | A06 |
| Notebook Section 4, f-string `{prob_q:.1f}%` | « probability 0.5% » | 52,3 % | A06 |
| Notebook Section 3, vega | « 1-point vol move changes price by $42.3464 » | 0,42 USD par point de vol | A18 |
| Notebook Section 1, texte | divergence 30 j/252 j maximale en mars 2020 | impossible : la vol 252 j n'existe qu'à partir du 2020-12-31 ; max 24,0 % le 2021-01-14 | A16 |
| Notebook Section 1, texte | vol 30 j de « 6–8 % » en S1 2021 | minimum S1 2021 : 8,4 % | A16 |

### M5 — Portée empirique sur-affirmée, scénario Section 8 irréaliste

- Toutes les expériences de couverture tournent sur des trajectoires GBM. SPY sert uniquement à mesurer des régimes de vol réalisée. « Includes empirical analysis on real SPY data » et « Discrete delta-hedging on real SPY data » (notes de projet) sont donc inexacts.
- Section 8 : σ_r = pic ex post de la vol 30 j (82,7 %, 2020-04-08) appliqué comme constant sur 30 jours, contre σ_h = 19 %. Aucun desk ne vendait de la vol à 19 % en avril 2020. *Hypothèse à vérifier sur FRED `VIXCLS`* : VIX autour de 40–50 % à cette date. En entretien, la question « quelle vol implicite as-tu utilisée ? » met ce scénario en défaut.
- **Correction** : présenter le Monte Carlo comme expérience contrôlée (σ_r connue), SPY comme contexte. Option : backtest sur trajectoires SPY réelles avec VIX comme σ_h (voir D3).

### M6 — Couverture du cahier des charges de tests

- **Couverture actuelle** : 89 % des lignes ; non couverts : `rolling_realized_vol`, branche Grecques à l'expiration, branche σ dégénérée scalaire.
- **Absents** : bornes d'arbitrage du call, limites σ → 0 / S → 0 / S → ∞, Grecques contre différences finies, arbre CRR (inexistant dans le code), ordre de convergence MC, régression sur données figées.
- **Test non discriminant** : `test_rmse_scales_as_sqrt_dt` compare deux échantillons indépendants avec la bande [1,13 ; 1,84]. Une pente de −0,25 (ratio 1,19) ou de −0,75 (ratio 1,68) passerait aussi (A10).
- **Comptage** : les « 2060 tests » sont des paramétrisations de 5 propriétés (480 cas de parité, 960 cas d'EDP). À présenter comme tel.

### M7 — Reproductibilité des données et du notebook

- `data/SPY.csv` est ignoré par git ; le notebook re-télécharge via yfinance puis **écrase le cache**.
- **Dérive mesurée (A21)** : un téléchargement du 2026-10-12 diffère du fichier joint d'un facteur uniforme 0,99496 (−0,50 % sur les niveaux). Les rendements log sont identiques à 9e-7 près. Les vols sont donc stables, mais les niveaux de prix affichés changent (`Price range : $204.94 - $597.11` devient `$203.91 - $594.10` à la ré-exécution).
- Le notebook exécuté et `black_scholes.html` (1,8 Mo) contiennent « HTTP Error 403: Host not in allowlist ». Ces traces d'exécution en sandbox sont visibles par un recruteur.
- `end="2024-12-31"` est exclusif : la série s'arrête au 2024-12-30.

### M8 — Packaging et intégration continue

- Pas de `pyproject.toml`. `requirements.txt` en `>=` mélange dépendances cœur et notebook (yfinance, jupyter). Module unique de 748 lignes, `conftest.py` vide comme contournement de chemin.
- **ruff** : 8 E501, 2 fichiers non formatés. **mypy** : 5 erreurs (`str` passé à un `Literal`, `.rolling` sur `ndarray`).
- CI : doctest + pytest seulement, ni lint, ni typage, ni couverture, ni badge. Statut réel des runs non vérifiable sans accès API. Pas de `LICENSE`, pas de tag.
- **Compatibilité** : la suite passe avec numpy 2.5 / pandas 3.0 (2060 réussis, 120 ignorés, 10 s).

### M9 — Résultat inventé dans `NOTEBOOK_PATCHES.md`

- Sortie « attendue » : `PUT MC: 7.2151`. Avec seed 42 et 200 000 trajectoires, la valeur réelle est **7,1480** (A20), identique au notebook exécuté.
- Le fichier documente un processus (« apply manually ») et non le projet. Ce chiffre non issu d'une exécution viole la règle « ne jamais inventer un résultat ».
- **Correction** : supprimer le fichier ; le CHANGELOG suffit.

### Mineurs

| ID | Constat | Preuve |
|---|---|---|
| m1 | Formule LaTeX de la vol réalisée (notebook Section 1) : le terme de moyenne vaut `(Σr/(w−1))²` au lieu de `w·r̄²/(w−1)`. L'implémentation (`std(ddof=1)`) est correcte. Écart maximal sur 2020–2024 : 0,65 % relatif. | A07 |
| m2 | README l. 233 : delta du put expiré à S = K = −1 ; code : 0. Docstring l. 16 : « rho … vol point ». README : « 35 doctests » (36 en réalité), « 570x » (225× mesuré ici sur 300 trajectoires, dépend de la machine). | A15, A17 |
| m3 | Tolérances EDP : test < 1e-9, README < 1e-10, maximum observé 1,3e-13. Passer à une tolérance relative au prix. | A19 |
| m4 | `test_hedging_call_put_symmetry` : la justification « symétrie du gamma » est fausse. L'égalité tient aussi pour σ_h ≠ σ_r (écart 4,2e-13) : c'est la parité call-put du moteur, ce qui reste un bon test. | A11 |
| m5 | Expérience 1 : la ligne « buy-and-hold » porte sur une option à 21 j, non comparable aux autres ; le ratio contre √21 n'a pas de sens. Expérience 3 : corrélation calculée sur n = 3. Texte « OTM cheapest to hedge » contre RMSE/C0 = 100,5 %. | lecture |
| m6 | `_d1_d2` renvoie des tableaux 0-d dans la voie dégénérée et des `float` sinon ; `option_type` accepte « CALL » malgré le `Literal`. | lecture |
| m7 | Notebook Section 2 : hypothèse « No dividends » et formules sans `q`, alors que le code gère `q`. | lecture |
| m8 | Pic de gamma annoncé « S/K ≈ 1,00 » ; valeur imprimée 0,9877, conforme à `S* = K·exp(−(r − q + 3σ²/2)T)`. Donner la formule plutôt que l'approximation. | lecture |
| m9 | Le `.pyc` du projet Claude est obsolète : le remplacer par la source ou le retirer pour éviter d'auditer une ancienne version. | — |

---

## 3. Ce qui est vérifié et solide

- **Parité call-put** sur la grille de tests ; **résidu EDP** maximal 1,3e-13 (960 combinaisons, calls et puts, avec q) (A19).
- **Grecques analytiques contre différences finies centrées** (240 combinaisons) : erreur absolue maximale delta 2,2e-5, gamma 1,3e-5 (pas `h = 1e-3·S`), theta 3,9e-9, vega 2,5e-7, rho 7,3e-8 (A09). Base des tolérances de la Phase 2.
- **Monte Carlo sous Q contre BS** à moins de 4 erreurs-types ; voies vectorielle et scalaire identiques (écart 0) ; parité du moteur de couverture.
- **Échelle de l'erreur de réplication (A10)**, aléas communs, 20 000 trajectoires, N de 3 à 480 : pente log-log **−0,484** (tous N), **−0,487** (N ≥ 10). Le ratio à l'approximation de Kamal–Derman `sd ≈ √(π/4)·σ·𝒱/√N` passe de 0,91 (N = 3) à 0,99 (N ≥ 240).
- **Vol mal spécifiée (A11)** : régression trajectoire par trajectoire du P&L sur `½Σ e^{r(T−t)} Γ_h S² (σ_h²δt − (δS/S)²)` → pente 0,98 / 1,03, R² 0,94 / 0,96 (σ_h = 0,15 / 0,23). Moyennes : −1,692 contre −1,695 prédit ; +1,687 contre +1,688.
- **Notebook** : ré-exécution complète en 11,6 s ; sorties identiques sauf niveaux de prix (M7) et un écart de 1e-4 sur une ligne de la Section 8.

---

## 4. Décisions à prendre avant la Phase 1

| # | Décision | Recommandation | Raison |
|---|---|---|---|
| D1 | Repartir de `6339479` ou d'une copie locale plus récente non poussée ? | Pousser ou abandonner la copie locale **avant** la Phase 1. | Évite deux branches divergentes. |
| D2 | Versionner `data/sample/SPY_2020_2024.csv` (source Yahoo) ? | Oui, avec date de téléchargement, SHA256 et mention de la source. | Seule façon d'avoir 2 commandes hors ligne et sans dérive. *Risque : les CGU Yahoo limitent la redistribution ; alternative Stooq, à vérifier.* |
| D3 | Contrainte « données réelles uniquement » contre Monte Carlo et smile synthétique | Assumer : Monte Carlo = expériences contrôlées (σ_r connue), SPY = contexte. Option v1.1 : backtest sur trajectoires SPY avec VIX (FRED) comme σ_h. | Les résultats 1–3 exigent une σ_r connue. |
| D4 | Critère de « fréquence optimale » avec coûts | Frontière (coût moyen ; écart-type) paramétrée par N, plus optimum du RMSE total. | « Optimal » dépend de l'aversion au risque ; la frontière ne tranche pas à la place du lecteur. |
| D5 | Résultat 4 (sticky-strike / sticky-delta) dans la v1.0 ? | Non, en v1.1. | Budget de 6–8 h/semaine ; 3 résultats solides valent mieux que 4 fragiles. |
| D6 | Nom du package | `bspricer` | Continuité avec `bs_pricer`. |

---

## 5. Plan des phases 1–6, adapté à ce repo

*Estimations de durée : jugement, non mesurées. Total ≈ 16–18 h, soit 2,5 à 3 semaines au rythme prévu.*

### Phase 1 — Structure et correctifs bloquants (≈ 3 h)

```
src/bspricer/{__init__, pricing, greeks, implied_vol, trees, montecarlo, hedging, vol}.py
tests/   notebooks/demo.ipynb   data/sample/SPY_2020_2024.csv (+ .sha256, SOURCE.md)
scripts/run_results.py   results/   docs/
pyproject.toml   CLAUDE.md   AUDIT.md   PROGRESS.md   LICENSE (MIT)   CHANGELOG.md
```

1. `pyproject.toml` : dépendances cœur épinglées (numpy, scipy, pandas) ; extras `[dev]` (pytest, pytest-cov, ruff, mypy) et `[notebook]` (matplotlib, jupyter). Installation par `pip install -e ".[dev]"`.
2. Découper `bs_pricer.py` sans changer les signatures publiques. **Un seul** moteur de couverture vectorisé ; la version scalaire passe dans les tests comme oracle.
3. Correctifs avec un test de non-régression minimal chacun (un commit par correctif) : **C3** (forward), **C4** (dividendes), **M1** (dt déduit de `T/N`), **M2** (IV mal conditionnée), **M3** (validation).
4. Moteur de couverture étendu : coût proportionnel `k`, vol de couverture ≠ vol de simulation, sortie détaillée par pas (gamma, theta, financement, coûts). Prépare les Phases 2 et 4.
5. Données : figer le CSV, ne plus jamais l'écraser, yfinance en script séparé `scripts/download_spy.py`.
6. Supprimer `NOTEBOOK_PATCHES.md`, `black_scholes.html` et `conftest.py` ; corriger les docstrings (C1, m2).

**Sortie** : `pip install -e .` fonctionne, la suite existante passe, les 5 correctifs sont testés, aucun chemin absolu.

### Phase 2 — Tests de cohérence financière (≈ 4 h)

| Bloc | Test | Tolérance documentée |
|---|---|---|
| Identités | Parité call-put avec q | 1e-10·max(1, S) |
| Bornes | `max(S·e^{−qT} − K·e^{−rT}, 0) ≤ C ≤ S·e^{−qT}` ; C décroissant et convexe en K ; C croissant en σ | exact (≥ −1e-12) |
| Limites | T → 0 : payoff ; σ → 0 : intrinsèque sur forward, y compris q > r et puts ; S → 0 : C → 0, P → K·e^{−rT} ; S → ∞ : C − (S·e^{−qT} − K·e^{−rT}) → 0 | 1e-10 |
| Grecques | Analytiques contre différences finies centrées, grille A09 | delta/gamma `atol 5e-5` (h = 1e-3·S) ; theta/vega/rho `atol 1e-6`. Justification : erreur de troncature O(h²), maxima observés en A09 |
| EDP | Résidu relatif au prix | 1e-10 |
| CRR | Erreur contre BS, N de 25 à 3 200, N pairs et impairs séparés ; pente log-log | pente ∈ [−1,2 ; −0,8] (ordre 1 attendu, oscillant) |
| MC | RMSE sur M seeds par N ; pente log-log | pente ∈ [−0,55 ; −0,45] ; \|MC − BS\| ≤ 4 SE |
| Couverture | Garde dt ; q > 0 sans biais (\|t\| < 4) ; parité moteur ; décomposition gamma + theta + financement = P&L total à O(δt) ; pente de réplication par régression sur ≥ 6 valeurs de N avec aléas communs | pente ∈ [−0,55 ; −0,45] |
| IV | Aller-retour ; erreur levée si mal conditionnée | 1e-8 |
| Régression | Chiffres de vol SPY figés (82,69 % le 2020-04-08 ; 6,84 % le 2024-07-16 ; divergence 24,0 % le 2021-01-14) | 1e-10 |

**Sortie** : couverture ≥ 80 % sur `src/bspricer` (objectif 90 %) ; suite < 30 s ; tests lents (convergence) marqués `@pytest.mark.slow`.

### Phase 3 — Qualité et intégration continue (≈ 1,5 h)

- ruff (lint + format) avec règles `E, F, W, I, B, UP, NPY` ; mypy strict sur `src/` (corriger les 5 erreurs).
- GitHub Actions : job lint/type, matrice de tests 3.10–3.12, rapport de couverture, badges dans le README. Pre-commit optionnel.

### Phase 4 — Résultats chiffrés (≈ 4 h)

`scripts/run_results.py` régénère `results/*.csv` et `results/figures/*.png` à partir d'une seed unique.

| # | Résultat | Méthode | Préliminaire (audit, à régénérer) |
|---|---|---|---|
| R1 | Erreur de réplication contre N | 20 000 trajectoires, aléas communs (sous-échantillonnage d'une grille fine), N de 3 à 960, pente log-log avec IC bootstrap, superposition de Kamal–Derman | pente −0,48 ; ratio à Kamal–Derman 0,91 → 0,99 (A10) |
| R2 | Coûts proportionnels, fréquence optimale | k ∈ {1, 5, 10, 25} bp ; frontière (coût moyen ; écart-type) ; N* minimisant le RMSE total ; vérifier N* ∝ 1/k et comparer à Leland (1985) | N* = 120 (5 bp), 60 (10 bp), > 480 (1 bp) (F01) → la grille de N doit aller en intrajournalier |
| R3 | Vol mal spécifiée | P&L moyen contre σ_h ; régression trajectoire par trajectoire sur la prédiction dollar-gamma (pente ≈ 1, R² ≈ 0,95) ; dépendance au chemin (P&L contre dollar-gamma moyen à σ_r fixée) ; couverture à σ_h contre σ_r (Ahmad–Wilmott 2005) | A11 |
| R4 | *(v1.1)* Sticky-strike contre sticky-delta | Smile synthétique paramétrique ; delta corrigé `Δ_BS + 𝒱·∂σ/∂S` | — |

Chaque résultat est interprété en une phrase. La figure de vol réalisée SPY est conservée comme contexte.

### Phase 5 — README (≈ 2 h)

Ordre imposé ; tous les chiffres sont lus depuis `results/*.csv` (aucune saisie manuelle). Section « Limits » : vol constante, GBM, pas de sauts, coûts linéaires, données de clôture uniquement. Supprimer les formulations promotionnelles (« desk-level command », « rigorously implemented »).

**Références** : Black–Scholes (1973), Merton (1973), Cox–Ross–Rubinstein (1979), Leland (1985), Kamal–Derman (1999, *Risk*), Bertsimas–Kogan–Lo (2000, *JFE*), Ahmad–Wilmott (2005, *Wilmott*), Hull (2022). *Références bibliographiques exactes à confirmer en Phase 5.*

### Phase 6 — Livraison (≈ 1,5 h)

- `notebooks/demo.ipynb` exécuté de bout en bout en moins de 2 min sur `data/sample`, sans réseau. Base actuelle : 11,6 s, téléchargement compris.
- Tag `v1.0.0` et notes de version ; repo épinglé.
- Brouillon LinkedIn en 5 lignes, chiffres issus de `results/`.
- Décider si `AUDIT.md` reste public : le déplacer dans `docs/` et le traduire, ou le retirer.

---

## 6. Fichiers livrés avec cet audit

| Fichier | Rôle |
|---|---|
| `audit_phase0.py` | Toutes les vérifications A01–A21 et F01, déterministe |
| `audit_phase0_output.txt` | Sortie brute de l'exécution |
| `audit_phase0_results.json` | Mêmes résultats, en format machine |
| `data_snapshots/SPY_yahoo_dl_2026-10-12.csv` | Téléchargement Yahoo utilisé pour A21 |
