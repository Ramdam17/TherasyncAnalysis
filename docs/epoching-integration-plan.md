# Plan d'Intégration de l'Epoching dans le Preprocessing

**Date**: 2025-11-12  
**Objectif**: Intégrer le calcul des ID d'epochs directement pendant le preprocessing pour éliminer la redondance de données.

---

## 1. État des Lieux

### 1.1 Architecture Actuelle (Mode "separate")

```
Étape 1: Preprocessing
  Input:  data/raw/sub-xxx/ses-yyy/{modality}/
  Output: data/derivatives/preprocessing/sub-xxx/ses-yyy/{modality}/
          └── *_desc-processed_recording-*.tsv
          └── *_desc-rrintervals_physio.tsv
          └── *_desc-*-metrics_*.tsv (non épochable)

Étape 2: Epoching (séparé)
  Input:  data/derivatives/preprocessing/
  Output: data/derivatives/epoched/sub-xxx/ses-yyy/{modality}/
          └── Copie des fichiers + ajout colonnes epoch_id_*
```

**Problème**: Les données sont **dupliquées** (preprocessing/ + epoched/)

### 1.2 Fichiers Existants

#### Scripts
- `scripts/physio/epoching/epoch_all_signals.py` (198 lignes)
  - CLI pour epoching batch
  - Lit preprocessing/, écrit dans epoched/

#### Modules
- `src/physio/epoching/epoch_assigner.py` (279 lignes)
  - Classe `EpochAssigner`
  - Méthodes: `assign_fixed_epochs()`, `assign_nsplit_epochs()`, `assign_sliding_epochs()`
  - **Logic hard-codée**: "if task == 'restingstate' → epoch_id=0"

- `src/physio/epoching/epoch_bids_writer.py` (252 lignes)
  - Classe `EpochBIDSWriter`
  - Lit preprocessing/, ajoute colonnes, écrit epoched/
  - Gestion include/exclude patterns (obsolète en mode preprocessing)

#### Fichiers de Preprocessing (BVP)
- `src/physio/preprocessing/bvp_bids_writer.py`
  - `save_rr_intervals()` : sauvegarde RR intervals
  - **À MODIFIER**: ajouter colonnes epoch_id_* ici

- `src/physio/preprocessing/base_bids_writer.py`
  - Classe de base pour writers BIDS
  - **À MODIFIER**: possiblement ajouter méthode helper pour epoching

### 1.3 Configuration

**Avant** (config.yaml - ancien):
```yaml
epoching:
  enabled: true
  methods:
    fixed: {duration: 30, overlap: 5}  # Paramètres globaux
  include: ["*_desc-processed_*.tsv"]   # Patterns de fichiers
  output: {base_dir: "epoched"}         # Répertoire séparé
```

**Après** (config.yaml - nouveau):
```yaml
epoching:
  enabled: true
  mode: "preprocessing"  # Nouveau: inline vs separate
  methods:
    fixed:
      restingstate: {duration: 30, overlap: 5}
      therapy: {duration: 30, overlap: 5}
    nsplit:
      restingstate: {n_epochs: 1}
      therapy: {n_epochs: 120}
    sliding:
      restingstate: {duration: 30, step: 5}
      therapy: {duration: 30, step: 5}
  # Plus de include/exclude/output (obsolète en mode preprocessing)
```

### 1.4 Données Actuelles

**État**: `data/derivatives/` est **VIDE** → pas de migration nécessaire ✅

---

## 2. Analyse des Dépendances

### 2.1 Qui utilise `derivatives/epoched/` ?

**Modules DPPA** (lecture seule):
- `src/physio/dppa/poincare_calculator.py` : lit RR intervals épochés
- `src/physio/dppa/epoch_animator.py` : lit RR intervals pour animation
- `scripts/physio/dppa/compute_poincare.py` : cherche dans epoched/

**Impact**: Ces modules devront lire depuis `preprocessing/` au lieu de `epoched/`

### 2.2 Fichiers Épochables vs Non-Épochables

**✅ ÉPOCHABLE** (signaux bruts):
- `*_desc-processed_recording-bvp_physio.tsv`
- `*_desc-processed_recording-eda_physio.tsv`
- `*_desc-rrintervals_physio.tsv`

**❌ NON ÉPOCHABLE** (métriques agrégées):
- `*_desc-hrv-metrics_*.tsv` (HRV_SDNN, HRV_RMSSD, etc.)
- `*_desc-eda-metrics_*.tsv`
- `*_desc-*-summary_*.json`

### 2.3 Format des Colonnes Epoch

**Actuel** (auto-généré dans le code):
```
epoch_id_fixed            : [0, 1, 2, ...]
epoch_id_nsplit120        : [0, 1, 2, ..., 119]
epoch_id_sliding_duration30s_step5s : [0, 1, 2, ...]
```

**Type**: `List[int]` en JSON → chaque RR interval peut appartenir à plusieurs époques

---

## 3. Stratégie de Migration

### 3.1 Approche Progressive (6 Phases)

#### Phase 1: Préparation (30 min)
- [x] Analyser architecture existante
- [x] Documenter dépendances
- [x] Créer plan de migration

#### Phase 2: Adapter `EpochAssigner` (1h)
- [ ] Modifier pour accepter paramètres par moment
- [ ] Supprimer logique hard-codée "restingstate → epoch_id=0"
- [ ] Lire nouvelle structure config (methods.fixed.restingstate, etc.)
- [ ] Tests: vérifier que les 3 méthodes fonctionnent avec les 2 moments

#### Phase 3: Modifier Writers de Preprocessing (2h)
- [ ] `base_bids_writer.py`: ajouter méthode `add_epoch_columns(df, time, task)`
- [ ] `bvp_bids_writer.py`: modifier `save_rr_intervals()` pour ajouter epochs
- [ ] Logique: si `epoching.enabled && mode=="preprocessing"` → ajouter colonnes
- [ ] Tests: vérifier que fichiers ont colonnes epoch_id_*

#### Phase 4: Adapter Modules DPPA (1h)
- [ ] `poincare_calculator.py`: chercher dans preprocessing/ au lieu de epoched/
- [ ] `epoch_animator.py`: idem
- [ ] `compute_poincare.py`: idem
- [ ] Tests: vérifier que DPPA fonctionne avec nouveaux chemins

#### Phase 5: Nettoyer Code Obsolète (30 min)
- [ ] **Garder**: `epoch_assigner.py` (réutilisé dans preprocessing)
- [ ] **Deprecate/Supprimer**: 
  - `epoch_bids_writer.py` (mode "separate" legacy)
  - `epoch_all_signals.py` (script séparé obsolète)
- [ ] Ajouter warnings de deprecation si on garde pour compatibilité

#### Phase 6: Tests & Documentation (1h)
- [ ] Tests end-to-end: preprocessing → DPPA
- [ ] Vérifier génération correcte des colonnes epoch_id_*
- [ ] Mettre à jour documentation (README, QUICKREF)
- [ ] Commit final

### 3.2 Ordre d'Exécution

**IMPORTANT**: Ne **PAS** tout faire d'un coup !

1. **Commit 1**: Phase 2 (EpochAssigner adapté)
2. **Commit 2**: Phase 3 (Writers preprocessing)
3. **Commit 3**: Phase 4 (DPPA adapté)
4. **Commit 4**: Phase 5 (Nettoyage)
5. **Commit 5**: Phase 6 (Tests + docs)

---

## 4. Décisions de Design

### 4.1 Questions Résolues

**Q1: Multiple méthodes simultanément ?**
→ **OUI**, toutes les méthodes enabled ajoutent leur colonne

**Q2: Backward compatibility avec mode "separate" ?**
→ **NON** pour l'instant, on supprime le code obsolète
→ Si besoin futur: garder `epoch_bids_writer.py` avec warning

**Q3: Appliquer à EDA et HR aussi ?**
→ **OUI**, même logique pour toutes les modalités

**Q4: Hard-coded "restingstate → epoch_id=0" ?**
→ **NON**, lire params depuis config par moment

### 4.2 Choix Techniques

**Mode par défaut**: `mode: "preprocessing"`
- Pas de duplication de données
- Colonnes epoch_id_* ajoutées inline

**Noms de colonnes**: Auto-générés (déjà existant)
- `epoch_id_fixed`
- `epoch_id_nsplit{N}`
- `epoch_id_sliding_duration{X}s_step{Y}s`

**Format**: `List[int]` en JSON (un RR peut être dans plusieurs époques)

---

## 5. Risques & Mitigations

### 5.1 Risques Identifiés

1. **Casser DPPA existant**
   - Mitigation: Adapter chemins avant de supprimer ancien code

2. **Logique par moment mal implémentée**
   - Mitigation: Tests unitaires pour chaque méthode × moment

3. **Config mal lue**
   - Mitigation: Valider structure config avec tests

4. **Code mort laissé**
   - Mitigation: grep/search pour trouver tous les usages avant suppression

### 5.2 Tests de Validation

Pour chaque phase:
- [ ] Tests unitaires passent
- [ ] Tests d'intégration passent
- [ ] DPPA pipeline fonctionne end-to-end
- [ ] Pas d'erreurs dans les logs

---

## 6. Estimation Temps

| Phase | Tâche | Temps Estimé |
|-------|-------|--------------|
| 1 | Préparation | 30 min ✅ |
| 2 | Adapter EpochAssigner | 1h |
| 3 | Modifier Writers | 2h |
| 4 | Adapter DPPA | 1h |
| 5 | Nettoyage | 30 min |
| 6 | Tests & Docs | 1h |
| **Total** | | **6h** |

---

## 7. Prochaines Étapes

**Maintenant**:
1. Valider ce plan avec l'utilisateur ✅
2. Commencer Phase 2: Adapter `EpochAssigner`
3. Tester localement
4. Commit
5. Continuer phase par phase

**Ne PAS faire**:
- ❌ Modifier plusieurs modules en même temps
- ❌ Supprimer du code avant d'avoir adapté les dépendances
- ❌ Commit massif de tout d'un coup

**Approche**:
✅ Incrémenter, tester, commiter, répéter
