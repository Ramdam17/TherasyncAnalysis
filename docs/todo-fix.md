# TODO - Améliorations et Corrections Futures

Date de création : 2025-11-12  
Branche : feature/dppa-viz

---

## ✅ ~~Renommage : f → g (Groupe + Participant)~~ - **TERMINÉ**

**Status** : ✅ Complété le 2025-11-12  
**Commit** : `47e9a01` - refactor: rename f (famille) to g (groupe) across entire codebase

**Résultat** :
- ✅ 53 fichiers modifiés (464 insertions, 903 suppressions)
- ✅ Données brutes : 35 dossiers renommés (sub-f* → sub-g*)
- ✅ Fichiers : 1010 fichiers avec f0 → g0
- ✅ Métadonnées : participants.tsv, participants.json, 505 JSON (FamilyID)
- ✅ Configuration : config/dppa_dyads.yaml (g01-g06)
- ✅ Code : scripts/, src/, tests/ (tous mis à jour)
- ✅ Documentation : docs/, README.md, QUICKREF.md, QUICKSTART.md
- ✅ Vérification : 0 référence f0X restante

**Format final** : `sub-g01p02` (g = groupe, p = participant)

---

## ✅ ~~Intégration de l'epoching dans le preprocessing~~ - **TERMINÉ**

**Status** : ✅ Complété le 2025-11-12  
**Branche** : `feature/epoching-integration-testing`  
**Commits** :
- `f0ef644` - config: restructure epoching methods with per-moment parameters
- `bb263a9` - config: remove obsolete epoching file patterns and output config
- `2c7d8b8` - refactor(epoching): adapt EpochAssigner for per-moment configuration
- `0dbb31a` - refactor(preprocessing): integrate epoching into preprocessing writers
- `ada78db` - refactor(dppa): adapt to read from preprocessing directory
- `9ef3f6a` - refactor(epoching): deprecate separate epoching mode files
- `fe9dce3` - docs: finalize epoching integration (Phase 6/6)

**Résultat** :
- ✅ Epoch columns ajoutées directement dans preprocessing (`mode="preprocessing"`)
- ✅ Configuration per-moment (restingstate: nsplit1, therapy: nsplit120)
- ✅ BVP signals, RR intervals, EDA signals epochés automatiquement
- ✅ DPPA modules adaptés pour lire depuis `derivatives/preprocessing/`
- ✅ Anciens fichiers dépréciés avec warnings (`mode="separate"` legacy)
- ✅ Validation réussie sur g01p01 (2 sessions)
- ✅ Documentation mise à jour

**Bénéfices obtenus** :
- 💾 Réduction espace disque (~50% - pas de duplication)
- ⚡ Performance améliorée (pas d'étape séparée)
- 🔧 Pipeline simplifié (epoching intégré)
- � Moins de fichiers à gérer (un seul répertoire)
- 🎯 Flexibilité per-moment (params différents par task)

**Configuration finale** (`config/config.yaml`) :
```yaml
epoching:
  enabled: true
  mode: "preprocessing"  # Intégré dans preprocessing (default)
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
```

**Fichiers modifiés** :
- Config : `config/config.yaml`
- Epoching : `src/physio/epoching/epoch_assigner.py`
- Preprocessing : `base_bids_writer.py`, `bvp_bids_writer.py`, `eda_bids_writer.py`
- DPPA : `poincare_calculator.py`, `epoch_animator.py`, `compute_poincare.py`
- Deprecated : `epoch_bids_writer.py`, `epoch_all_signals.py`
- Docs : `docs/epoching-integration-plan.md`, `docs/resources.md`, `docs/api_reference.md`

---
- ✅ Pas de duplication des données
- ✅ Pipeline simplifié (une étape en moins)
- ✅ Cohérence : toutes les infos dans un seul fichier
- ✅ Plus rapide : pas de lecture/écriture supplémentaire

**Implémentation** :
```python
# Dans BVPBIDSWriter.save_rr_intervals()
# Après le calcul des RR intervals

# 1. Déterminer les méthodes d'epoching à appliquer
epoch_methods = config.get('epoching', {}).get('methods', [])

# 2. Pour chaque méthode configurée
for method in epoch_methods:
    if method['name'] == 'nsplit120':
        # Diviser en 120 epochs égaux
        rr_df['epoch_id'] = assign_equal_epochs(rr_df, n_splits=120)
    elif method['name'] == 'sliding':
        # Epochs glissants (durée, pas)
        rr_df['epoch_id'] = assign_sliding_epochs(
            rr_df, 
            duration=method['duration'], 
            step=method['step']
        )
    
    rr_df['epoch_start'] = rr_df.groupby('epoch_id')['time'].transform('first')
    rr_df['epoch_duration'] = rr_df.groupby('epoch_id')['time'].transform(lambda x: x.max() - x.min())

# 3. Sauvegarder avec les colonnes d'epoch incluses
# Nom de fichier : sub-g01p01_ses-01_task-therapy_desc-rrintervals_physio.tsv
```

**Fichiers à modifier** :
1. **Configuration** (`config/config.yaml`) :
   ```yaml
   epoching:
     enabled: true  # Active l'epoching durant preprocessing
     methods:
       - name: nsplit120
         description: "120 epochs égaux"
       - name: sliding_duration30s_step5s
         duration: 30  # secondes
         step: 5       # secondes
   ```

2. **Preprocessing** :
   - `src/physio/preprocessing/bvp_bids_writer.py` : Ajouter logique d'epoching dans `save_rr_intervals()`
**Fichiers modifiés** :
- Config : `config/config.yaml`
- Epoching : `src/physio/epoching/epoch_assigner.py`
- Preprocessing : `base_bids_writer.py`, `bvp_bids_writer.py`, `eda_bids_writer.py`
- DPPA : `poincare_calculator.py`, `epoch_animator.py`, `compute_poincare.py`
- Deprecated : `epoch_bids_writer.py`, `epoch_all_signals.py`
- Docs : `docs/epoching-integration-plan.md`, `docs/resources.md`, `docs/api_reference.md`

---

## 🔢 Validation de la robustesse au nombre de moments

**Problème** : Le code n'a pas été testé avec un nombre variable de sessions par participant.

**Cas à tester** :
- ✅ 1 session (g05, g06)
- ✅ 2 sessions (g01, g03)
- ✅ 3 sessions (g02)
- ⚠️ 5 sessions (g04) - **À VALIDER**
- ❌ Sessions manquantes (gaps)
- ❌ Sessions non-séquentielles

**Tests à implémenter** :
1. `test_variable_session_counts.py`
   - Traitement de g04 (5 sessions)
   - Gestion des sessions manquantes
   - Validation des outputs inter-session

2. `test_dyad_config_robustness.py`
   - Génération de dyades avec sessions différentes
   - Validation des paires inter/intra

3. `test_epoch_robustness.py`
   - Epoching avec durées variables
   - Gestion des bordures

**Estimation** : 4-5 heures  
**Priorité** : Haute  
**Impact** : Fiabilité du pipeline

---

## 📁 Réorganisation des dossiers de visualisation

**Problème** : Structure actuelle mélange données et visualisations, hiérarchie incohérente.

**Structure actuelle** :
```
data/derivatives/dppa/
├── figures/           # ❌ Mélangé avec les données
├── frames/            # ❌ Pas de hiérarchie
└── sub-*/             # ✅ Données numériques OK
```

**Structure cible** :
```
data/derivatives/
├── dppa/                              # Données numériques uniquement
│   ├── sub-*/ses-*/poincare/         # Centroids par sujet
│   ├── inter_session/                 # ICDs inter-session
│   └── intra_family/                  # ICDs intra-famille
│
└── visualization/                     # Visualisations séparées
    ├── dppa/
    │   ├── static/
    │   │   ├── inter/nsplit120/
    │   │   └── intra/
    │   │       ├── nsplit120/
    │   │       └── sliding_duration30s_step5s/
    │   ├── frames/
    │   │   └── intra/sliding_duration30s_step5s/{dyad}/
    │   └── videos/
    │       └── intra/sliding_duration30s_step5s/
    │
    ├── eda/                           # Futur : analyses exploratoires
    └── other_modalities/              # Futur : autres modalités
```

**Scripts à modifier** :
- `scripts/physio/dppa/plot_dyad.py` → output vers `visualization/dppa/static/`
- `scripts/physio/dppa/generate_epoch_frames.py` → output vers `visualization/dppa/frames/`
- `scripts/physio/dppa/generate_video.py` (futur) → output vers `visualization/dppa/videos/`

**Estimation** : 2 heures  
**Priorité** : Moyenne  
**Impact** : Organisation du projet

---

## 🏷️ Renommage : inter/intra → Termes plus explicites

**Problème** : "inter" et "intra" sont ambigus et prêtent à confusion.

**Terminologie actuelle** :
- `inter` = inter-session (même personne, sessions différentes)
- `intra` = intra-famille (personnes différentes, même session)

**Propositions de renommage** :

### Option 1 : Explicite
- `inter` → `self_comparison` ou `longitudinal`
- `intra` → `dyadic_comparison` ou `synchrony`

### Option 2 : Court
- `inter` → `self`
- `intra` → `dyad`

### Option 3 : Académique
- `inter` → `within_subject`
- `intra` → `between_subjects`

**À décider** : Quelle option privilégier ?

**Fichiers à modifier** :
- Tous les scripts CLI (arguments `--mode`)
- Modules de configuration (`DyadConfigLoader`)
- Structure des dossiers
- Documentation
- Tests

**Estimation** : 3-4 heures  
**Priorité** : Basse (peut attendre un refactoring plus large)  
**Impact** : Clarté conceptuelle

---

## 📊 Finalisation des figures DPPA

**Problème** : Visualisations DPPA incomplètes, plusieurs améliorations à apporter.

**Tâches restantes** :

### 1. Figures statiques (nsplit120)
- [ ] Améliorer la légende (taille, position)
- [ ] Ajouter annotations statistiques (p-values, effect sizes)
- [ ] Variantes de couleurs (colorblind-friendly)
- [ ] Export en haute résolution (300 DPI pour publications)

### 2. Animations (sliding windows)
- [ ] ✅ Frames epoch-by-epoch (FAIT)
- [ ] Génération vidéo (Stage 3 - ffmpeg)
- [ ] Overlay texte dynamique (epoch number, time, stats)
- [ ] Barre de progression temporelle
- [ ] Compression optimisée (H.264, qualité/taille)

### 3. Rapports HTML interactifs
- [ ] Figures Plotly interactives
- [ ] Dashboard avec sélection dyade/méthode/tâche
- [ ] Export des métriques en tableau
- [ ] Intégration avec Jupyter notebooks

### 4. Validation visuelle
- [ ] Vérifier cohérence nsplit120 vs sliding
- [ ] Comparer ICD calculés vs visualisés
- [ ] Test sur toutes les dyades (inter + intra)
- [ ] Documentation des cas limites

**Estimation** : 8-10 heures  
**Priorité** : Moyenne (selon besoins de publication)  
**Impact** : Qualité des figures scientifiques

---

## 📝 Notes générales

**Ordre suggéré de traitement** :
1. ✅ ~~Renommage f→g~~ (TERMINÉ - 2025-11-12)
2. Intégration epoching dans preprocessing (Haute priorité - optimisation majeure)
3. Validation robustesse sessions variables (Haute priorité - fiabilité)
4. Réorganisation dossiers visualisation (Moyenne priorité, bloque autres tâches)
5. Finalisation figures DPPA (Selon besoins de publication)
6. Renommage inter/intra (Basse priorité, peut attendre)

**Stratégie** :
- Créer une branche dédiée pour chaque grosse modification
- Tester sur petit échantillon avant batch complet
- Mettre à jour la documentation en parallèle
- Ajouter tests de non-régression
- Pour l'intégration epoching : prévoir migration des données existantes

---

**Dernière mise à jour** : 2025-11-12  
**Responsable** : Lena Adel, Remy Ramadour
