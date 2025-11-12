# TODO - Améliorations et Corrections Futures

Date de création : 2025-11-12  
Branche : feature/dppa-viz

---

## 🔤 Renommage : f → g (Groupe + Participant)

**Problème** : Convention de nommage actuelle utilise `f` (famille) mais devrait être `g` (groupe).

**Format cible** :
- Ancien : `sub-f01p02` 
- Nouveau : `sub-g01p02` (g = groupe, p = participant)

**Fichiers à modifier** :
1. **Données brutes** (sourcedata/) :
   - Renommer tous les dossiers `sub-f*` → `sub-g*`
   - Vérifier les métadonnées JSON

2. **Code** :
   - Tous les scripts dans `scripts/`
   - Tous les modules dans `src/`
   - Tests dans `tests/`
   - Documentation dans `docs/`
   - Fichiers de configuration `config/`

3. **Données dérivées** :
   - Régénérer ou renommer `data/derivatives/`

**Estimation** : 2-3 heures  
**Priorité** : Moyenne  
**Impact** : Toute la codebase

---

## 🔢 Validation de la robustesse au nombre de moments

**Problème** : Le code n'a pas été testé avec un nombre variable de sessions par participant.

**Cas à tester** :
- ✅ 1 session (f05, f06)
- ✅ 2 sessions (f01, f03)
- ✅ 3 sessions (f02)
- ⚠️ 5 sessions (f04) - **À VALIDER**
- ❌ Sessions manquantes (gaps)
- ❌ Sessions non-séquentielles

**Tests à implémenter** :
1. `test_variable_session_counts.py`
   - Traitement de f04 (5 sessions)
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
1. Validation robustesse (Haute priorité)
2. Réorganisation dossiers (Moyenne priorité, bloque autres tâches)
3. Renommage f→g (Moyenne priorité, large impact)
4. Finalisation figures DPPA (Selon besoins)
5. Renommage inter/intra (Basse priorité, peut attendre)

**Stratégie** :
- Créer une branche dédiée pour chaque grosse modification
- Tester sur petit échantillon avant batch complet
- Mettre à jour la documentation en parallèle
- Ajouter tests de non-régression

---

**Dernière mise à jour** : 2025-11-12  
**Responsable** : Lena Adel, Remy Ramadour
