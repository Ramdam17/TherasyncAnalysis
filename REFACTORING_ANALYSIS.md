# Analyse Comparative des Modalités - Refactoring Strategy

**Date**: 11 novembre 2025  
**Branche**: `refactor/code-cleanup`  
**Objectif**: Homogénéiser les conventions entre BVP, EDA et HR

---

## 1. Analyse des Incohérences

### 1.1 Conventions de Nommage de Fichiers

#### **BVP** (Blood Volume Pulse / Cardiac)
```
Structure: derivatives/preprocessing/sub-{id}/ses-{id}/bvp/

Fichiers par moment (restingstate, therapy):
- sub-{id}_ses-{id}_task-{moment}_desc-processed_recording-bvp.tsv
- sub-{id}_ses-{id}_task-{moment}_desc-processed_recording-bvp.json
- sub-{id}_ses-{id}_task-{moment}_desc-processing_recording-bvp.json

Fichiers de session:
- sub-{id}_ses-{id}_desc-bvp-metrics_physio.tsv
- sub-{id}_ses-{id}_desc-bvp-metrics_physio.json
- sub-{id}_ses-{id}_desc-bvp-summary_recording-bvp.json
```

**Caractéristiques**:
- ✅ Préfixe `sub-` et `ses-` dans les noms de fichiers
- ✅ Utilise `task-{moment}` pour les données par moment
- ✅ Suffixe `_recording-bvp` pour identifier la modalité
- ✅ Métriques globales avec `desc-bvp-metrics_physio`
- ⚠️ Pas de fichiers d'événements séparés (pics BVP intégrés dans processing metadata)

---

#### **EDA** (Electrodermal Activity / Arousal)
```
Structure: derivatives/preprocessing/sub-{id}/ses-{id}/eda/

Fichiers par moment (restingstate, therapy):
- sub-{id}_ses-{id}_task-{moment}_desc-processed_recording-eda.tsv
- sub-{id}_ses-{id}_task-{moment}_desc-processed_recording-eda.json
- sub-{id}_ses-{id}_task-{moment}_desc-processing_recording-eda.json
- sub-{id}_ses-{id}_task-{moment}_desc-scr_events.tsv
- sub-{id}_ses-{id}_task-{moment}_desc-scr_events.json

Fichiers de session:
- sub-{id}_ses-{id}_desc-eda-metrics_physio.tsv
- sub-{id}_ses-{id}_desc-eda-metrics_physio.json
- sub-{id}_ses-{id}_desc-eda-summary_recording-eda.json
```

**Caractéristiques**:
- ✅ Préfixe `sub-` et `ses-` dans les noms de fichiers
- ✅ Utilise `task-{moment}` pour les données par moment
- ✅ Suffixe `_recording-eda` pour identifier la modalité
- ✅ Fichiers d'événements séparés (`desc-scr_events`) pour les pics SCR
- ✅ Métriques globales avec `desc-eda-metrics_physio`
- ✅ Similaire à BVP avec ajout des événements SCR

---

#### **HR** (Heart Rate)
```
Structure: derivatives/preprocessing/sub-{id}/ses-{id}/hr/

Fichiers UNIQUES (task-combined):
- sub-{id}_ses-{id}_task-combined_physio.tsv.gz  ⚠️ COMPRESSÉ
- sub-{id}_ses-{id}_task-combined_physio.json
- sub-{id}_ses-{id}_task-combined_events.tsv
- sub-{id}_ses-{id}_task-combined_events.json
- sub-{id}_ses-{id}_task-combined_hr-metrics.tsv
- sub-{id}_ses-{id}_task-combined_hr-metrics.json
- sub-{id}_ses-{id}_task-combined_hr-summary.json
```

**Caractéristiques**:
- ✅ Préfixe `sub-` et `ses-` dans les noms de fichiers
- ❌ **DIFFÉRENT**: Utilise `task-combined` au lieu de moments séparés
- ❌ **DIFFÉRENT**: Pas de suffixe `_recording-hr` (pas de modalité dans le nom)
- ✅ Fichiers d'événements séparés (`events.tsv`)
- ❌ **DIFFÉRENT**: Métriques avec `hr-metrics` (tiret) vs `{modality}-metrics` pour BVP/EDA
- ⚠️ **COMPRESSION**: Fichiers `.tsv.gz` vs `.tsv` non compressés pour BVP/EDA
- ❌ **PAS DE MOMENTS SÉPARÉS**: Une seule sortie combinée au lieu de restingstate + therapy

---

### 1.2 Structure des Données de Sortie

#### **Colonnes dans les signaux traités**

**BVP** (`_desc-processed_recording-bvp.tsv`):
```python
Colonnes:
- time (s)
- PPG_Clean (a.u.)          # Signal BVP nettoyé
- PPG_Rate (BPM)            # Fréquence cardiaque instantanée
- PPG_Peaks (binary)        # Marqueurs de pics R
- PPG_Quality (0-1)         # Score de qualité du signal
```

**EDA** (`_desc-processed_recording-eda.tsv`):
```python
Colonnes:
- time (s)
- EDA_Raw (µS)              # Signal brut
- EDA_Clean (µS)            # Signal nettoyé
- EDA_Tonic (µS)            # Composante tonique (baseline)
- EDA_Phasic (µS)           # Composante phasique (réponses)
# SCR_Peaks NON inclus dans TSV (fichier events séparé)
```

**HR** (`_task-combined_physio.tsv.gz`):
```python
Colonnes:
- time (s)
- hr (BPM)                  # Fréquence cardiaque nettoyée
- quality (0-1)             # Score de qualité
- outlier (0/1)             # Flag d'outlier
- interpolated (0/1)        # Flag d'interpolation
```

**Incohérences identifiées**:
- ❌ BVP utilise `PPG_*` (Photoplethysmography), pas `BVP_*`
- ❌ EDA utilise `EDA_*`, cohérent avec la modalité
- ❌ HR utilise `hr` (minuscule), pas `HR_*`
- ❌ Colonnes de qualité: `PPG_Quality` vs `quality` vs pas de colonne qualité pour EDA
- ❌ Flags d'outliers/interpolation: uniquement pour HR, pas pour BVP/EDA

---

### 1.3 Métriques Extraites

#### **BVP Metrics** (`desc-bvp-metrics_physio.tsv`)
```python
Colonnes de sortie:
- moment (restingstate, therapy)
- HRV_MeanNN, HRV_SDNN, HRV_RMSSD, HRV_pNN50, ...  # Métriques HRV temps
- HRV_LF, HRV_HF, HRV_LFHF, HRV_LFn, HRV_HFn, ...  # Métriques HRV fréquence
- HRV_SD1, HRV_SD2, HRV_SD1SD2, ...                # Métriques non-linéaires
- BVP_NumPeaks, BVP_Duration, BVP_PeakRate, ...    # Métriques signal
```

**Caractéristiques**:
- ✅ Préfixe `HRV_` pour métriques de variabilité
- ✅ Préfixe `BVP_` pour métriques de signal
- ✅ Format long : une ligne par moment
- ✅ Noms explicites et descriptifs

---

#### **EDA Metrics** (`desc-eda-metrics_physio.tsv`)
```python
Colonnes de sortie:
- moment (restingstate, therapy)
- EDA_Tonic_Mean, EDA_Tonic_Std, EDA_Tonic_Min, EDA_Tonic_Max
- EDA_Phasic_Mean, EDA_Phasic_Std, EDA_Phasic_Min, EDA_Phasic_Max
- SCR_Peaks_N, SCR_Peaks_Rate
- SCR_Amplitude_Mean, SCR_Amplitude_Max
- SCR_RiseTime_Mean, SCR_RecoveryTime_Mean
```

**Caractéristiques**:
- ✅ Préfixe `EDA_` pour métriques de signal
- ✅ Préfixe `SCR_` pour métriques d'événements
- ✅ Format long : une ligne par moment
- ✅ Noms explicites avec stats (Mean, Std, Min, Max)

---

#### **HR Metrics** (`task-combined_hr-metrics.tsv`)
```python
Colonnes de sortie:
- moment (combined)  ⚠️ UNE SEULE VALEUR
- hr_mean, hr_std, hr_min, hr_max, hr_median
- hr_iqr, hr_cv, hr_range
- hr_percentile_25, hr_percentile_75, hr_percentile_90
- hr_skewness, hr_kurtosis
- hr_rmssd, hr_sdnn
- elevated_periods_count, elevated_periods_total_duration
- ...
```

**Caractéristiques**:
- ❌ Préfixe `hr_` (minuscule) vs `HRV_`, `BVP_`, `EDA_`, `SCR_` (majuscules)
- ❌ Une seule ligne (moment=combined) vs plusieurs moments pour BVP/EDA
- ✅ Noms descriptifs avec stats
- ⚠️ Mélange de métriques HR et HRV (`hr_rmssd`, `hr_sdnn`)

---

### 1.4 Conventions de Code

#### **Architecture des Writers**

**BVP** (`bvp_bids_writer.py`):
```python
class BVPBIDSWriter:
    def save_processed_data(
        subject_id: str,          # Format: 'sub-f01p01'
        session_id: str,          # Format: 'ses-01'
        processed_results: Dict[str, Tuple[pd.DataFrame, Dict]],  # Tuple!
        session_metrics: Dict[str, Dict[str, float]],
        processing_metadata: Optional[Dict] = None
    ) -> Dict[str, List[str]]
```

**EDA** (`eda_bids_writer.py`):
```python
class EDABIDSWriter:
    def save_processed_data(
        subject_id: str,          # Format: 'sub-f01p01'
        session_id: str,          # Format: 'ses-01'
        processed_results: Dict[str, pd.DataFrame],  # DataFrame seulement!
        session_metrics: pd.DataFrame,               # DataFrame, pas Dict!
        processing_metadata: Optional[Dict] = None
    ) -> Dict[str, List[str]]
```

**HR** (`hr_bids_writer.py`):
```python
class HRBIDSWriter:
    def write_hr_results(
        subject: str,             # Format: 'f01p01' (PAS de préfixe 'sub-')
        session: str,             # Format: '01' (PAS de préfixe 'ses-')
        moment: str,              # 'combined' uniquement
        cleaned_data: pd.DataFrame,
        metrics: Dict[str, Any],
        cleaning_metadata: Dict[str, Any]
    ) -> Dict[str, Path]
```

**Incohérences majeures**:
- ❌ **Noms de méthodes**: `save_processed_data` (BVP/EDA) vs `write_hr_results` (HR)
- ❌ **Format ID**: `'sub-f01p01'` (BVP/EDA) vs `'f01p01'` (HR sans préfixe)
- ❌ **Type processed_results**: `Tuple[DataFrame, Dict]` (BVP) vs `DataFrame` (EDA/HR)
- ❌ **Type session_metrics**: `Dict[str, Dict]` (BVP) vs `DataFrame` (EDA) vs `Dict[str, Any]` (HR)
- ❌ **Return type**: `Dict[str, List[str]]` (BVP/EDA) vs `Dict[str, Path]` (HR)

---

### 1.5 Traitement par Moments

#### **BVP**:
```python
# Traite chaque moment séparément
for moment in ['restingstate', 'therapy']:
    processed_signals, processing_info = cleaner.process_moment(...)
    # Sortie: 2 fichiers processed, 2 fichiers processing
```

#### **EDA**:
```python
# Traite chaque moment séparément
for moment in ['restingstate', 'therapy']:
    processed_signals = cleaner.clean_signal(...)
    # Sortie: 2 fichiers processed, 2 fichiers processing, 2 fichiers events
```

#### **HR**:
```python
# Traite TOUS les moments ensemble → "combined"
all_moments_data = loader.load_subject_session(...)  # Charge tout
cleaned_combined = cleaner.clean_signal(...)
# Sortie: 1 fichier physio (combined), 1 fichier events, 1 fichier metrics
```

**Incohérence majeure**:
- ❌ BVP/EDA génèrent des fichiers **par moment** (restingstate, therapy)
- ❌ HR génère **UN SEUL fichier combiné** (task-combined)
- ⚠️ Impossible de comparer restingstate vs therapy pour HR isolément

---

## 2. Stratégie d'Homogénéisation Proposée

### 2.1 Principes Directeurs

1. **Cohérence BIDS**: Suivre strictement les spécifications BIDS pour derivatives
2. **Modularité**: Chaque moment doit être traitable indépendamment
3. **Préfixes clairs**: Majuscules pour noms de colonnes (BVP_, EDA_, HR_)
4. **Symétrie**: Même structure de fichiers pour toutes les modalités
5. **Backward compatibility**: Permettre la lecture des anciens formats

---

### 2.2 Convention de Nommage Unifiée

#### **Fichiers de Signaux Traités**

**Proposition (OPTION A - moments séparés)** ✅ RECOMMANDÉ
```
Format pour TOUTES les modalités:
sub-{id}_ses-{id}_task-{moment}_desc-processed_recording-{modality}.tsv
sub-{id}_ses-{id}_task-{moment}_desc-processed_recording-{modality}.json
sub-{id}_ses-{id}_task-{moment}_desc-processing_recording-{modality}.json

Exemples:
- sub-f01p01_ses-01_task-restingstate_desc-processed_recording-bvp.tsv
- sub-f01p01_ses-01_task-therapy_desc-processed_recording-eda.tsv
- sub-f01p01_ses-01_task-restingstate_desc-processed_recording-hr.tsv  ← NOUVEAU
```

**Justification**:
- ✅ Permet comparaison restingstate vs therapy pour HR
- ✅ Structure identique entre modalités
- ✅ Compatible avec analyses multi-modalités
- ✅ Suit convention BIDS `task-{taskname}`

**Proposition alternative (OPTION B - combined + moments)**:
```
Garder combined pour HR mais ajouter moments séparés:
- sub-{id}_ses-{id}_task-combined_desc-processed_recording-hr.tsv
- sub-{id}_ses-{id}_task-restingstate_desc-processed_recording-hr.tsv  ← AJOUTER
- sub-{id}_ses-{id}_task-therapy_desc-processed_recording-hr.tsv        ← AJOUTER
```

**Justification**:
- ✅ Backward compatible (garde combined)
- ✅ Ajoute granularité par moment
- ⚠️ Duplication de données (mais utile pour analyses)

---

#### **Fichiers d'Événements**

**Proposition** ✅
```
Format unifié:
sub-{id}_ses-{id}_task-{moment}_desc-{eventtype}_events.tsv
sub-{id}_ses-{id}_task-{moment}_desc-{eventtype}_events.json

Exemples:
- sub-f01p01_ses-01_task-restingstate_desc-peaks_events.tsv     ← BVP peaks
- sub-f01p01_ses-01_task-therapy_desc-scr_events.tsv            ← EDA SCRs (existant)
- sub-f01p01_ses-01_task-combined_desc-elevated_events.tsv      ← HR elevated periods
```

**Justification**:
- ✅ Format BIDS `_events.tsv`
- ✅ `desc-{eventtype}` identifie le type d'événement
- ✅ Structure cohérente entre modalités

---

#### **Fichiers de Métriques**

**Proposition** ✅
```
Format unifié:
sub-{id}_ses-{id}_desc-{modality}-metrics_physio.tsv
sub-{id}_ses-{id}_desc-{modality}-metrics_physio.json

Exemples (existants, à garder):
- sub-f01p01_ses-01_desc-bvp-metrics_physio.tsv   ✅ OK
- sub-f01p01_ses-01_desc-eda-metrics_physio.tsv   ✅ OK
- sub-f01p01_ses-01_desc-hr-metrics_physio.tsv    ✅ OK
```

**Justification**:
- ✅ Déjà cohérent entre BVP/EDA/HR
- ✅ Suffixe `_physio` BIDS-compliant
- ✅ Une ligne par moment dans le TSV

---

### 2.3 Convention de Colonnes Unifiée

#### **Signaux Traités**

**Proposition** ✅
```python
# BVP
time, BVP_Raw, BVP_Clean, BVP_Rate, BVP_Peaks, BVP_Quality

# EDA  
time, EDA_Raw, EDA_Clean, EDA_Tonic, EDA_Phasic, EDA_Quality  ← AJOUTER Quality

# HR
time, HR_Raw, HR_Clean, HR_Quality, HR_Outlier, HR_Interpolated  ← AJOUTER Raw, renommer

# Colonnes communes pour TOUTES:
- time (s)
- {MODALITY}_Raw (unité)         ← Signal original
- {MODALITY}_Clean (unité)       ← Signal nettoyé
- {MODALITY}_Quality (0-1)       ← Score de qualité (OBLIGATOIRE)
```

**Justification**:
- ✅ Préfixe majuscule cohérent: `BVP_`, `EDA_`, `HR_`
- ✅ Colonnes Raw/Clean/Quality communes
- ✅ Colonnes spécifiques à la modalité après
- ✅ Facilite scripts multi-modalités

---

#### **Métriques**

**Proposition** ✅
```python
# Convention: {MODALITY}_{METRIC}_{STAT}

# BVP (renommer PPG → BVP où pertinent)
HRV_MeanNN, HRV_SDNN, HRV_RMSSD, ...  ← Garder HRV_ (domaine HRV)
BVP_NumPeaks, BVP_Duration, ...        ← OK

# EDA
EDA_Tonic_Mean, EDA_Tonic_Std, ...     ← OK
SCR_Peaks_N, SCR_Amplitude_Mean, ...   ← OK

# HR (renommer hr_ → HR_)
HR_Mean, HR_Std, HR_Min, HR_Max, ...           ← MAJUSCULES
HR_RMSSD, HR_SDNN, ...                          ← Cohérent avec HRV_
HR_ElevatedPeriods_Count, HR_ElevatedPeriods_Duration, ...  ← CamelCase
```

**Justification**:
- ✅ Majuscules cohérentes
- ✅ Séparation par underscores
- ✅ Préfixes clairs par modalité/domaine

---

### 2.4 API Unifiée des Writers

**Proposition** ✅
```python
# Interface commune pour TOUS les writers
class PhysioBIDSWriter(ABC):
    """Base class for all physio BIDS writers."""
    
    @abstractmethod
    def save_processed_data(
        self,
        subject_id: str,          # Format: 'sub-f01p01' (AVEC préfixe)
        session_id: str,          # Format: 'ses-01' (AVEC préfixe)
        processed_results: Dict[str, pd.DataFrame],  # Clé: moment, Valeur: signals
        session_metrics: pd.DataFrame,               # DataFrame unifié
        processing_metadata: Optional[Dict] = None
    ) -> Dict[str, List[Path]]:                     # Return Paths, pas strings
        """Save processed data in BIDS format."""
        pass
```

**Implémentations**:
```python
class BVPBIDSWriter(PhysioBIDSWriter):
    # Adapter pour correspondre à la signature commune
    
class EDABIDSWriter(PhysioBIDSWriter):
    # Déjà proche, ajuster return type
    
class HRBIDSWriter(PhysioBIDSWriter):
    # Renommer write_hr_results → save_processed_data
    # Changer format IDs (ajouter préfixes sub-/ses-)
    # Changer type metrics (Dict → DataFrame)
```

**Justification**:
- ✅ Polymorphisme possible
- ✅ Tests unitaires réutilisables
- ✅ Documentation unifiée
- ✅ Facilite maintenance

---

## 3. Plan de Migration

### Phase 1: Analyse et Documentation ✅ EN COURS
- [x] Identifier toutes les incohérences
- [x] Documenter conventions actuelles
- [x] Proposer stratégie unifiée
- [ ] Discussion avec l'équipe

### Phase 2: Refactoring Writers (1-2 jours)
1. **Créer classe de base** `PhysioBIDSWriter`
2. **HR**: Adapter à la nouvelle interface
   - Renommer `write_hr_results` → `save_processed_data`
   - Ajouter préfixes `sub-`/`ses-` aux IDs
   - Changer type métriques (Dict → DataFrame)
   - Générer fichiers par moment (pas seulement combined)
3. **BVP**: Adapter au format DataFrame simple
   - Changer `Tuple[DataFrame, Dict]` → `DataFrame`
   - Passer metadata séparément
4. **EDA**: Harmoniser return types (Path vs string)

### Phase 3: Refactoring Colonnes (1 jour)
1. **HR**: Renommer colonnes
   - `hr` → `HR_Clean`
   - `quality` → `HR_Quality`
   - `outlier` → `HR_Outlier`
   - `interpolated` → `HR_Interpolated`
   - Ajouter `HR_Raw`
2. **EDA**: Ajouter colonne qualité
   - Calculer `EDA_Quality` basé sur tonic stability
3. **BVP**: Renommer si besoin
   - `PPG_*` → garder (cohérent avec domaine HRV)

### Phase 4: Refactoring Métriques (1 jour)
1. **HR**: Renommer métriques
   - `hr_mean` → `HR_Mean`
   - `hr_rmssd` → `HR_RMSSD` (cohérent avec HRV)
   - `elevated_periods_count` → `HR_ElevatedPeriods_Count`
2. **Validation**: Vérifier cohérence BVP/EDA/HR

### Phase 5: Tests et Validation (1 jour)
1. **Tests unitaires** pour chaque writer
2. **Tests d'intégration** bout-en-bout
3. **Retraitement** d'un sous-ensemble de données
4. **Comparaison** anciennes vs nouvelles métriques

### Phase 6: Migration Complète (optionnel)
1. **Script de conversion** anciens → nouveaux formats
2. **Retraitement** de toutes les données
3. **Validation** qualité des outputs
4. **Documentation** changements breaking

---

## 4. Questions pour Discussion

### 4.1 HR: Moments séparés ou combined?

**Option A**: Générer restingstate + therapy + combined
- ✅ Cohérent avec BVP/EDA
- ✅ Permet comparaisons par moment
- ⚠️ Duplication de données (~3x taille)

**Option B**: Garder uniquement combined
- ✅ Moins de stockage
- ❌ Incohérent avec BVP/EDA
- ❌ Perd information par moment

**Recommandation**: Option A, avec possibilité de désactiver combined dans config

---

### 4.2 Compression: .tsv.gz pour tous?

**Actuellement**: Seulement HR compressé

**Option A**: Compresser toutes les modalités
- ✅ Économie d'espace (~70-90% réduction)
- ⚠️ Légèrement plus lent à lire
- ✅ BIDS-compliant (supporte .gz)

**Option B**: Décompresser HR
- ✅ Lecture plus rapide
- ❌ Utilise plus d'espace disque
- ✅ Cohérent avec BVP/EDA actuels

**Recommandation**: Option A (compresser tout) avec option config

---

### 4.3 Noms de colonnes: PPG vs BVP?

**Actuellement**: BVP utilise `PPG_*` (Photoplethysmography)

**Option A**: Garder PPG_* (terminologie scientifique)
- ✅ Cohérent avec littérature HRV
- ✅ Déjà utilisé dans le code
- ⚠️ Moins cohérent avec nom de modalité "BVP"

**Option B**: Renommer PPG_* → BVP_*
- ✅ Cohérent avec nom de modalité
- ✅ Plus simple pour utilisateurs
- ⚠️ Breaking change

**Recommandation**: Garder PPG_* (Option A) - c'est la terminologie correcte

---

### 4.4 Backward Compatibility

**Question**: Doit-on supporter la lecture des anciens formats?

**Option A**: Reader compatible avec ancien et nouveau
- ✅ Pas de rupture pour utilisateurs
- ⚠️ Code plus complexe
- ⚠️ Maintenance long terme

**Option B**: Breaking change avec script de migration
- ✅ Code plus propre
- ✅ Pas de dette technique
- ⚠️ Nécessite retraitement de toutes les données

**Recommandation**: Option B avec script automatique de migration

---

## 5. Résumé des Changements Proposés

### Fichiers HR (MAJEURS)
- [ ] Renommer `write_hr_results()` → `save_processed_data()`
- [ ] Ajouter préfixes `sub-`/`ses-` aux IDs
- [ ] Générer fichiers par moment (restingstate, therapy) en plus de combined
- [ ] Renommer colonnes: `hr` → `HR_Clean`, `quality` → `HR_Quality`, etc.
- [ ] Renommer métriques: `hr_mean` → `HR_Mean`, etc.
- [ ] Ajouter colonne `HR_Raw`
- [ ] Optionnel: décompresser `.tsv.gz` → `.tsv`

### Fichiers EDA (MINEURS)
- [ ] Ajouter colonne `EDA_Quality`
- [ ] Harmoniser return types (Path vs string)

### Fichiers BVP (MINEURS)
- [ ] Simplifier API: `Tuple[DataFrame, Dict]` → `DataFrame` + metadata séparé
- [ ] Harmoniser return types (Path vs string)

### Classe de Base (NOUVEAU)
- [ ] Créer `PhysioBIDSWriter` abstract base class
- [ ] Implémenter dans BVP/EDA/HR Writers

---

## 6. Métriques de Succès

✅ **Cohérence**: Même structure de fichiers pour BVP/EDA/HR  
✅ **Modularité**: Traitement par moment possible pour toutes modalités  
✅ **BIDS Compliance**: Respect strict de la spécification BIDS  
✅ **Lisibilité**: Noms de colonnes/métriques auto-documentés  
✅ **Maintenabilité**: Code factorisé, moins de duplication  
✅ **Testabilité**: Tests unitaires partagés via classe de base  

---

**Prochaine étape**: Discussion de cette analyse et validation de la stratégie proposée.
