# Phase 2: Refactoring Writers - Plan de Travail

**Date début**: 11 novembre 2025  
**Branche**: `refactor/code-cleanup`  
**Status**: 🔄 EN COURS

---

## Objectifs Phase 2

1. ✅ Créer classe de base `PhysioBIDSWriter`
2. ⏳ Adapter `HRBIDSWriter` à la nouvelle interface
3. ⏳ Adapter `BVPBIDSWriter` à la nouvelle interface
4. ⏳ Adapter `EDABIDSWriter` à la nouvelle interface
5. ⏳ Harmoniser les noms de colonnes et clés de dictionnaires

---

## Changements à Implémenter

### 1. HRBIDSWriter ⏳

#### API Changes
- [ ] Hériter de `PhysioBIDSWriter`
- [ ] Renommer `write_hr_results()` → `save_processed_data()`
- [ ] Modifier signature pour accepter IDs avec préfixes (`sub-`, `ses-`)
- [ ] Changer `processed_results: Dict[str, pd.DataFrame]` (au lieu de données directes)
- [ ] Changer `session_metrics: pd.DataFrame` (au lieu de Dict)
- [ ] Return type: `Dict[str, List[Path]]` (au lieu de `Dict[str, Path]`)

#### Traitement par Moments
- [ ] Supprimer logique "combined" unique
- [ ] Implémenter traitement par moment (restingstate, therapy)
- [ ] Générer fichiers séparés pour chaque moment
- [ ] Optionnel: Garder génération "combined" en plus

#### Noms de Colonnes
- [ ] `hr` → `HR_Raw` (ajouter signal brut)
- [ ] `hr_clean` → `HR_Clean`
- [ ] `quality` → `HR_Quality`
- [ ] `outlier` → `HR_Outlier`
- [ ] `interpolated` → `HR_Interpolated`

#### Noms de Fichiers
- [ ] `task-combined_physio.tsv.gz` → `task-{moment}_desc-processed_recording-hr.tsv`
- [ ] Décompresser: `.tsv.gz` → `.tsv`
- [ ] Ajouter suffixe modalité: `_recording-hr`
- [ ] `hr-metrics.tsv` → `desc-hr-metrics_physio.tsv` (déjà OK)

#### Métriques (voir Phase 4 pour détails)
- [ ] Renommer toutes métriques: `hr_*` → `HR_*`
- [ ] `hr_mean` → `HR_Mean`
- [ ] `hr_rmssd` → `HR_RMSSD`
- [ ] `elevated_periods_count` → `HR_ElevatedPeriods_Count`
- [ ] etc. (liste complète dans Phase 4)

---

### 2. BVPBIDSWriter ⏳

#### API Changes
- [ ] Hériter de `PhysioBIDSWriter`
- [ ] Garder nom `save_processed_data()` ✅
- [ ] Modifier `processed_results` de `Dict[str, Tuple[pd.DataFrame, Dict]]` → `Dict[str, pd.DataFrame]`
- [ ] Passer `processing_info` dans `processing_metadata` au lieu de tuple
- [ ] Return type: `Dict[str, List[Path]]` déjà OK ✅

#### Noms de Colonnes
- [ ] Garder `PPG_*` (terminologie scientifique correcte)
- [ ] Vérifier cohérence: `PPG_Clean`, `PPG_Rate`, `PPG_Peaks`, `PPG_Quality`

#### Clés de Dictionnaires
- [ ] Vérifier cohérence dans `processing_info`
- [ ] Standardiser noms de clés (majuscules)

---

### 3. EDABIDSWriter ⏳

#### API Changes
- [ ] Hériter de `PhysioBIDSWriter`
- [ ] Garder nom `save_processed_data()` ✅
- [ ] `session_metrics` déjà DataFrame ✅
- [ ] Return type: harmoniser `List[str]` → `List[Path]`

#### Noms de Colonnes
- [ ] Ajouter colonne `EDA_Quality`
- [ ] Garder: `EDA_Raw`, `EDA_Clean`, `EDA_Tonic`, `EDA_Phasic` ✅

#### Calcul Quality
- [ ] Implémenter calcul `EDA_Quality` basé sur:
  - Stabilité du signal tonic
  - Ratio signal/bruit
  - Variance phasique

#### Clés de Dictionnaires
- [ ] Vérifier cohérence dans metadata
- [ ] Standardiser noms de clés

---

## Harmonisation Noms de Colonnes

### Colonnes Communes (TOUTES modalités)

```python
REQUIRED_COLUMNS = [
    'time',              # Temps en secondes (float)
    '{MOD}_Raw',         # Signal brut
    '{MOD}_Clean',       # Signal nettoyé
    '{MOD}_Quality'      # Score qualité 0-1
]
```

### Colonnes Spécifiques

**BVP**:
```python
- PPG_Raw            # Signal PPG brut
- PPG_Clean          # Signal PPG nettoyé
- PPG_Rate           # Fréquence cardiaque instantanée (BPM)
- PPG_Peaks          # Marqueurs de pics R (0/1)
- PPG_Quality        # Score de qualité (0-1)
```

**EDA**:
```python
- EDA_Raw            # Signal EDA brut (µS)
- EDA_Clean          # Signal EDA nettoyé (µS)
- EDA_Tonic          # Composante tonique (µS)
- EDA_Phasic         # Composante phasique (µS)
- EDA_Quality        # Score de qualité (0-1) ← NOUVEAU
```

**HR**:
```python
- HR_Raw             # Signal HR brut (BPM) ← NOUVEAU
- HR_Clean           # Signal HR nettoyé (BPM)
- HR_Quality         # Score de qualité (0-1)
- HR_Outlier         # Flag outlier (0/1)
- HR_Interpolated    # Flag interpolation (0/1)
```

---

## Harmonisation Clés de Dictionnaires

### Metadata de Traitement

```python
PROCESSING_METADATA_KEYS = {
    # Informations temporelles
    'sampling_rate': float,      # Hz
    'duration': float,            # secondes
    'start_time': float,          # timestamp
    
    # Qualité
    'quality_score': float,       # 0-1
    'valid_samples': int,         # nombre
    'total_samples': int,         # nombre
    
    # Détection de pics/événements
    'num_peaks': int,             # BVP, EDA (SCR)
    'peak_rate': float,           # pics/minute
    
    # Flags de traitement
    'outliers_removed': int,      # HR, BVP
    'interpolated_samples': int,  # HR
    
    # Paramètres de traitement
    'processing_parameters': {
        'method': str,
        'threshold': float,
        # ...
    }
}
```

### Noms de Moments

```python
VALID_MOMENTS = [
    'restingstate',   # État de repos
    'therapy',        # Thérapie
    'combined'        # Optionnel: tous moments combinés
]
```

---

## Tests de Validation

### Tests Unitaires à Créer

```python
class TestPhysioBIDSWriter:
    """Tests pour classe de base."""
    
    def test_ensure_prefix():
        """Test ajout de préfixes sub-/ses-."""
        pass
    
    def test_strip_prefix():
        """Test suppression de préfixes."""
        pass
    
    def test_get_subject_session_dir():
        """Test création de répertoires."""
        pass
    
    def test_json_serializer():
        """Test sérialisation JSON."""
        pass


class TestHRBIDSWriter:
    """Tests pour HR writer."""
    
    def test_save_processed_data_signature():
        """Vérifier nouvelle signature API."""
        pass
    
    def test_moment_separation():
        """Vérifier génération par moment."""
        pass
    
    def test_column_names():
        """Vérifier noms de colonnes."""
        pass
    
    def test_file_naming():
        """Vérifier noms de fichiers."""
        pass
    
    def test_decompression():
        """Vérifier fichiers .tsv non compressés."""
        pass


class TestBVPBIDSWriter:
    """Tests pour BVP writer."""
    
    def test_processed_results_format():
        """Vérifier format DataFrame simple."""
        pass
    
    def test_column_names():
        """Vérifier colonnes PPG_*."""
        pass


class TestEDABIDSWriter:
    """Tests pour EDA writer."""
    
    def test_quality_column():
        """Vérifier ajout colonne EDA_Quality."""
        pass
    
    def test_return_type():
        """Vérifier return List[Path]."""
        pass
```

---

## Checklist Phase 2

### Étape 1: Classe de Base ✅
- [x] Créer `src/physio/preprocessing/base_bids_writer.py`
- [x] Définir interface abstraite `PhysioBIDSWriter`
- [x] Implémenter méthodes communes (prefixes, serialization, etc.)

### Étape 2: Adapter HR Writer
- [ ] Importer et hériter de `PhysioBIDSWriter`
- [ ] Renommer méthode principale
- [ ] Modifier traitement pour générer par moment
- [ ] Changer signatures de méthodes
- [ ] Renommer colonnes de signaux
- [ ] Décompresser fichiers
- [ ] Tester sur 1 sujet

### Étape 3: Adapter BVP Writer
- [ ] Importer et hériter de `PhysioBIDSWriter`
- [ ] Simplifier format `processed_results`
- [ ] Harmoniser return types
- [ ] Tester sur 1 sujet

### Étape 4: Adapter EDA Writer
- [ ] Importer et hériter de `PhysioBIDSWriter`
- [ ] Ajouter calcul `EDA_Quality`
- [ ] Harmoniser return types
- [ ] Tester sur 1 sujet

### Étape 5: Tests d'Intégration
- [ ] Tester BVP + EDA + HR sur même sujet
- [ ] Vérifier cohérence des structures de fichiers
- [ ] Vérifier cohérence des noms de colonnes
- [ ] Valider avec visualizations

---

## Notes de Migration

### Breaking Changes

⚠️ **ATTENTION**: Ces changements nécessitent retraitement complet

1. **HR**: Fichiers `task-combined` → `task-restingstate`, `task-therapy`
2. **HR**: Colonnes `hr`, `quality` → `HR_Clean`, `HR_Quality`
3. **HR**: Décompression `.tsv.gz` → `.tsv`
4. **EDA**: Ajout colonne `EDA_Quality`
5. **Tous**: Return types harmonisés

### Script de Migration

Créer `scripts/migration/migrate_to_v2.py`:
- Lire anciens formats
- Convertir noms de colonnes
- Régénérer fichiers au nouveau format
- Valider cohérence

---

## Prochaines Étapes

Après Phase 2 complétée:
1. **Phase 3**: Refactoring métriques (noms majuscules)
2. **Phase 4**: Tests complets
3. **Phase 5**: Retraitement de toutes les données
4. **Phase 6**: Documentation mise à jour

---

**Status actuel**: Classe de base créée ✅  
**Prochaine action**: Adapter HRBIDSWriter
