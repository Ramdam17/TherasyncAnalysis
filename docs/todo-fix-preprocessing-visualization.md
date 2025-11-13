# TODO - Rendre les Visualisations Preprocessing Flexibles aux Moments

**Date de création** : 2025-11-12  
**Branche** : `feature/preprocessing-viz-redesign`  
**Objectif** : Rendre les visualisations robustes pour supporter un nombre variable de moments avec des noms arbitraires

---

## Problème Actuel

Les visualisations du preprocessing sont hard-codées pour exactement 2 moments : `restingstate` et `therapy`.

**Limitations** :
- ❌ Impossible d'ajouter de nouveaux moments (ex: "baseline", "intervention", "recovery")
- ❌ Ne fonctionne pas avec un seul moment
- ❌ Ne fonctionne pas avec plus de 2 moments
- ❌ Les noms de moments sont hard-codés dans le code

**Endroits concernés** :
1. `src/visualization/data_loader.py` : Boucles `for moment in ['restingstate', 'therapy']:`
2. `src/visualization/config.py` : Dictionnaires de couleurs et labels hard-codés
3. `src/visualization/plotters/hrv_plots.py` : Moments hard-codés
4. `src/visualization/plotters/signal_plots.py` : Moments hard-codés
5. `src/visualization/plotters/eda_plots.py` : Possiblement aussi concerné

---

## Solution Proposée

### 1. Détection Automatique des Moments

Au lieu de hard-coder les moments, les détecter automatiquement depuis :
- Les fichiers disponibles dans `data/derivatives/preprocessing/sub-XXX/ses-YY/{modality}/`
- Extraction du nom du moment depuis le pattern BIDS : `task-{moment}_`

**Exemple** :
```python
def discover_moments(subject: str, session: str, base_path: Path) -> List[str]:
    """
    Découvre automatiquement les moments disponibles pour un sujet/session.
    
    Scanne les fichiers preprocessing et extrait les noms de moments depuis
    les patterns BIDS (task-{moment}).
    
    Returns:
        Liste des moments trouvés (ex: ['restingstate', 'therapy', 'recovery'])
    """
    moments = set()
    for modality in ['bvp', 'eda', 'hr']:
        modality_dir = base_path / f"sub-{subject}" / f"ses-{session}" / modality
        if modality_dir.exists():
            for file in modality_dir.glob("*.tsv"):
                # Extract task-{moment} from filename
                match = re.search(r'task-(\w+)_', file.name)
                if match:
                    moments.add(match.group(1))
    return sorted(moments)
```

### 2. Configuration Dynamique des Couleurs

Au lieu de hard-coder les couleurs par moment, utiliser :
- Une palette de couleurs par défaut
- Attribution automatique selon l'ordre (moment 1 → couleur 1, etc.)
- Possibilité de surcharge dans config.yaml

**Exemple** :
```python
DEFAULT_MOMENT_COLORS = [
    '#3498db',  # Blue
    '#e74c3c',  # Red
    '#2ecc71',  # Green
    '#f39c12',  # Orange
    '#9b59b6',  # Purple
    '#1abc9c',  # Teal
]

def get_moment_color(moment: str, moment_index: int, config: dict = None) -> str:
    """
    Retourne la couleur pour un moment donné.
    
    Ordre de priorité:
    1. Couleur spécifiée dans config.yaml pour ce moment
    2. Couleur par défaut selon l'index
    3. Couleur de fallback si index trop grand
    """
    # Check config override
    if config and 'visualization' in config:
        moment_colors = config['visualization'].get('moment_colors', {})
        if moment in moment_colors:
            return moment_colors[moment]
    
    # Use default palette
    if moment_index < len(DEFAULT_MOMENT_COLORS):
        return DEFAULT_MOMENT_COLORS[moment_index]
    
    # Fallback: generate color
    return generate_color_from_index(moment_index)
```

### 3. Labels Dynamiques

Au lieu de hard-coder "Resting State" et "Therapy Session", générer automatiquement :

**Exemple** :
```python
def get_moment_label(moment: str, config: dict = None) -> str:
    """
    Retourne un label formaté pour un moment.
    
    Ordre de priorité:
    1. Label spécifié dans config.yaml
    2. Capitalisation du nom (restingstate → Resting State)
    """
    # Check config override
    if config and 'visualization' in config:
        moment_labels = config['visualization'].get('moment_labels', {})
        if moment in moment_labels:
            return moment_labels[moment]
    
    # Auto-generate: capitalize and replace underscores
    return moment.replace('_', ' ').title()
```

### 4. Layout Adaptatif

Les visualisations doivent s'adapter au nombre de moments :
- 1 moment : Pas de comparaison, juste affichage simple
- 2 moments : Layout actuel (2 subplots côte à côte)
- 3+ moments : Grid layout (2 colonnes, N/2 lignes arrondies)

---

## Plan de Modification

### Phase 1 : Détection Automatique des Moments

**Fichiers à modifier** :
- `src/visualization/data_loader.py`

**Changements** :
1. ✅ Ajouter fonction `discover_moments(subject, session, base_path)`
2. ✅ Remplacer `for moment in ['restingstate', 'therapy']:` par `for moment in moments:`
3. ✅ Passer `moments` comme paramètre aux méthodes de chargement

**Estimation** : 1h

---

### Phase 2 : Configuration Dynamique

**Fichiers à modifier** :
- `src/visualization/config.py`

**Changements** :
1. ✅ Remplacer `MOMENT_COLORS` dict par liste `DEFAULT_MOMENT_COLORS`
2. ✅ Créer `get_moment_color(moment, index, config)` avec logique de fallback
3. ✅ Créer `get_moment_label(moment, config)` avec auto-génération
4. ✅ Créer `get_moment_order(moment, moments_list)` pour ordre cohérent
5. ✅ Ajouter section `visualization.moment_colors` et `visualization.moment_labels` au config.yaml (optionnel)

**Estimation** : 1h

---

### Phase 3 : Adaptation des Plotters

**Fichiers à modifier** :
- `src/visualization/plotters/signal_plots.py`
- `src/visualization/plotters/hrv_plots.py`
- `src/visualization/plotters/eda_plots.py`

**Changements** :
1. ✅ Remplacer `moments = ['restingstate', 'therapy']` par paramètre `moments: List[str]`
2. ✅ Adapter les boucles pour itérer sur `moments`
3. ✅ Utiliser `get_moment_color()` au lieu d'accès direct au dict
4. ✅ Utiliser `get_moment_label()` au lieu de hard-coded labels
5. ✅ Gérer le cas où certains moments sont absents (skip gracefully)

**Estimation** : 2h

---

### Phase 4 : Layout Adaptatif

**Fichiers à modifier** :
- `src/visualization/plotters/signal_plots.py` (multi-signal dashboard)
- `src/visualization/plotters/hrv_plots.py` (poincaré, autonomic balance)
- `src/visualization/plotters/eda_plots.py` (arousal profile, SCR distribution)

**Changements** :
1. ✅ Fonction `calculate_subplot_layout(n_moments)` → (n_rows, n_cols)
2. ✅ Adapter création de subplots : `fig, axes = plt.subplots(n_rows, n_cols)`
3. ✅ Gérer cas spécial 1 moment : pas de comparaison visuelle
4. ✅ Ajuster tailles de figures selon nombre de moments

**Estimation** : 2h

---

### Phase 5 : Tests et Validation

**Fichiers à créer/modifier** :
- `tests/test_visualization_flexible_moments.py` (nouveau)
- Adapter tests existants si nécessaire

**Cas de test** :
1. ✅ 1 moment unique
2. ✅ 2 moments (comportement actuel)
3. ✅ 3 moments
4. ✅ 5 moments
5. ✅ Moments avec noms custom ("baseline", "intervention", "recovery")
6. ✅ Moment manquant pour un participant (gestion d'erreur)

**Estimation** : 2h

---

### Phase 6 : Documentation

**Fichiers à modifier** :
- `docs/resources.md` : Documenter flexibilité des moments
- `README.md` : Mentionner support multi-moments
- `config/config.yaml` : Ajouter exemples de configuration (optionnel)

**Estimation** : 1h

---

## Estimation Totale

**Durée estimée** : 9 heures

- Phase 1 : 1h (Détection automatique)
- Phase 2 : 1h (Configuration dynamique)
- Phase 3 : 2h (Adaptation plotters)
- Phase 4 : 2h (Layout adaptatif)
- Phase 5 : 2h (Tests)
- Phase 6 : 1h (Documentation)

---

## Bénéfices Attendus

1. ✅ **Flexibilité** : Support de n'importe quel nombre de moments
2. ✅ **Robustesse** : Gestion des moments manquants
3. ✅ **Extensibilité** : Facile d'ajouter de nouveaux moments sans changer le code
4. ✅ **Maintenabilité** : Moins de code hard-codé
5. ✅ **Généralisation** : Applicable à d'autres études avec différents protocoles

---

## Risques et Considérations

1. ⚠️ **Backward Compatibility** : S'assurer que le comportement actuel (2 moments) reste identique
2. ⚠️ **Performance** : Détection automatique ne doit pas ralentir le chargement
3. ⚠️ **Layout** : Avec beaucoup de moments (>6), les visualisations peuvent devenir illisibles
4. ⚠️ **Couleurs** : Palette limitée à ~6-8 couleurs distinctes
5. ⚠️ **Tests** : Nécessite des fixtures de test avec moments variables

---

## Questions Ouvertes

1. **Nombre maximum de moments supportés** : Limiter à 6-8 pour lisibilité ?
2. **Ordre des moments** : Alphabétique ? Chronologique ? Configurable ?
3. **Fallback colors** : Générer automatiquement ou limiter à palette fixe ?
4. **Layout alternatifs** : Tabs ? Accordéon ? Pour >6 moments ?

---

## État d'Avancement

- [ ] Phase 1 : Détection automatique des moments
- [ ] Phase 2 : Configuration dynamique
- [ ] Phase 3 : Adaptation des plotters
- [ ] Phase 4 : Layout adaptatif
- [ ] Phase 5 : Tests et validation
- [ ] Phase 6 : Documentation
