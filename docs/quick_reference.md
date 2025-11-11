# Therasync Pipeline - Quick Reference Card

A comprehensive command reference for the Therasync physiological data processing pipeline.

## Quick Links

- **Installation Guide**: `README.md` - Setup instructions
- **Quick Start**: `QUICKSTART.md` - Fast processing guide  
- **API Documentation**: `docs/api_reference.md` - Complete API reference
- **Troubleshooting**: `docs/troubleshooting.md` - Common issues

---

## Batch Processing Commands

### Preprocessing (All Subjects)

```bash
# Process all subjects/sessions
poetry run python scripts/batch/run_all_preprocessing.py

# Skip already processed (recommended)
poetry run python scripts/batch/run_all_preprocessing.py --skip-existing

# Preview without executing
poetry run python scripts/batch/run_all_preprocessing.py --dry-run

# Process specific subjects
poetry run python scripts/batch/run_all_preprocessing.py --subjects f01p01 f02p01

# Verbose output
poetry run python scripts/batch/run_all_preprocessing.py --verbose
```

**Outputs**: `data/derivatives/preprocessing/sub-*/ses-*/{bvp,eda,hr}/`  
**Logs**: `log/batch_preprocessing_YYYYMMDD_HHMMSS.log`  
**Time**: ~45 minutes for 51 sessions

### Visualization (All Subjects)

```bash
# Generate all visualizations
poetry run python scripts/batch/run_all_visualizations.py

# Preview without generating
poetry run python scripts/batch/run_all_visualizations.py --dry-run

# Specific plots only (1-6)
poetry run python scripts/batch/run_all_visualizations.py --plots 1 2 3

# Specific subjects
poetry run python scripts/batch/run_all_visualizations.py --subjects f01p01

# Verbose output
poetry run python scripts/batch/run_all_visualizations.py --verbose
```

**Outputs**: `data/derivatives/visualization/sub-*/ses-*/figures/*.png`  
**Logs**: `log/batch_visualization_YYYYMMDD_HHMMSS.log`  
**Time**: ~3 minutes for 51 sessions (6 plots each)

---

## Single Subject Processing

### BVP Preprocessing

```bash
# Basic usage
poetry run python scripts/physio/preprocessing/preprocess_bvp.py \
  --subject f01p01 --session 01

# Custom config
poetry run python scripts/physio/preprocessing/preprocess_bvp.py \
  --subject f01p01 --session 01 --config config/custom.yaml

# View outputs
tree data/derivatives/preprocessing/sub-f01p01/ses-01/bvp/
```

**Outputs** (9 files):
- Cleaned signals, peaks, quality metrics
- HRV metrics (20 measures)
- Time/frequency/non-linear domain metrics

**Processing time**: ~1 minute per session

### EDA Preprocessing

```bash
# Basic usage
poetry run python scripts/physio/preprocessing/preprocess_eda.py \
  --subject f01p01 --session 01

# View SCR events
head data/derivatives/preprocessing/sub-f01p01/ses-01/eda/*_scr_events.tsv

# View metrics
cat data/derivatives/preprocessing/sub-f01p01/ses-01/eda/*_eda-metrics_physio.tsv
```

**Outputs** (13 files):
- Cleaned signals, tonic/phasic components
- SCR events and peaks
- EDA metrics (23 measures)

**Processing time**: ~1 second per session

### HR Preprocessing

```bash
# Basic usage
poetry run python scripts/physio/preprocessing/preprocess_hr.py \
  --subject f01p01 --session 01

# View metrics
cat data/derivatives/preprocessing/sub-f01p01/ses-01/hr/*_hr-metrics.tsv
```

**Outputs** (7 files):
- Cleaned HR signals, outliers removed
- HR metrics (26 measures)
- Quality and variability metrics

**Processing time**: ~1 second per session

### Visualization (Single Subject)

```bash
# Generate all 6 plots
poetry run python scripts/visualization/generate_visualizations.py \
  --subject f01p01 --session 01

# Custom config
poetry run python scripts/visualization/generate_visualizations.py \
  --subject f01p01 --session 01 --config config/custom.yaml

# View outputs
ls data/derivatives/visualization/sub-f01p01/ses-01/figures/
```

**Outputs** (6 PNG files):
1. `01_dashboard_multisignals.png` - Multi-signal overview
2. `02_poincare_hrv.png` - HRV Poincaré plot
3. `03_autonomic_balance.png` - LF/HF timeline
4. `04_eda_arousal_profile.png` - EDA decomposition
5. `05_scr_distribution.png` - SCR histogram
6. `06_hr_dynamics_timeline.png` - HR evolution

**Processing time**: ~3 seconds per session

---

## Configuration

### Main Config File

```bash
# Location
config/config.yaml

# Edit configuration
nano config/config.yaml

# Validate configuration
poetry run python -c "from src.core.config_loader import ConfigLoader; ConfigLoader('config/config.yaml')"
```

### Key Configuration Sections

```yaml
# Moments (tasks)
moments:
  - name: restingstate
    duration: 60
  - name: therapy
    is_default: true

# BVP Processing
bvp:
  cleaning:
    method: "neurokit"
    algorithm: "elgendi"
  
# EDA Processing
eda:
  cleaning:
    method: "neurokit"
  decomposition:
    method: "cvxeda"

# HR Processing
hr:
  outlier_detection:
    method: "percentile"
    lower_percentile: 1
    upper_percentile: 99

# Visualization
visualization:
  dpi: 300
  figure_format: "png"
```

---

## Utility Scripts

### Clean Outputs

```bash
# Clean all derivatives
poetry run python scripts/utils/clean_outputs.py --derivatives --force

# Clean specific modality
poetry run python scripts/utils/clean_outputs.py --derivatives --modality bvp

# Dry run (preview)
poetry run python scripts/utils/clean_outputs.py --derivatives
```

### View Logs

```bash
# Latest batch preprocessing log
tail -f log/batch_preprocessing_*.log | tail -1

# Latest batch visualization log
tail -f log/batch_visualization_*.log | tail -1

# Pipeline-specific logs
tail -f log/bvp_preprocessing.log
tail -f log/eda_preprocessing.log
tail -f log/hr_preprocessing.log
```

### Check Processing Status

```bash
# Count preprocessed sessions
find data/derivatives/preprocessing -name "*_bvp-metrics.tsv" | wc -l
find data/derivatives/preprocessing -name "*_eda-metrics_physio.tsv" | wc -l
find data/derivatives/preprocessing -name "*_hr-metrics.tsv" | wc -l

# Count visualizations
find data/derivatives/visualization -name "*.png" | wc -l

# List all processed subjects
ls data/derivatives/preprocessing/
```

---

## Data Organization

### Input (BIDS Format)

```
data/raw/
└── sub-fXXpYY/          # Family XX, Participant YY
    └── ses-XX/          # Session number
        └── physio/      # Physiological data
            ├── *_task-restingstate_recording-bvp.tsv
            ├── *_task-restingstate_recording-eda.tsv
            ├── *_task-restingstate_recording-hr.tsv
            ├── *_task-therapy_recording-bvp.tsv
            ├── *_task-therapy_recording-eda.tsv
            └── *_task-therapy_recording-hr.tsv
```

### Preprocessing Outputs

```
data/derivatives/preprocessing/
└── sub-fXXpYY/
    └── ses-XX/
        ├── bvp/    # 9 files (signals, peaks, HRV metrics)
        ├── eda/    # 13 files (signals, SCR events, metrics)
        └── hr/     # 7 files (signals, metrics)
```

### Visualization Outputs

```
data/derivatives/visualization/
└── sub-fXXpYY/
    └── ses-XX/
        └── figures/  # 6 PNG files
```

---

## Common Workflows

### Full Dataset Processing

```bash
# 1. Process all subjects (preprocessing)
poetry run python scripts/batch/run_all_preprocessing.py

# 2. Generate all visualizations
poetry run python scripts/batch/run_all_visualizations.py

# 3. Check results
tail -30 log/batch_preprocessing_*.log
tail -30 log/batch_visualization_*.log
```

### Incremental Processing (New Data)

```bash
# Only process new subjects/sessions
poetry run python scripts/batch/run_all_preprocessing.py --skip-existing
poetry run python scripts/batch/run_all_visualizations.py
```

### Single Subject Complete Pipeline

```bash
export SUBJECT=f01p01
export SESSION=01

# Preprocessing
poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject $SUBJECT --session $SESSION
poetry run python scripts/physio/preprocessing/preprocess_eda.py --subject $SUBJECT --session $SESSION
poetry run python scripts/physio/preprocessing/preprocess_hr.py --subject $SUBJECT --session $SESSION

# Visualization
poetry run python scripts/visualization/generate_visualizations.py --subject $SUBJECT --session $SESSION
```

### Regenerate Visualizations Only

```bash
# Preprocessing already done, update plots
poetry run python scripts/batch/run_all_visualizations.py

# Or specific subjects
poetry run python scripts/batch/run_all_visualizations.py --subjects f01p01 f02p01
```

### Clean and Restart

```bash
# Remove all outputs
poetry run python scripts/utils/clean_outputs.py --derivatives --force

# Reprocess everything
poetry run python scripts/batch/run_all_preprocessing.py
poetry run python scripts/batch/run_all_visualizations.py
```

---

## Visualization Details

### Available Plots (1-6)

| # | Filename | Description | Key Metrics |
|---|----------|-------------|-------------|
| 1 | `01_dashboard_multisignals.png` | Multi-signal overview | BVP, EDA, HR with moment markers |
| 2 | `02_poincare_hrv.png` | HRV Poincaré plot | SD1, SD2, SD1/SD2 ratio |
| 3 | `03_autonomic_balance.png` | Autonomic balance | LF/HF ratio over time |
| 4 | `04_eda_arousal_profile.png` | EDA arousal profile | Tonic/phasic, SCR events |
| 5 | `05_scr_distribution.png` | SCR distribution | Amplitude histogram, stats |
| 6 | `06_hr_dynamics_timeline.png` | HR dynamics | Heart rate + variability bands |

### Generate Specific Plots

```bash
# Only plots 1, 2, and 3
poetry run python scripts/batch/run_all_visualizations.py --plots 1 2 3

# For single subject
poetry run python scripts/visualization/generate_visualizations.py \
  --subject f01p01 --session 01
```

---

## Metrics Reference

### BVP/HRV Metrics (20 total)

**Time-domain** (8):
- Mean HR, SDNN, RMSSD, pNN50, NN50, SD1, SD2, SD1/SD2

**Frequency-domain** (7):
- LF power, HF power, LF/HF ratio, Total power, VLF power, LF%, HF%

**Non-linear** (5):
- Sample entropy, Approximate entropy, DFA α1/α2, Correlation dimension

### EDA Metrics (23 total)

**SCR Metrics** (9):
- Peak count, amplitude mean/SD, rise time, recovery time, frequency

**Tonic Metrics** (5):
- Mean, SD, min, max, range

**Phasic Metrics** (5):
- Mean, SD, min, max, range

**Temporal Metrics** (4):
- First/last peak times, peak rate, AUC

### HR Metrics (26 total)

**Basic** (6):
- Mean, median, SD, min, max, range

**Variability** (8):
- RMSSD, SDSD, CV, IQR, MAD, stability, smoothness

**Quality** (7):
- Coverage, interpolated%, outlier%, gap count, longest gap

**Temporal** (5):
- Trend, first/last values, range%, AUC

---

## Troubleshooting

### Common Errors

```bash
# "Poetry could not find pyproject.toml"
cd /path/to/TherasyncPipeline  # Ensure in correct directory

# "Empty HR data" error
# Check if raw file exists and contains data
wc -l data/raw/sub-fXXpYY/ses-XX/physio/*_hr.tsv

# Plots appear empty
# Verify preprocessing completed successfully
ls data/derivatives/preprocessing/sub-fXXpYY/ses-XX/*/

# Check for errors
grep -i "error\|failed" log/batch_*.log
```

### Performance Issues

```bash
# Check disk space
df -h data/

# Check memory usage
free -h

# Monitor batch processing
tail -f log/batch_preprocessing_*.log
```

---

## Environment

### Dependencies

```bash
# Install/update dependencies
poetry install

# Update single package
poetry update neurokit2

# Show installed packages
poetry show

# Export requirements.txt
poetry export -f requirements.txt -o requirements.txt
```

### Python Environment

```bash
# Activate poetry shell
poetry shell

# Run without activating
poetry run python script.py

# Check Python version
poetry run python --version
```

---

## For Developers

### Code Quality

```bash
# Run tests
poetry run pytest tests/

# Test coverage
poetry run pytest --cov=src tests/

# Linting
poetry run black src/ scripts/
poetry run flake8 src/ scripts/
```

### Documentation

```bash
# Generate API docs
poetry run pdoc --html --output-dir docs/api src/

# Update TODO
nano TODO.md
```

---

## Performance Benchmarks

| Operation | Time per Session | Total (51 sessions) |
|-----------|------------------|---------------------|
| BVP Preprocessing | ~60 seconds | ~51 minutes |
| EDA Preprocessing | ~1 second | ~51 seconds |
| HR Preprocessing | ~1 second | ~51 seconds |
| **Total Preprocessing** | **~62 seconds** | **~53 minutes** |
| Visualization (6 plots) | ~3 seconds | ~3 minutes |
| **Grand Total** | **~65 seconds** | **~56 minutes** |

**Disk Usage**:
- Preprocessing: ~30 MB per session
- Visualization: ~5 MB per session (6 plots @ 300 DPI)
- **Total**: ~35 MB per session × 51 = ~1.8 GB

---

## Quick Help

```bash
# Show script help
poetry run python scripts/batch/run_all_preprocessing.py --help
poetry run python scripts/batch/run_all_visualizations.py --help

# Check version
cat pyproject.toml | grep version

# View README
cat README.md | less

# View this reference
cat docs/quick_reference.md | less
```

---

**For complete documentation**: See `README.md` and `docs/` directory  
**For API reference**: See `docs/api_reference.md`  
**For troubleshooting**: See `docs/troubleshooting.md`
