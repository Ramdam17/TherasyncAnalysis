# Therasync Pipeline - Quick Start Guide

A minimal guide to get started processing physiological data and generating visualizations.

## Installation (One-time setup)

```bash
# Clone and setup
git clone <repository-url>
cd TherasyncPipeline
poetry install
```

## Processing All Data (Batch Mode)

The fastest way to process your entire dataset:

### 1. Preprocess All Subjects

```bash
# Process all 51 sessions (~45 minutes)
poetry run python scripts/batch/run_all_preprocessing.py

# Skip already processed sessions (recommended after first run)
poetry run python scripts/batch/run_all_preprocessing.py --skip-existing

# Preview what will be processed (no execution)
poetry run python scripts/batch/run_all_preprocessing.py --dry-run
```

**Output**: `data/derivatives/preprocessing/sub-*/ses-*/` (BVP, EDA, HR data)

### 2. Generate All Visualizations

```bash
# Generate 6 plots for each preprocessed session (~3 minutes)
poetry run python scripts/batch/run_all_visualizations.py

# Preview what will be generated
poetry run python scripts/batch/run_all_visualizations.py --dry-run
```

**Output**: `data/derivatives/visualization/sub-*/ses-*/figures/` (6 PNG files per session)

### 3. Check Results

```bash
# View preprocessing summary
tail -30 log/batch_preprocessing_*.log

# View visualization summary
tail -30 log/batch_visualization_*.log

# Browse output files
tree data/derivatives/preprocessing/ | head -50
tree data/derivatives/visualization/ | head -50
```

## Processing Single Subject

For testing or re-processing specific subjects:

```bash
# Set subject and session
export SUBJECT=f01p01
export SESSION=01

# Run preprocessing pipeline
poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject $SUBJECT --session $SESSION
poetry run python scripts/physio/preprocessing/preprocess_eda.py --subject $SUBJECT --session $SESSION
poetry run python scripts/physio/preprocessing/preprocess_hr.py --subject $SUBJECT --session $SESSION

# Generate visualizations
poetry run python scripts/visualization/generate_visualizations.py --subject $SUBJECT --session $SESSION

# View results
ls data/derivatives/preprocessing/sub-$SUBJECT/ses-$SESSION/*/
ls data/derivatives/visualization/sub-$SUBJECT/ses-$SESSION/figures/
```

## Batch Processing Options

### Preprocessing Options

```bash
# Process specific subjects only
poetry run python scripts/batch/run_all_preprocessing.py --subjects f01p01 f02p01 f03p01

# Skip already processed (incremental updates)
poetry run python scripts/batch/run_all_preprocessing.py --skip-existing

# Dry run (preview without execution)
poetry run python scripts/batch/run_all_preprocessing.py --dry-run

# Verbose logging
poetry run python scripts/batch/run_all_preprocessing.py --verbose
```

### Visualization Options

```bash
# Generate specific plots only (1-6)
poetry run python scripts/batch/run_all_visualizations.py --plots 1 2 3

# Process specific subjects
poetry run python scripts/batch/run_all_visualizations.py --subjects f01p01 f02p01

# Dry run
poetry run python scripts/batch/run_all_visualizations.py --dry-run
```

## Generated Visualizations

Each subject/session gets 6 publication-ready plots:

1. **`01_dashboard_multisignals.png`** - Multi-signal overview (BVP, EDA, HR)
2. **`02_poincare_hrv.png`** - HRV Poincaré plot (SD1/SD2 analysis)
3. **`03_autonomic_balance.png`** - LF/HF ratio timeline
4. **`04_eda_arousal_profile.png`** - EDA tonic/phasic decomposition with SCR events
5. **`05_scr_distribution.png`** - SCR amplitude distribution and statistics
6. **`06_hr_dynamics_timeline.png`** - Heart rate evolution with variability

## Output Structure

```
data/derivatives/
├── preprocessing/
│   └── sub-f01p01/
│       └── ses-01/
│           ├── bvp/    # 9 files (signals, peaks, metrics, HRV)
│           ├── eda/    # 13 files (signals, SCR events, metrics)
│           └── hr/     # 7 files (signals, metrics)
└── visualization/
    └── sub-f01p01/
        └── ses-01/
            └── figures/  # 6 PNG files
```

## Logs

All batch processing creates timestamped log files:

```bash
# Preprocessing logs
log/batch_preprocessing_YYYYMMDD_HHMMSS.log

# Visualization logs
log/batch_visualization_YYYYMMDD_HHMMSS.log

# Individual pipeline logs
log/bvp_preprocessing.log
log/eda_preprocessing.log
log/hr_preprocessing.log
```

## Common Workflows

### Full Pipeline (All Data)

```bash
# Complete processing
poetry run python scripts/batch/run_all_preprocessing.py
poetry run python scripts/batch/run_all_visualizations.py
```

### Incremental Updates (New Data Added)

```bash
# Only process new subjects/sessions
poetry run python scripts/batch/run_all_preprocessing.py --skip-existing
poetry run python scripts/batch/run_all_visualizations.py
```

### Regenerate Visualizations Only

```bash
# Preprocessing already done, just update plots
poetry run python scripts/batch/run_all_visualizations.py
```

### Clean and Restart

```bash
# Remove all outputs and restart
poetry run python scripts/utils/clean_outputs.py --derivatives --force
poetry run python scripts/batch/run_all_preprocessing.py
poetry run python scripts/batch/run_all_visualizations.py
```

## Troubleshooting

### Check Processing Status

```bash
# Count processed sessions
find data/derivatives/preprocessing -name "*_bvp-metrics.tsv" | wc -l

# Count generated visualizations
find data/derivatives/visualization -name "*.png" | wc -l

# Check for errors
grep -i "error\|failed" log/batch_preprocessing_*.log
```

### Common Issues

**Issue**: "Empty HR data" error  
**Solution**: Check if raw HR file exists and contains data (some sessions may have missing recordings)

**Issue**: "Poetry could not find pyproject.toml"  
**Solution**: Ensure you're in the `TherasyncPipeline` directory

**Issue**: Plots look empty  
**Solution**: Verify preprocessing completed successfully before generating visualizations

## Performance

**Typical execution times:**
- Preprocessing: ~1 minute per session (51 sessions ≈ 45 minutes total)
- Visualization: ~3 seconds per session (51 sessions ≈ 3 minutes total)

**Resource requirements:**
- Disk space: ~50 MB per session (preprocessing + visualization)
- Memory: <2 GB RAM
- CPU: Single-threaded (parallelization planned for future)

## Next Steps

- View complete documentation in `README.md`
- Check API reference in `docs/api_reference.md`
- See command reference in `docs/quick_reference.md`
- Review batch processing logs for any errors

---

**Need help?** See `docs/troubleshooting.md` or check the full `README.md`
