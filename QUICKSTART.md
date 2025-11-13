# Therasync Pipeline - Quick Start Guide

Get started processing physiological data and generating visualizations in minutes.

**Current Status**: Production ready with 96% preprocessing success rate and 100% visualization success rate.

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
# Process all 51 sessions (~3 minutes on modern hardware)
poetry run python scripts/batch/run_all_preprocessing.py

# Skip already processed sessions (recommended for incremental updates)
poetry run python scripts/batch/run_all_preprocessing.py --skip-existing

# Preview what will be processed (no execution)
poetry run python scripts/batch/run_all_preprocessing.py --dry-run

# Verbose mode with detailed logging
poetry run python scripts/batch/run_all_preprocessing.py --verbose
```

**Expected Results**:
- 49/51 sessions successfully processed (96% success rate)
- 2 sessions may fail due to missing/empty source data
- Output: `data/derivatives/preprocessing/sub-*/ses-*/` (BVP, EDA, HR data)
- Total files: ~1,470 BIDS-compliant output files (30 per successful session)

### 2. Generate All Visualizations

```bash
# Generate 6 plots for each preprocessed session (~3 minutes)
poetry run python scripts/batch/run_all_visualizations.py

# Preview what will be generated
poetry run python scripts/batch/run_all_visualizations.py --dry-run

# Generate specific plots only (plots 1-6)
poetry run python scripts/batch/run_all_visualizations.py --plots 1 6
```

**Expected Results**:
- 306/306 visualizations generated successfully (100% success rate)
- 51 sessions × 6 plots each
- Output: `data/derivatives/visualization/sub-*/ses-*/figures/` (6 PNG files per session)
- Publication-quality 300 DPI images

### 3. Generate Quality Report

```bash
# Generate comprehensive quality analysis
poetry run python scripts/analysis/generate_quality_report.py

# View the report
cat data/derivatives/reports/quality_report_YYYYMMDD.txt
```

**Expected Results**:
- 114 quality flags identified across all sessions
- Signal quality assessment for BVP, EDA, and HR
- Per-session and aggregate statistics

### 4. Check Results

```bash
# View processing summary
tail -30 visualization_with_hr_fixed.log

# Count successful outputs
find data/derivatives/preprocessing -type d -name "bvp" | wc -l  # Should be 49-51
find data/derivatives/visualization -name "*.png" | wc -l        # Should be 306

# Browse specific session
tree data/derivatives/preprocessing/sub-g01p01/ses-01/
tree data/derivatives/visualization/sub-g01p01/ses-01/figures/
```

## Processing Single Subject

For testing or re-processing specific subjects:

```bash
# Set subject and session
export SUBJECT=g01p01
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
poetry run python scripts/batch/run_all_preprocessing.py --subjects g01p01 g02p01 g03p01

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
poetry run python scripts/batch/run_all_visualizations.py --subjects g01p01 g02p01

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
│   └── sub-g01p01/
│       └── ses-01/
│           ├── bvp/    # 9 files: processed signals, peaks, HRV metrics
│           ├── eda/    # 13 files: signals, tonic/phasic, SCR events, metrics
│           └── hr/     # 14 files: 7 per moment (restingstate + therapy)
├── visualization/
│   └── sub-g01p01/
│       └── ses-01/
│           └── figures/  # 6 PNG files (300 DPI, publication-ready)
└── reports/
    └── quality_report_YYYYMMDD.txt  # Comprehensive quality analysis
```

## Logs

Batch processing output is saved to log files for tracking:

```bash
# View latest preprocessing log
cat preprocessing_phase2_complete.log

# View latest visualization log  
cat visualization_with_hr_fixed.log

# Check for any errors
grep -i "error\|failed" *.log
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

**Issue**: Some sessions show "Empty HR data" or preprocessing failures  
**Solution**: This is expected for 2 sessions with missing/empty source data. The pipeline handles this gracefully and continues processing other sessions.

**Issue**: "Poetry could not find pyproject.toml"  
**Solution**: Ensure you're in the `TherasyncPipeline` directory (not the parent `Therasync` folder).

**Issue**: HR data not appearing in visualizations  
**Solution**: Ensure you're using the latest code (Phase 2 complete). The visualization pipeline was updated to handle per-moment HR data structure.

**Issue**: Tests failing  
**Solution**: Run `poetry run pytest tests/ -v` to see which tests are failing. All 56 tests should pass in the current version.

## DPPA (Dyadic Poincaré Plot Analysis)

After preprocessing, you can analyze physiological synchrony between participants using DPPA.

### Step 1: Compute Poincaré Centroids

```bash
# Batch: Compute centroids for all participants/sessions
poetry run python scripts/physio/dppa/compute_poincare.py --batch
```

**Output**: `data/derivatives/dppa/sub-{participant}/ses-{session}/poincare/`  
**Files**: 606 centroid files (2 methods × 2 tasks × 51 sessions)  
**Time**: ~2 minutes for full dataset

### Step 2: Calculate Inter-Centroid Distances (ICDs)

```bash
# Compute ICDs for all dyad pairs
poetry run python scripts/physio/dppa/compute_dppa.py --mode both --task all --batch
```

**Outputs**:
- Inter-session: `data/derivatives/dppa/inter_session/` (~1,275 dyad pairs)
- Intra-family: `data/derivatives/dppa/intra_family/` (81 dyad pairs)
**Format**: Rectangular CSV (epochs × dyads)  
**Time**: ~5 minutes for full dataset

### Interpret Results

Open the CSV files in your analysis software. ICD values indicate synchrony:
- **0-20 ms**: High synchrony (similar autonomic states)
- **20-50 ms**: Moderate synchrony
- **50+ ms**: Low synchrony (divergent states)

**Typical workflow**:
1. Filter dyads of interest (e.g., parent-child, therapist-family)
2. Compare therapy vs. restingstate ICDs (baseline)
3. Track ICD evolution across epochs (temporal dynamics)

## Performance

**Current benchmarks (tested on real dataset):**
- **Preprocessing**: 49/51 sessions completed successfully (96%)
  - ~3-4 seconds per session
  - ~3 minutes total for full dataset
- **Visualization**: 306/306 plots generated (100%)
  - ~3.4 seconds per session (6 plots)
  - ~3 minutes total for full dataset
- **Quality Analysis**: <1 second for full dataset report

**Resource requirements:**
- **Disk space**: ~50 MB per session (preprocessing + visualization)
- **Memory**: <2 GB RAM per process
- **CPU**: Single-threaded (parallelization possible for future optimization)

## Next Steps

- View complete documentation in `README.md`
- Check API reference in `docs/api_reference.md`
- See command reference in `docs/quick_reference.md`
- Review batch processing logs for any errors

---

**Need help?** See `docs/troubleshooting.md` or check the full `README.md`
