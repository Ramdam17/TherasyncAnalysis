# Therasync Pipeline - Quick Reference Card# Therasync Pipeline Quick Reference



One-page reference for common commands and workflows.## 🎯 Most Common Commands



## 🚀 Quick Start (Most Common)### BVP Processing

```bash

```bash# Single subject

# Process all data + generate all visualizationspoetry run python scripts/physio/preprocessing/preprocess_bvp.py -s f01p01 -e 01

poetry run python scripts/batch/run_all_preprocessing.py --skip-existing

poetry run python scripts/batch/run_all_visualizations.py# With config

poetry run python scripts/physio/preprocessing/preprocess_bvp.py -s f01p01 -e 01 -c config/config.yaml

# Generate quality report```

poetry run python scripts/analysis/generate_quality_report.py

```### EDA Processing

```bash

## 📦 Batch Processing# Single subject

poetry run python scripts/physio/preprocessing/preprocess_eda.py -s f01p01 -e 01

### Preprocessing (All Modalities)

```bash# Verbose output

# Process everythingpoetry run python scripts/physio/preprocessing/preprocess_eda.py -s f01p01 -e 01 -v

poetry run python scripts/batch/run_all_preprocessing.py```



# Common options### HR Processing

--skip-existing          # Skip already processed sessions```bash

--dry-run               # Preview without execution# Single subject

--verbose               # Detailed loggingpoetry run python scripts/physio/preprocessing/preprocess_hr.py -s f01p01 -e 01

--subjects f01p01 f02p01  # Specific subjects only```

```

### Clean Outputs (Dry Run First!)

### Visualization (6 Plots/Session)```bash

```bash# See what would be deleted

# Generate all visualizationspoetry run python scripts/utils/clean_outputs.py --all --dry-run

poetry run python scripts/batch/run_all_visualizations.py

# Actually clean

# Common optionspoetry run python scripts/utils/clean_outputs.py --all

--dry-run               # Preview```

--plots 1 6             # Specific plots only

--subjects f01p01       # Specific subjects### Clean Specific Subject

``````bash

poetry run python scripts/utils/clean_outputs.py -d -s f01p01 -e 01 -f

### Quality Analysis```

```bash

# Generate quality report## 📂 Key File Locations

poetry run python scripts/analysis/generate_quality_report.py

``````

config/config.yaml                      # Main configuration

## 🔧 Single Subject Processingdata/raw/                               # Input data (BIDS format)

data/derivatives/preprocessing/         # Pipeline outputs (modular)

```bash  └── sub-{subject}/ses-{session}/

# Set variables      ├── bvp/                          # BVP/HRV outputs

export SUBJECT=f01p01      ├── eda/                          # EDA outputs

export SESSION=01      └── hr/                           # HR outputs

log/                                    # Log files

# Individual modality pipelinesscripts/physio/preprocessing/           # Preprocessing scripts

poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject $SUBJECT --session $SESSION  ├── preprocess_bvp.py

poetry run python scripts/physio/preprocessing/preprocess_eda.py --subject $SUBJECT --session $SESSION  ├── preprocess_eda.py

poetry run python scripts/physio/preprocessing/preprocess_hr.py --subject $SUBJECT --session $SESSION  └── preprocess_hr.py

scripts/utils/clean_outputs.py          # Cleanup utility

# Generate visualizations```

poetry run python scripts/visualization/generate_visualizations.py --subject $SUBJECT --session $SESSION

```## 🔍 Check Your Data



## 📂 Directory Structure### Input Structure (Expected)

```

```data/raw/sub-f01p01/ses-01/physio/

TherasyncPipeline/├── sub-f01p01_ses-01_task-restingstate_recording-bvp.tsv

├── config/config.yaml          # Configuration├── sub-f01p01_ses-01_task-restingstate_recording-bvp.json

├── data/├── sub-f01p01_ses-01_task-restingstate_recording-eda.tsv

│   ├── raw/                    # Input: BIDS-formatted source data├── sub-f01p01_ses-01_task-restingstate_recording-eda.json

│   │   └── sub-*/ses-*/physio/├── sub-f01p01_ses-01_task-restingstate_recording-hr.tsv

│   └── derivatives/            # Output: Processed data└── sub-f01p01_ses-01_task-restingstate_recording-hr.json

│       ├── preprocessing/      # BVP, EDA, HR outputs```

│       │   └── sub-*/ses-*/

│       │       ├── bvp/        # 9 files per session### Output Structure (Generated)

│       │       ├── eda/        # 13 files per session```

│       │       └── hr/         # 14 files per session (7 per moment)data/derivatives/preprocessing/sub-f01p01/ses-01/

│       ├── visualization/      # 6 PNG plots per session├── bvp/

│       │   └── sub-*/ses-*/figures/│   ├── *_task-*_desc-processed_recording-bvp.tsv

│       └── reports/            # Quality analysis reports│   ├── *_task-*_desc-processed_recording-bvp.json

├── scripts/│   ├── *_desc-bvp-metrics_physio.tsv

│   ├── batch/                  # Batch processing scripts│   └── *_desc-bvp-metrics_physio.json

│   ├── physio/preprocessing/   # Individual pipeline scripts├── eda/

│   ├── visualization/          # Visualization generation│   ├── *_task-*_desc-processed_recording-eda.tsv

│   └── analysis/               # Quality analysis│   ├── *_desc-eda-metrics_physio.tsv

└── tests/                      # Unit tests (34 tests)│   └── *_desc-scr_events.tsv

```└── hr/

    ├── *_task-combined_physio.tsv.gz

## 📊 Output Files Per Session    ├── *_task-combined_hr-metrics.tsv

    └── *_task-combined_events.tsv

### BVP (9 files)```

- `*_task-{moment}_desc-processed_recording-bvp.tsv` (×2: restingstate, therapy)

- `*_task-{moment}_desc-processed_recording-bvp.json` (×2)## ⚠️ Common Issues

- `*_task-{moment}_desc-peaks_events.tsv` (×2)

- `*_task-{moment}_desc-peaks_events.json` (×2)| Error | Quick Fix |

- `*_desc-bvp-metrics_physio.tsv` (combined metrics)|-------|-----------|

| Import errors | No longer needed - scripts handle paths automatically |

### EDA (13 files)| File not found | Check BIDS naming in `data/raw/` |

- `*_task-{moment}_desc-processed_recording-eda.tsv` (×2)| Insufficient peaks (BVP/HR) | Signal too short/noisy - check source data |

- `*_task-{moment}_desc-processed_recording-eda.json` (×2)| Config errors | Compare with `config/config.yaml` |

- `*_task-{moment}_desc-scr_events.tsv` (×2)| Module not found | Ensure you're in the project root directory |

- `*_task-{moment}_desc-scr_events.json` (×2)

- `*_task-{moment}_desc-eda-metrics_physio.json` (×2)## 📊 Output Validation

- `*_desc-eda-metrics_physio.tsv` (combined metrics)

- `*_desc-eda-summary.json`Quick checks after processing:

- `*_desc-eda-summary.txt````bash

# List all outputs for a subject

### HR (14 files)tree data/derivatives/preprocessing/sub-f01p01/

- `*_task-{moment}_desc-processed_recording-hr.tsv` (×2)

- `*_task-{moment}_desc-processed_recording-hr.json` (×2)# Check BVP metrics

- `*_task-{moment}_desc-outliers_events.tsv` (×2)cat data/derivatives/preprocessing/sub-f01p01/ses-01/bvp/*_desc-bvp-metrics_physio.tsv

- `*_task-{moment}_desc-outliers_events.json` (×2)

- `*_task-{moment}_desc-hr-metrics_physio.json` (×2)# Check EDA metrics

- `*_desc-hr-metrics_physio.tsv` (combined metrics)cat data/derivatives/preprocessing/sub-f01p01/ses-01/eda/*_desc-eda-metrics_physio.tsv

- `*_desc-hr-summary.json`

- `*_desc-hr-summary.txt`# Check HR metrics

cat data/derivatives/preprocessing/sub-f01p01/ses-01/hr/*_hr-metrics.tsv

### Visualizations (6 files)

1. `01_dashboard_multisignals.png` - Multi-signal overview# View latest log

2. `02_poincare_hrv.png` - HRV Poincaré plottail -50 log/*_preprocessing.log

3. `03_autonomic_balance.png` - LF/HF ratio timeline```

4. `04_eda_arousal_profile.png` - EDA tonic/phasic with SCR

5. `05_scr_distribution.png` - SCR statistics## 🔄 Testing Iteration Loop

6. `06_hr_dynamics_timeline.png` - HR evolution with zones

```bash

## 🧪 Testing# 1. Process

PYTHONPATH=. poetry run python scripts/preprocess_bvp.py -s sub-f01p01 -e ses-01 -v

```bash

# Run all tests (should be 34/34 passing)# 2. Check

poetry run pytest tests/ -vls -R data/derivatives/therasync-bvp/sub-f01p01/



# Run with coverage# 3. Clean

poetry run pytest --cov=src tests/PYTHONPATH=. poetry run python scripts/clean_outputs.py -d -s sub-f01p01 -e ses-01 -f



# Run specific test file# Repeat!

poetry run pytest tests/test_bvp_pipeline.py -v```

```

## 💡 Pro Tips

## ⚡ Performance Benchmarks

- Always use `--dry-run` with clean script first

| Task | Sessions | Time | Success Rate |- Use `--verbose` for debugging

|------|----------|------|--------------|- Check logs in `log/` directory

| Preprocessing | 51 | ~3 min | 96% (49/51) |- Keep `config/config.yaml` under version control

| Visualization | 51 | ~3 min | 100% (306/306) |- Use `--continue-on-error` for batch processing

| Quality Report | 51 | <1 sec | 100% |

## 📚 Full Documentation

**System Requirements:** <2 GB RAM, ~2.5 GB disk space for full dataset

- **Testing Guide**: `docs/testing_guide.md`

## 🔍 Quick Checks- **Sprint 2 Summary**: `docs/sprint_2_summary.md`

- **Example Config**: `config/example_config.yaml`

```bash
# Count processed sessions
find data/derivatives/preprocessing -name "*_bvp-metrics.tsv" | wc -l  # Should be 49-51

# Count visualizations
find data/derivatives/visualization -name "*.png" | wc -l  # Should be 306

# Check for errors in logs
grep -i "error\|failed" *.log

# View specific session
tree data/derivatives/preprocessing/sub-f01p01/ses-01/
ls data/derivatives/visualization/sub-f01p01/ses-01/figures/
```

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| Poetry not found | Run in `TherasyncPipeline/` directory, not parent folder |
| 2 preprocessing failures | Expected - sessions have missing/empty source data |
| Import errors | Use `poetry run python` prefix for all commands |
| Empty visualizations | Ensure preprocessing completed before visualization |
| Tests failing | Should be 34/34 passing in current version |

## 📚 Documentation Links

- **Full Guide**: `README.md`
- **Quick Start**: `QUICKSTART.md`
- **API Reference**: `docs/api_reference.md`
- **Testing Guide**: `docs/testing_guide.md`
- **Troubleshooting**: `docs/troubleshooting.md`

## 💡 Tips

- Always use `--skip-existing` for incremental updates
- Use `--dry-run` to preview batch operations
- Check quality report after full preprocessing
- Logs are saved with timestamped filenames
- All outputs are BIDS-compliant and self-documented

---

**Version**: 1.0.0 | **Status**: Production Ready | **Tests**: 34/34 Passing
