# Therasync Pipeline Quick Reference

## 🎯 Most Common Commands

### BVP Processing
```bash
# Single subject
poetry run python scripts/physio/preprocessing/preprocess_bvp.py -s f01p01 -e 01

# With config
poetry run python scripts/physio/preprocessing/preprocess_bvp.py -s f01p01 -e 01 -c config/config.yaml
```

### EDA Processing
```bash
# Single subject
poetry run python scripts/physio/preprocessing/preprocess_eda.py -s f01p01 -e 01

# Verbose output
poetry run python scripts/physio/preprocessing/preprocess_eda.py -s f01p01 -e 01 -v
```

### HR Processing
```bash
# Single subject
poetry run python scripts/physio/preprocessing/preprocess_hr.py -s f01p01 -e 01
```

### Clean Outputs (Dry Run First!)
```bash
# See what would be deleted
poetry run python scripts/utils/clean_outputs.py --all --dry-run

# Actually clean
poetry run python scripts/utils/clean_outputs.py --all
```

### Clean Specific Subject
```bash
poetry run python scripts/utils/clean_outputs.py -d -s f01p01 -e 01 -f
```

## 📂 Key File Locations

```
config/config.yaml                      # Main configuration
data/raw/                               # Input data (BIDS format)
data/derivatives/preprocessing/         # Pipeline outputs (modular)
  └── sub-{subject}/ses-{session}/
      ├── bvp/                          # BVP/HRV outputs
      ├── eda/                          # EDA outputs
      └── hr/                           # HR outputs
log/                                    # Log files
scripts/physio/preprocessing/           # Preprocessing scripts
  ├── preprocess_bvp.py
  ├── preprocess_eda.py
  └── preprocess_hr.py
scripts/utils/clean_outputs.py          # Cleanup utility
```

## 🔍 Check Your Data

### Input Structure (Expected)
```
data/raw/sub-f01p01/ses-01/physio/
├── sub-f01p01_ses-01_task-restingstate_recording-bvp.tsv
├── sub-f01p01_ses-01_task-restingstate_recording-bvp.json
├── sub-f01p01_ses-01_task-restingstate_recording-eda.tsv
├── sub-f01p01_ses-01_task-restingstate_recording-eda.json
├── sub-f01p01_ses-01_task-restingstate_recording-hr.tsv
└── sub-f01p01_ses-01_task-restingstate_recording-hr.json
```

### Output Structure (Generated)
```
data/derivatives/preprocessing/sub-f01p01/ses-01/
├── bvp/
│   ├── *_task-*_desc-processed_recording-bvp.tsv
│   ├── *_task-*_desc-processed_recording-bvp.json
│   ├── *_desc-bvp-metrics_physio.tsv
│   └── *_desc-bvp-metrics_physio.json
├── eda/
│   ├── *_task-*_desc-processed_recording-eda.tsv
│   ├── *_desc-eda-metrics_physio.tsv
│   └── *_desc-scr_events.tsv
└── hr/
    ├── *_task-combined_physio.tsv.gz
    ├── *_task-combined_hr-metrics.tsv
    └── *_task-combined_events.tsv
```

## ⚠️ Common Issues

| Error | Quick Fix |
|-------|-----------|
| Import errors | No longer needed - scripts handle paths automatically |
| File not found | Check BIDS naming in `data/raw/` |
| Insufficient peaks (BVP/HR) | Signal too short/noisy - check source data |
| Config errors | Compare with `config/config.yaml` |
| Module not found | Ensure you're in the project root directory |

## 📊 Output Validation

Quick checks after processing:
```bash
# List all outputs for a subject
tree data/derivatives/preprocessing/sub-f01p01/

# Check BVP metrics
cat data/derivatives/preprocessing/sub-f01p01/ses-01/bvp/*_desc-bvp-metrics_physio.tsv

# Check EDA metrics
cat data/derivatives/preprocessing/sub-f01p01/ses-01/eda/*_desc-eda-metrics_physio.tsv

# Check HR metrics
cat data/derivatives/preprocessing/sub-f01p01/ses-01/hr/*_hr-metrics.tsv

# View latest log
tail -50 log/*_preprocessing.log
```

## 🔄 Testing Iteration Loop

```bash
# 1. Process
PYTHONPATH=. poetry run python scripts/preprocess_bvp.py -s sub-f01p01 -e ses-01 -v

# 2. Check
ls -R data/derivatives/therasync-bvp/sub-f01p01/

# 3. Clean
PYTHONPATH=. poetry run python scripts/clean_outputs.py -d -s sub-f01p01 -e ses-01 -f

# Repeat!
```

## 💡 Pro Tips

- Always use `--dry-run` with clean script first
- Use `--verbose` for debugging
- Check logs in `log/` directory
- Keep `config/config.yaml` under version control
- Use `--continue-on-error` for batch processing

## 📚 Full Documentation

- **Testing Guide**: `docs/testing_guide.md`
- **Sprint 2 Summary**: `docs/sprint_2_summary.md`
- **Example Config**: `config/example_config.yaml`
