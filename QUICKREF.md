# BVP Pipeline Quick Reference

## 🎯 Most Common Commands

### Single Subject Processing
```bash
PYTHONPATH=. poetry run python scripts/preprocess_bvp.py -s sub-f01p01 -e ses-01 -v
```

### Clean Outputs (Dry Run First!)
```bash
# See what would be deleted
PYTHONPATH=. poetry run python scripts/clean_outputs.py --all --dry-run

# Actually clean
PYTHONPATH=. poetry run python scripts/clean_outputs.py --all
```

### Clean Specific Subject
```bash
PYTHONPATH=. poetry run python scripts/clean_outputs.py -d -s sub-f01p01 -e ses-01 -f
```

## 📂 Key File Locations

```
config/config.yaml          # Main configuration
data/sourcedata/            # Input BVP data (BIDS format)
data/derivatives/therasync-bvp/  # Pipeline outputs
log/                        # Log files
scripts/preprocess_bvp.py   # Main processing script
scripts/clean_outputs.py    # Cleanup utility
```

## 🔍 Check Your Data

### Input Structure (Expected)
```
data/sourcedata/sub-f01p01/ses-01/physio/
├── sub-f01p01_ses-01_task-restingstate_recording-bvp.tsv
├── sub-f01p01_ses-01_task-restingstate_recording-bvp.json
├── sub-f01p01_ses-01_task-therapy_recording-bvp.tsv
└── sub-f01p01_ses-01_task-therapy_recording-bvp.json
```

### Output Structure (Generated)
```
data/derivatives/therasync-bvp/sub-f01p01/ses-01/physio/
├── *_task-*_recording-bvp_physio.tsv      # Processed signals
├── *_task-*_recording-bvp_physio.json     # Signal metadata
├── *_desc-bvpmetrics_physio.tsv           # HRV metrics
└── *_desc-bvpmetrics_physio.json          # Metrics metadata
```

## ⚠️ Common Issues

| Error | Quick Fix |
|-------|-----------|
| Import errors | Use `PYTHONPATH=.` before commands |
| File not found | Check BIDS naming in `data/sourcedata/` |
| Insufficient peaks | Signal too short/noisy - check source data |
| Config errors | Compare with `config/example_config.yaml` |

## 📊 Output Validation

Quick checks after processing:
```bash
# List outputs
ls -R data/derivatives/therasync-bvp/

# Check metrics file
cat data/derivatives/therasync-bvp/sub-f01p01/ses-01/physio/*bvpmetrics*.tsv

# View latest log
tail -50 log/bvp_preprocessing_*.log
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
