# Testing BVP Pipeline on Real Data

## 🚀 Quick Start Guide

### Prerequisites
- Poetry installed and dependencies ready
- Real BVP data in BIDS-compliant structure under `data/sourcedata/`
- Configuration file set up (use `config/config.yaml` or `config/example_config.yaml` as template)

### Testing Workflow

#### 1. Test on Single Subject/Session

```bash
# Process a single subject/session
PYTHONPATH=. poetry run python scripts/preprocess_bvp.py \
  --subject sub-f01p01 \
  --session ses-01 \
  --verbose
```

#### 2. Check the Outputs

Outputs will be in `data/derivatives/therasync-bvp/`:
```
data/derivatives/therasync-bvp/
├── dataset_description.json
└── sub-f01p01/
    └── ses-01/
        └── physio/
            ├── sub-f01p01_ses-01_task-*_recording-bvp_physio.tsv
            ├── sub-f01p01_ses-01_task-*_recording-bvp_physio.json
            ├── sub-f01p01_ses-01_desc-bvpmetrics_physio.tsv
            └── sub-f01p01_ses-01_desc-bvpmetrics_physio.json
```

Logs will be in `log/`:
```
log/
├── bvp_preprocessing_YYYYMMDD_HHMMSS.log
└── therasyncpipeline.log
```

#### 3. Clean Outputs for Re-testing

```bash
# Dry run to see what would be deleted
PYTHONPATH=. poetry run python scripts/clean_outputs.py --all --dry-run

# Clean specific subject/session
PYTHONPATH=. poetry run python scripts/clean_outputs.py \
  --derivatives \
  --subject sub-f01p01 \
  --session ses-01

# Clean everything (with confirmation)
PYTHONPATH=. poetry run python scripts/clean_outputs.py --all

# Force clean without confirmation (be careful!)
PYTHONPATH=. poetry run python scripts/clean_outputs.py --all --force
```

#### 4. Iterate and Refine

Repeat steps 1-3 as needed during development and testing.

## 📋 Common Testing Scenarios

### Test with Specific Moments Only
```bash
PYTHONPATH=. poetry run python scripts/preprocess_bvp.py \
  --subject sub-f01p01 \
  --session ses-01 \
  --moments restingstate therapy
```

### Test with Custom Configuration
```bash
PYTHONPATH=. poetry run python scripts/preprocess_bvp.py \
  --subject sub-f01p01 \
  --session ses-01 \
  --config config/example_config.yaml
```

### Batch Process Multiple Subjects
```bash
# Process all subjects matching pattern
PYTHONPATH=. poetry run python scripts/preprocess_bvp.py \
  --batch \
  --subject-pattern "sub-f01*" \
  --continue-on-error
```

### Verbose Output for Debugging
```bash
PYTHONPATH=. poetry run python scripts/preprocess_bvp.py \
  --subject sub-f01p01 \
  --session ses-01 \
  --verbose
```

## 🔍 Troubleshooting

### Check Data Structure
Your data should follow this structure:
```
data/sourcedata/
└── sub-f01p01/
    └── ses-01/
        └── physio/
            ├── sub-f01p01_ses-01_task-restingstate_recording-bvp.tsv
            ├── sub-f01p01_ses-01_task-restingstate_recording-bvp.json
            ├── sub-f01p01_ses-01_task-therapy_recording-bvp.tsv
            └── sub-f01p01_ses-01_task-therapy_recording-bvp.json
```

### Common Issues

**Issue**: `FileNotFoundError: BVP file not found`
- **Solution**: Check that your data files follow BIDS naming conventions
- **Check**: File paths and naming in the error message

**Issue**: `ValueError: Insufficient peaks for HRV analysis`
- **Solution**: Signal might be too short or too noisy
- **Check**: Signal duration (need at least ~30 seconds) and quality

**Issue**: Import errors
- **Solution**: Always use `PYTHONPATH=.` before the command
- **Alternative**: Run from project root directory

**Issue**: Configuration errors
- **Solution**: Verify your `config/config.yaml` has all required fields
- **Check**: Use `config/example_config.yaml` as reference

### View Logs
```bash
# View latest processing log
tail -f log/bvp_preprocessing_*.log

# View all logs
ls -lht log/

# Clean old logs
PYTHONPATH=. poetry run python scripts/clean_outputs.py --logs
```

## ✅ Validation Checklist

After processing, verify:

- [ ] Processed signals TSV files exist for each moment
- [ ] Corresponding JSON metadata files exist
- [ ] Metrics TSV file contains all expected HRV metrics
- [ ] Metrics JSON describes the columns correctly
- [ ] Dataset description JSON is present at pipeline root
- [ ] Log files show successful processing
- [ ] No error messages in logs
- [ ] HRV values are physiologically reasonable (e.g., heart rate 50-100 BPM)

## 🔄 Iterative Testing Workflow

```bash
# 1. Process data
PYTHONPATH=. poetry run python scripts/preprocess_bvp.py -s sub-f01p01 -e ses-01 -v

# 2. Check outputs
ls -R data/derivatives/therasync-bvp/sub-f01p01/ses-01/

# 3. Review logs
tail -20 log/bvp_preprocessing_*.log

# 4. Clean for next iteration
PYTHONPATH=. poetry run python scripts/clean_outputs.py -d -s sub-f01p01 -e ses-01 -f

# 5. Repeat with modifications
```

## 📊 Expected HRV Metrics

Your output should include these 12 essential metrics:

**Time-Domain:**
- HRV_MeanNN (typically 600-1000 ms for adults)
- HRV_SDNN (typically 20-100 ms)
- HRV_RMSSD (typically 20-50 ms)
- HRV_pNN50 (typically 5-30%)
- HRV_CVNN (coefficient of variation)

**Frequency-Domain:**
- HRV_LF (low frequency power)
- HRV_HF (high frequency power)
- HRV_LFHF (LF/HF ratio, typically 0.5-3.0)
- HRV_TP (total power)

**Nonlinear:**
- HRV_SD1 (short-term variability)
- HRV_SD2 (long-term variability)
- HRV_SampEn (sample entropy)

## 🛟 Need Help?

- Check logs in `log/` directory
- Review configuration in `config/config.yaml`
- Verify data structure matches BIDS format
- Use `--verbose` flag for detailed output
- Run `--dry-run` with clean script to preview deletions

Happy testing! 🎉
