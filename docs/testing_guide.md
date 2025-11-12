# Testing Guide for Therasync Preprocessing Pipelines

**Last Updated:** November 12, 2025  
**Project Version:** v1.2.0 (DPPA Release)  
**Test Status:** 56/56 passing (100%)  
**Production Validation:** 49/51 preprocessing sessions (96%), 2,514 DPPA ICD pairs (100%)

This guide covers automated unit testing, integration testing, and production validation for all three preprocessing pipelines (BVP, EDA, HR) plus visualization and batch processing.

## 📋 Table of Contents
- [Unit Tests](#unit-tests)
- [Testing on Real Data](#testing-on-real-data)
- [Pipeline-Specific Testing](#pipeline-specific-testing)
- [Troubleshooting](#troubleshooting)
- [Validation Checklist](#validation-checklist)

---

## Unit Tests

### Running All Tests

```bash
# Run all 56 unit tests (34 preprocessing + 22 DPPA)
poetry run pytest tests/

# Run with coverage report
poetry run pytest --cov=src tests/

# Run with verbose output
poetry run pytest -v tests/

# Run specific test files
poetry run pytest tests/test_bvp_pipeline.py
poetry run pytest tests/test_eda_pipeline.py
poetry run pytest tests/test_hr_pipeline.py
poetry run pytest tests/test_dppa.py
```

### Test Organization

```
tests/
├── test_bvp_pipeline.py      # BVP preprocessing tests
├── test_eda_pipeline.py      # EDA preprocessing tests
├── test_hr_pipeline.py       # HR preprocessing tests
└── test_config/              # Test configuration files
    ├── test_config_bvp.yaml
    ├── test_config_eda.yaml
    └── test_config_hr.yaml
```

### Current Test Status (November 2025)

✅ **All 56 unit tests passing (100%)** (34 preprocessing + 22 DPPA)  
✅ **Production validation: 49/51 preprocessing sessions (96%)**  
✅ **DPPA validation: 2,514 ICD pairs computed (100%)**  
✅ **Visualization: 306/306 plots generated (100%)**  
✅ **Quality analysis: 114 flags tracked across all modalities**

**BVP Pipeline Tests (16 tests):**
- Loader initialization and data loading
- Signal cleaning and peak detection (Elgendi method)
- Metrics extraction (20 HRV metrics)
- BIDS output writing (9 files per session)
- Complete end-to-end pipeline integration

**EDA Pipeline Tests (18 tests):**
- Loader initialization and data loading
- Signal decomposition (cvxEDA tonic/phasic)
- SCR detection and analysis
- Metrics extraction (23 EDA metrics)
- BIDS output writing (13 files per session)
- Complete end-to-end pipeline integration

**HR Pipeline Tests:**
- Loader initialization and data loading
- Signal cleaning and outlier detection
- Metrics extraction (26 HR metrics)
- BIDS output writing (14 files per session: 7 per moment)
- Complete end-to-end pipeline integration

**DPPA Pipeline Tests (22 tests):**
- PoincareCalculator: Centroid computation with RRₙ/RRₙ₊₁ pairing (5 tests)
- CentroidLoader: File loading with LRU caching (4 tests)
- DyadConfigLoader: Inter-session and intra-family pair generation (4 tests)
- ICDCalculator: Euclidean distance with NaN propagation (5 tests)
- DPPAWriter: Rectangular CSV export (3 tests)
- End-to-end integration test (1 test)

**Phase 2 Validation:**
- All BIDS writers harmonized with identical code patterns
- Visualization pipeline integrated with per-moment HR data
- All outputs BIDS-compliant and validated

---

## Testing on Real Data

## Testing on Real Data

### Prerequisites
- Poetry installed with all dependencies
- Real physiological data in BIDS-compliant structure under `data/sourcedata/`
- Configuration file set up (`config/config.yaml`)

### Data Structure

Your data should follow this structure:
```
data/sourcedata/
└── sub-{subject}/
    └── ses-{session}/
        └── physio/
            ├── sub-{subject}_ses-{session}_task-restingstate_recording-bvp.tsv
            ├── sub-{subject}_ses-{session}_task-restingstate_recording-bvp.json
            ├── sub-{subject}_ses-{session}_task-restingstate_recording-eda.tsv
            ├── sub-{subject}_ses-{session}_task-restingstate_recording-eda.json
            ├── sub-{subject}_ses-{session}_task-restingstate_recording-hr.tsv
            ├── sub-{subject}_ses-{session}_task-restingstate_recording-hr.json
            ├── sub-{subject}_ses-{session}_task-therapy_recording-bvp.tsv
            ├── sub-{subject}_ses-{session}_task-therapy_recording-eda.tsv
            └── sub-{subject}_ses-{session}_task-therapy_recording-hr.tsv
```

### Quick Start Workflow

#### 1. Test Single Subject/Session

```bash
# BVP preprocessing
poetry run python scripts/physio/preprocessing/preprocess_bvp.py \
  --subject f01p01 \
  --session 01 \
  --verbose

# EDA preprocessing
poetry run python scripts/physio/preprocessing/preprocess_eda.py \
  --subject f01p01 \
  --session 01 \
  --verbose

# HR preprocessing
poetry run python scripts/physio/preprocessing/preprocess_hr.py \
  --subject f01p01 \
  --session 01 \
  --verbose
```

**Note:** Subject and session IDs are provided WITHOUT prefixes (e.g., `f01p01` not `sub-f01p01`, `01` not `ses-01`)

#### 2. Check the Outputs

Outputs are organized by modality in the new structure:
```
data/derivatives/preprocessing/
└── sub-f01p01/
    └── ses-01/
        ├── bvp/                                           # 9 files
        │   ├── sub-f01p01_ses-01_task-restingstate_desc-bvp-processed_physio.tsv.gz
        │   ├── sub-f01p01_ses-01_task-restingstate_desc-bvp-processed_physio.json
        │   ├── sub-f01p01_ses-01_task-restingstate_desc-bvp-metrics_physio.tsv
        │   ├── sub-f01p01_ses-01_task-therapy_desc-bvp-processed_physio.tsv.gz
        │   └── ...
        ├── eda/                                           # 13 files
        │   ├── sub-f01p01_ses-01_task-restingstate_desc-eda-processed_physio.tsv.gz
        │   ├── sub-f01p01_ses-01_task-restingstate_desc-scr_events.tsv
        │   ├── sub-f01p01_ses-01_task-restingstate_desc-eda-metrics_physio.tsv
        │   └── ...
        └── hr/                                            # 7 files
            ├── sub-f01p01_ses-01_task-restingstate_desc-hr-processed_physio.tsv.gz
            ├── sub-f01p01_ses-01_task-restingstate_desc-hr-metrics_physio.tsv
            └── ...
```

Logs will be in `log/`:
```
log/
├── preprocessing_bvp_YYYYMMDD_HHMMSS.log
├── preprocessing_eda_YYYYMMDD_HHMMSS.log
├── preprocessing_hr_YYYYMMDD_HHMMSS.log
└── therasyncpipeline.log
```

#### 3. Clean Outputs for Re-testing

```bash
# Clean specific subject/session outputs
poetry run python scripts/utils/clean_outputs.py \
  --subject f01p01 \
  --session 01

# Clean specific modality
poetry run python scripts/utils/clean_outputs.py \
  --subject f01p01 \
  --session 01 \
  --modality bvp

# Clean all derivatives (with confirmation)
poetry run python scripts/utils/clean_outputs.py --all

# View what would be deleted (dry run)
poetry run python scripts/utils/clean_outputs.py --all --dry-run
```

---

## Pipeline-Specific Testing

### BVP Pipeline

**Expected Outputs:** 9 files per session
- Processed signals (compressed TSV + JSON) for each moment
- Metrics file (TSV + JSON) with 18 HRV metrics
- Summary JSON

**Key Metrics to Validate:**
- HRV_MeanNN (typically 600-1000 ms for adults)
- HRV_SDNN (typically 20-100 ms)
- HRV_RMSSD (typically 20-50 ms)
- HRV_LFHF (LF/HF ratio, typically 0.5-3.0)

**Test Command:**
```bash
poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject f01p01 --session 01 --verbose
```

### EDA Pipeline

**Expected Outputs:** 13 files per session
- Processed signals (compressed TSV + JSON) for each moment
- SCR events (TSV + JSON) for each moment
- Metrics file (TSV + JSON) with 23 EDA metrics
- Summary JSON

**Key Metrics to Validate:**
- SCR rate (typically 1-30 per minute depending on arousal)
- Tonic EDA level (typically 0.001-0.5 μS)
- SCR amplitude (typically 0.01-1.0 μS)

**Test Command:**
```bash
poetry run python scripts/physio/preprocessing/preprocess_eda.py --subject f01p01 --session 01 --verbose
```

### HR Pipeline

**Expected Outputs:** 7 files per session
- Processed signals (compressed TSV + JSON) for each moment
- Metrics file (TSV + JSON) with basic HR metrics
- Summary JSON

**Key Metrics to Validate:**
- Mean HR (typically 60-100 BPM for adults)
- HR variability measures
- HR trends over time

**Test Command:**
```bash
poetry run python scripts/physio/preprocessing/preprocess_hr.py --subject f01p01 --session 01 --verbose
```

### DPPA Pipeline

**Expected Outputs:** 
- Centroids: 606 TSV files (2 methods × 2 tasks × 51 sessions)
- ICDs: 4 CSV files (2 modes × 2 tasks)

**Test Commands:**

```bash
# Test centroid computation (single participant)
poetry run python scripts/physio/dppa/compute_poincare.py -s f01p01 -e 01 -v

# Test ICD calculation (single family)
poetry run python scripts/physio/dppa/compute_dppa.py --mode intra --task therapy --batch

# Run all DPPA unit tests
poetry run pytest tests/test_dppa.py -v
```

**Expected Test Output:**
- ✅ 22/22 tests passing (100%)
- 5 test classes covering all modules:
  - TestPoincareCalculator (5 tests)
  - TestCentroidLoader (4 tests)
  - TestDyadConfigLoader (4 tests)
  - TestICDCalculator (5 tests)
  - TestDPPAWriter (3 tests)
  - TestDPPAIntegration (1 test)

**Key Validations:**
- Centroid computation: RRₙ vs RRₙ₊₁ pairing correct
- ICD formula: Euclidean distance √[(x₁-x₂)²+(y₁-y₂)²]
- NaN handling: Propagates correctly through pipeline
- Output format: Rectangular CSV (epochs × dyads)
- Batch success rate: 100% (2,514 ICD pairs computed)

**Production Validation:**
- Inter-session: 1,176 dyad pairs × 2 tasks = 2,352 ICDs
- Intra-family: 81 dyad pairs × 2 tasks = 162 ICDs
- Total: 2,514 ICD computations (100% success)

---

## Troubleshooting

## Troubleshooting

### Common Issues

**Issue:** `FileNotFoundError: File not found`
- **Solution:** Check that your data files follow BIDS naming conventions
- **Verify:** Files are in `data/sourcedata/sub-{subject}/ses-{session}/physio/`
- **Check:** Subject/session IDs don't include prefixes in command arguments

**Issue:** `ValueError: Insufficient peaks for HRV analysis` (BVP)
- **Solution:** Signal might be too short or too noisy
- **Check:** Signal duration (need at least ~30 seconds) and quality
- **Try:** Adjust peak detection threshold in config.yaml

**Issue:** `cvxEDA convergence warning` (EDA)
- **Solution:** Usually not critical; decomposition still works
- **Check:** If SCR counts are reasonable (1-30 per minute)
- **Reference:** See docs/troubleshooting.md for detailed guidance

**Issue:** `No SCRs detected` (EDA)
- **Solution:** Participant might have low arousal or signal quality issues
- **Check:** Verify raw EDA signal is reasonable (not all zeros)
- **Try:** Adjust SCR threshold in config.yaml (default: 0.01 μS)

**Issue:** Import errors or module not found
- **Solution:** Ensure you're using Poetry environment
- **Run:** `poetry shell` to activate environment
- **No need for PYTHONPATH:** Scripts handle path setup internally

**Issue:** Configuration errors
- **Solution:** Verify your `config/config.yaml` has all required fields
- **Check:** Compare with template sections in README.md
- **Validate:** Run unit tests to catch config issues

### View Logs

```bash
# View latest processing log
tail -f log/preprocessing_*_*.log

# View all logs
ls -lht log/

# Search for errors in logs
grep -i error log/preprocessing_*.log
```

---

## Validation Checklist

## Validation Checklist

### After Processing BVP Data:
- [ ] 9 files created in `data/derivatives/preprocessing/sub-{subject}/ses-{session}/bvp/`
- [ ] Processed signals (TSV.GZ + JSON) exist for each moment
- [ ] Metrics TSV file contains all 18 HRV metrics
- [ ] JSON metadata files describe columns correctly
- [ ] Log files show successful processing
- [ ] HRV values are physiologically reasonable (e.g., heart rate 50-100 BPM)

### After Processing EDA Data:
- [ ] 13 files created in `data/derivatives/preprocessing/sub-{subject}/ses-{session}/eda/`
- [ ] Processed signals (TSV.GZ + JSON) exist for each moment
- [ ] SCR events (TSV + JSON) exist for each moment
- [ ] Metrics TSV file contains all 23 EDA metrics
- [ ] SCR counts are reasonable (typically 1-30 per minute)
- [ ] Tonic EDA levels are in normal range (0.001-0.5 μS)

### After Processing HR Data:
- [ ] 7 files created in `data/derivatives/preprocessing/sub-{subject}/ses-{session}/hr/`
- [ ] Processed signals (TSV.GZ + JSON) exist for each moment
- [ ] Metrics TSV file contains basic HR metrics
- [ ] Mean HR is physiologically reasonable (60-100 BPM)
- [ ] No error messages in logs

### General Validation:
- [ ] All output files follow BIDS naming conventions
- [ ] No error messages in log files
- [ ] File sizes are reasonable (not 0 bytes)
- [ ] JSON files are valid JSON format
- [ ] TSV files can be opened and inspected

---

## Iterative Testing Workflow

```bash
# Example workflow for testing changes

# 1. Process all three modalities for one subject
poetry run python scripts/physio/preprocessing/preprocess_bvp.py -s f01p01 -e 01 -v
poetry run python scripts/physio/preprocessing/preprocess_eda.py -s f01p01 -e 01 -v
poetry run python scripts/physio/preprocessing/preprocess_hr.py -s f01p01 -e 01 -v

# 2. Check outputs
ls -lh data/derivatives/preprocessing/sub-f01p01/ses-01/*/

# 3. Review logs for any issues
tail -50 log/preprocessing_*.log | grep -i "error\|warning"

# 4. Clean for next iteration (if needed)
poetry run python scripts/utils/clean_outputs.py -s f01p01 -e 01

# 5. Run unit tests to verify nothing broke
poetry run pytest tests/ -v

# 6. Repeat with modifications
```

---

## End-to-End Validation (October 2025)

All three pipelines have been validated on real data:

**BVP Pipeline:**
- ✅ Tested on multiple subjects and sessions
- ✅ Produces 9 BIDS-compliant files per session
- ✅ All 18 HRV metrics calculated correctly
- ✅ Physiologically reasonable values confirmed

**EDA Pipeline:**
- ✅ Tested on 5+ subjects (families f01, f02)
- ✅ Produces 13 BIDS-compliant files per session
- ✅ All 23 EDA metrics calculated correctly
- ✅ SCR rates validated across different arousal levels
- ✅ cvxEDA decomposition working correctly

**HR Pipeline:**
- ✅ Tested on multiple subjects and sessions
- ✅ Produces 7 BIDS-compliant files per session
- ✅ Basic HR metrics calculated correctly
- ✅ Physiologically reasonable values confirmed

---

## Need Help?

## Need Help?

**Documentation:**
- Check `README.md` for complete usage guide
- Review `QUICKREF.md` for quick command reference
- See `docs/troubleshooting.md` for detailed error solutions
- Read `docs/api_reference.md` for module documentation

**Logs and Debugging:**
- Check logs in `log/` directory for detailed error messages
- Use `--verbose` flag for detailed output during processing
- Run unit tests to verify core functionality: `poetry run pytest tests/ -v`

**Configuration:**
- Review `config/config.yaml` for current settings
- Adjust parameters if needed (thresholds, methods, etc.)
- Refer to README for configuration options

**Project Status:**
- Current version: v0.3.0 (Modular Architecture)
- Test status: 34/34 passing (100%)
- All three pipelines production-ready and validated

Happy testing! 🎉
