# Preprocessing Artifacts Investigation

**Date identified**: November 10, 2025  
**Context**: Visualization module development (fix/visualization-empty-plots branch)  
**Subject**: sub-f01p01, ses-01

## Issues Identified

### 1. HRV Frequency Domain - Aberrant LF/HF Ratio

**Observation**:
- LF/HF ratio = **28.78** for therapy moment
- LF/HF ratio = 0.25 for restingstate (normal)

**Data**:
```
restingstate: LF=0.018, HF=0.071 → ratio = 0.25 ✓
therapy:      LF=0.0065, HF=0.00023 → ratio = 28.78 ❌
```

**Problem**: HF power is quasi-zero (0.00023 ms²) during therapy, which is physiologically impossible.

**Normal range**: LF/HF ratio should be 0.5-10 max, typically 1-3.

**Potential causes**:
- FFT window size inappropriate for therapy duration
- Respiratory frequency outside HF band (0.15-0.4 Hz)
- Signal quality issues (BVP_MeanQuality = 0.70 therapy vs 0.81 restingstate)
- Filtering artifacts in HRV preprocessing

**Action items**:
- [ ] Check HRV frequency analysis parameters (window size, overlap)
- [ ] Verify respiratory rate during therapy (should be 9-24 breaths/min)
- [ ] Review BVP signal quality during therapy session
- [ ] Compare with other subjects/sessions to see if systematic

---

### 2. EDA Tonic - Negative Values

**Observation**:
- EDA_Tonic min = **-0.051 µS** for therapy moment
- EDA_Tonic min = 1.321 µS for restingstate (normal)

**Data**:
```
restingstate: Tonic min=1.321, max=2.074, mean=1.810
therapy:      Tonic min=-0.051, max=2.959, mean=1.639
```

**Problem**: Skin conductance cannot be negative (physical impossibility).

**Normal range**: EDA tonic should be 0.5-20 µS, always positive.

**Potential causes**:
- cvxEDA optimization issue (negative baseline artifact)
- Overly aggressive high-pass filtering
- Baseline correction error
- Signal calibration problem

**Action items**:
- [ ] Review cvxEDA parameters (regularization weights)
- [ ] Check if negative values are isolated or span long periods
- [ ] Verify EDA_Raw signal quality before decomposition
- [ ] Compare preprocessing across subjects to identify pattern

---

## Workarounds Applied (Visualization Module)

1. **HRV visualization #3**: Replaced LF/HF ratio with SDNN/RMSSD metrics
   - More robust time-domain metrics
   - Avoids division-by-zero issues
   
2. **EDA visualization #4**: Clipped tonic values to 0
   - `tonic_clipped = df['EDA_Tonic'].clip(lower=0)`
   - Prevents negative display while preserving other stats

---

## Investigation Priority

**HIGH**: These are not isolated edge cases - they indicate systematic preprocessing issues that could affect all analyses.

**Next steps after visualization module completion**:
1. Run quality check script across all subjects/sessions
2. Identify prevalence of issues
3. Review and adjust preprocessing pipeline parameters
4. Reprocess affected data if needed

---

## Investigation Strategy (November 11, 2025)

**Branch**: `analysis/preprocessing-artifacts`  
**Status**: Ready to execute

### Phase 1: Global Exploratory Analysis (Quick Wins)

**Objective**: Get a comprehensive statistical overview of all 50 preprocessed sessions.

**Actions**:
1. Create `scripts/analysis/compute_preprocessing_stats.py`
   - Iterate over all sessions in `data/derivatives/preprocessing/`
   - Extract key metrics per session:
     - **BVP/HRV**: min/max/mean HR, RMSSD, SDNN, LF power, HF power, LF/HF ratio
     - **EDA**: min/max/mean tonic, number of SCR, mean SCR amplitude, phasic peak
     - **HR**: min/max/mean HR, heart rate variability
   - Generate consolidated CSV: `data/derivatives/analysis/preprocessing_stats.csv`
   - Create distribution plots: histograms + boxplots for each metric

2. Identify outliers automatically
   - Sessions with LF/HF > 10 (physiologically aberrant)
   - Sessions with EDA_Tonic min < 0 (physical impossibility)
   - Sessions with HR < 30 or > 200 bpm (measurement errors)

**Expected output**:
- CSV with ~15-20 metrics × 50 sessions
- 6-8 visualization plots (distributions)
- List of flagged sessions for Phase 2

**Estimated time**: 30 min coding + 5 min execution

---

### Phase 2: Targeted Investigation of Outliers

**Objective**: Understand root causes of aberrant values.

**Actions**:
1. For each flagged session:
   - Visual inspection: raw signal vs. preprocessed signal
   - Check moment transitions (onset/offset timestamps)
   - Review preprocessing logs for warnings/errors
   - Compare signal quality metrics (BVP_MeanQuality, EDA_SCR_Quality)

2. Create detailed case studies
   - Document 2-3 worst cases with full diagnostic
   - Identify common patterns across outliers

**Expected output**:
- Detailed report per outlier session
- Hypothesis about algorithmic vs. data quality issues

**Estimated time**: 1-2 hours depending on findings

---

### Phase 3: Validation and Correction

**Objective**: Fix preprocessing pipeline if necessary.

**Actions**:
1. Test parameter adjustments
   - HRV: FFT window size, frequency bands
   - EDA: cvxEDA regularization weights, baseline correction
   - Signal quality thresholds

2. Validate on test subset (5-10 sessions)
   - Reprocess with adjusted parameters
   - Compare old vs. new metrics
   - Ensure no regression on previously valid sessions

3. Batch re-processing if validated
   - Run corrected pipeline on all affected sessions
   - Update `data/derivatives/preprocessing/`
   - Regenerate visualizations if needed

**Expected output**:
- Updated preprocessing configuration
- Re-processed data (if needed)
- Final validation report

**Estimated time**: 2-4 hours depending on corrections needed

---

### Success Criteria

- [ ] All 50 sessions analyzed with comprehensive stats
- [ ] Outliers identified and documented
- [ ] Root causes understood (algorithm vs. data quality)
- [ ] Preprocessing pipeline validated or corrected
- [ ] Decision made: keep current preprocessing OR reprocess with fixes
- [ ] Documentation updated with findings

---

**Start date**: November 11, 2025  
**Target completion**: TBD
