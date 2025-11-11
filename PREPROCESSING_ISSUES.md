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
