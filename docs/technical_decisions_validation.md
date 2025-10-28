# Technical Decisions Validation Session

## Purpose

This document consolidates all technical decisions made during Sprint 2 (BVP) and Sprint 3 (EDA) pipeline development to enable formal validation and documentation.

---

## Sprint 2: BVP Pipeline Decisions

### 🔥 DECISION 1: BVP Preprocessing Method

**Research Document:** `docs/bvp_preprocessing_research.md`

**Options Presented:**
1. ✅ **Automatic NeuroKit2 Pipeline** (SELECTED)
   - Uses `nk.ppg_process()` with "elgendi" peak detection
   - Fully automated, optimized for PPG signals
   
2. Manual Pipeline with Standard Methods
   - Step-by-step control over cleaning, detrending, filtering
   
3. Windowed Standardization Pipeline
   - Combines cleaning with windowed standardization

**Current Implementation:**
```python
# In BVPCleaner (src/physio/bvp_cleaner.py)
signals, info = nk.ppg_process(
    ppg_signal, 
    sampling_rate=64, 
    method="elgendi"  # Peak detection method
)
# Quality assessment with template matching
quality = nk.ppg_quality(signals['PPG_Clean'], method="templatematch")
```

**Configuration:**
```yaml
physio:
  bvp:
    processing:
      method: "elgendi"
      quality_method: "templatematch"
```

**Validation Questions:**
- [ ] Confirm Option 1 (Automatic) is appropriate for research-grade analysis?
- [ ] Is "elgendi" peak detection method suitable for family therapy context?
- [ ] Should we expose more preprocessing parameters in config.yaml?

---

### 🔥 DECISION 2: BVP Metrics Selection

**Research Document:** `docs/bvp_metrics_research.md`

**Options Presented:**
1. Essential Set (12 metrics) - Basic clinical/research metrics
2. ✅ **Extended Set (~18 metrics)** (SELECTED - approximated)
3. All Available (40+ metrics) - Complete NeuroKit2 HRV suite
4. Custom Selection

**Current Implementation:** (~18 metrics)

**Time-Domain HRV (5 metrics):**
- `HRV_MeanNN` - Mean RR intervals (ms)
- `HRV_SDNN` - Overall HRV
- `HRV_RMSSD` - Parasympathetic activity
- `HRV_CVNN` - Normalized variability
- `HRV_pNN50` - Parasympathetic activity

**Frequency-Domain HRV (4 metrics):**
- `HRV_LF` - Low frequency power
- `HRV_HF` - High frequency power (respiratory)
- `HRV_TP` - Total power
- `HRV_LFHF` - Autonomic balance ratio

**Non-Linear HRV (3 metrics):**
- `HRV_SD1` - Short-term variability (Poincaré)
- `HRV_SD2` - Long-term variability (Poincaré)
- `HRV_SampEn` - Sample entropy (complexity)

**Quality/Metadata (6 metrics):**
- `BVP_NumPeaks` - Number of detected peaks
- `BVP_Duration` - Recording duration
- `BVP_PeakRate` - Heart rate (BPM)
- `BVP_MeanQuality` - Average signal quality
- `BVP_QualityStd` - Quality variability
- `BVP_MeanAmplitude`, `BVP_StdAmplitude`, `BVP_RangeAmplitude`

**Configuration:**
```yaml
physio:
  bvp:
    metrics:
      extract_all: false
      time_domain:
        - "HRV_MeanNN"
        - "HRV_SDNN"
        - "HRV_RMSSD"
        - "HRV_CVNN"
        - "HRV_pNN50"
      frequency_domain:
        - "HRV_LF"
        - "HRV_HF"
        - "HRV_TP"
        - "HRV_LFHF"
      nonlinear:
        - "HRV_SD1"
        - "HRV_SD2"
        - "HRV_SampEn"
```

**Validation Questions:**
- [ ] Confirm Extended Set (~18 metrics) provides good balance?
- [ ] Missing any critical HRV metrics for family therapy research?
- [ ] Should we add: VLF (very low frequency), DFA (fractal analysis), CSI/CVI (autonomic indices)?
- [ ] Are quality metrics sufficient for data validation?

---

## Sprint 3: EDA Pipeline Decisions

### 🔥 DECISION 3: EDA Preprocessing Method

**Research Document:** `docs/eda_preprocessing_research.md` ✅ CREATED

**Options Presented:**
1. ✅ **Automatic NeuroKit2 Pipeline with cvxEDA** (SELECTED)
   - Uses `nk.eda_process()` with cvxEDA decomposition
   - Gold standard method for tonic-phasic separation
   - Best accuracy, slower computation
   
2. Fast Processing with Sparse Decomposition
   - Good balance of speed and accuracy
   - 5-10x faster than cvxEDA
   
3. Ultra-Fast Pipeline with Smoothmedian
   - Very fast for exploration
   - Lower accuracy

**Current Implementation:**
```python
# In EDACleaner (src/physio/eda_cleaner.py)
signals, info = nk.eda_process(
    eda_signal,
    sampling_rate=4,
    method="neurokit"  # Uses cvxEDA internally
)
# Extracts: EDA_Clean, EDA_Tonic, EDA_Phasic, SCR_Peaks
```

**Configuration:**
```yaml
physio:
  eda:
    sampling_rate: 4  # Empatica E4
    processing:
      method: "neurokit"  # cvxEDA decomposition
      scr_threshold: 0.01  # μS minimum amplitude
      filter: true
```

**Validation Questions:**
- [ ] Confirm cvxEDA (Option 1) is appropriate despite slower computation?
- [ ] Is 0.01 μS SCR threshold suitable for your population?
- [ ] Processing speed acceptable? (1-2 seconds per 45-min session)
- [ ] Should we offer Option 2 (sparse) as alternative for batch processing?

---

### 🔥 DECISION 4: EDA Metrics Selection

**Research Document:** `docs/eda_metrics_research.md` ✅ CREATED

**Options Presented:**
1. Essential Set (12 metrics) - Most interpretable, minimal redundancy
2. ✅ **Extended Set (23 metrics)** (SELECTED - IMPLEMENTED)
3. Comprehensive Set (40+ metrics) - Complete analysis
4. Custom Selection

**Current Implementation:** (23 metrics)

**SCR Metrics (9 metrics):**
- `SCR_Peaks_N` - Number of SCRs
- `SCR_Peaks_Rate` - SCRs per minute
- `SCR_Peaks_Amplitude_Mean` - Average intensity
- `SCR_Peaks_Amplitude_Max` - Maximum intensity
- `SCR_Peaks_Amplitude_SD` - Intensity variability
- `SCR_RiseTime_Mean` - Average rise time
- `SCR_RiseTime_SD` - Rise time variability
- `SCR_RecoveryTime_Mean` - Average recovery time
- `SCR_RecoveryTime_SD` - Recovery time variability

**Tonic Metrics (5 metrics):**
- `EDA_Tonic_Mean` - Baseline arousal level
- `EDA_Tonic_SD` - Arousal stability
- `EDA_Tonic_Min` - Minimum arousal
- `EDA_Tonic_Max` - Maximum arousal
- `EDA_Tonic_Range` - Arousal dynamic range

**Phasic Metrics (6 metrics):**
- `EDA_Phasic_Mean` - Average phasic activity
- `EDA_Phasic_SD` - Phasic variability
- `EDA_Phasic_Min` - Minimum phasic
- `EDA_Phasic_Max` - Maximum phasic
- `EDA_Phasic_Range` - Phasic dynamic range
- `EDA_Phasic_Rate` - Frequency of rapid changes

**Metadata (2 metrics):**
- `EDA_Duration` - Recording length
- `EDA_SamplingRate` - Sampling frequency

**Validation Questions:**
- [ ] Confirm Extended Set (23 metrics) is appropriate?
- [ ] Missing any critical EDA metrics for emotional regulation research?
- [ ] Should we add: Tonic_Slope (arousal trend), SCR_Amplitude_Sum (total reactivity)?
- [ ] Should we reduce to Essential Set (12 metrics) for simpler analysis?

---

## Test Results Summary

### BVP Pipeline (sub-f01p01/ses-01):
- ✅ Restingstate: 100 peaks, 60.0 BPM average
- ✅ Therapy (46 min): Successful processing
- ✅ HRV metrics extracted successfully
- ✅ BIDS-compliant output generated

### EDA Pipeline (sub-f01p01/ses-01):
- ✅ Restingstate (60s): 22 SCRs, 22.0/min rate, 0.567 μS amplitude, 1.810 μS tonic
- ✅ Therapy (46.3 min): 791 SCRs, 17.08/min rate, 0.745 μS amplitude, 1.639 μS tonic
- ✅ All 23 metrics extracted successfully
- ✅ BIDS-compliant output generated (13 files)

**Physiological Reasonability:**
- ✅ SCR rates physiologically plausible (1-30/min range)
- ✅ Tonic levels within normal range (1.6-1.8 μS)
- ✅ SCR amplitudes reasonable (0.5-0.7 μS mean)

---

## Configuration Summary

### Current config.yaml Structure:
```yaml
physio:
  bvp:
    sampling_rate: 64
    processing:
      method: "elgendi"
      quality_method: "templatematch"
    metrics:
      extract_all: false
      # Lists of selected metrics by category
  
  eda:
    sampling_rate: 4
    processing:
      method: "neurokit"
      scr_threshold: 0.01
      filter: true
    metrics:
      extract_all: false
      # Lists of selected metrics by category
```

---

## Validation Checklist

### For BVP:
- [ ] Validate preprocessing method (elgendi automatic)
- [ ] Validate metrics selection (~18 metrics)
- [ ] Confirm no missing critical metrics
- [ ] Document decision rationale

### For EDA:
- [ ] Validate preprocessing method (cvxEDA)
- [ ] Validate SCR threshold (0.01 μS)
- [ ] Validate metrics selection (23 metrics)
- [ ] Confirm no missing critical metrics
- [ ] Document decision rationale

### Documentation Tasks:
- [ ] Create formal decision documents (bvp_decisions.md, eda_decisions.md)
- [ ] Update TODO.md with completed decisions
- [ ] Add decision rationale to config.yaml comments
- [ ] Update sprint summary documents

---

## Next Steps

1. **Review this document** with user
2. **Answer validation questions** for each decision
3. **Adjust implementations** if needed based on feedback
4. **Document final decisions** in separate decision files
5. **Update TODO.md** to mark decisions as completed
6. **Commit validation documentation** to git

---

**Date:** October 28, 2025  
**Sprint:** Sprint 3 (EDA Pipeline)  
**Branch:** sprint-3/eda-preprocessing  
**Status:** Awaiting User Validation
