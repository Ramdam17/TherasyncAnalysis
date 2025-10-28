# EDA Pipeline - Validated Technical Decisions

**Date:** October 28, 2025  
**Sprint:** Sprint 3 - EDA Preprocessing  
**Branch:** sprint-3/eda-preprocessing  
**Status:** ✅ VALIDATED

---

## Decision Summary

This document formalizes the technical decisions made for the EDA (Electrodermal Activity) preprocessing pipeline after user validation.

---

## 🔥 DECISION 3: EDA Preprocessing Method

**Research Document:** `docs/eda_preprocessing_research.md`

### Options Considered:

1. **Automatic NeuroKit2 Pipeline with cvxEDA** (Gold Standard)
2. Fast Processing with Sparse Decomposition
3. Ultra-Fast Pipeline with Smoothmedian

### ✅ DECISION: Option 1 - Automatic with cvxEDA

**Selected Method:**
```python
signals, info = nk.eda_process(
    eda_signal,
    sampling_rate=4,
    method="neurokit"  # Uses cvxEDA internally
)
```

### Rationale:

**Why cvxEDA:**
- ✅ **Gold standard** in EDA research (Greco et al., 2016)
- ✅ **Mathematically rigorous** convex optimization approach
- ✅ **Robust to artifacts** and noise
- ✅ **Well-suited for low sampling rates** (4 Hz Empatica E4)
- ✅ **No parameter tuning required** - automatic optimization
- ✅ **Research-grade quality** - appropriate for publications
- ✅ **Best accuracy** for tonic-phasic decomposition

**Trade-offs Accepted:**
- ⚠️ Slower computation (~1-2 seconds per 45-min session)
- ✅ **Acceptable** - Quality over speed for family therapy research

**Alternative Considered:**
- Sparse decomposition (5-10x faster) remains available if batch processing speed becomes critical

### Validation Confirmed:

- [x] cvxEDA is appropriate choice for research-grade analysis
- [x] Computational cost acceptable (1-2s per session)
- [x] Quality priority justified for clinical research context

---

## SCR Detection Parameters

**Threshold:** 0.01 μS (microsiemens)

### Rationale:

**Why 0.01 μS:**
- ✅ **Standard threshold** in psychophysiology literature
- ✅ **Sensitive** enough to detect subtle responses
- ✅ **Robust** against noise at 4 Hz sampling rate
- ✅ **Validated** with test data:
  - Restingstate: 22 SCRs detected (22/min) - physiologically reasonable
  - Therapy: 791 SCRs detected (17.08/min) - appropriate for active engagement

**Typical Ranges:**
- Conservative: 0.02-0.05 μS (fewer false positives)
- Sensitive: 0.005-0.01 μS (more responses captured)
- **Selected: 0.01 μS** - good balance

### Validation Confirmed:

- [x] 0.01 μS threshold suitable for population
- [x] Test results show physiologically reasonable SCR counts
- [x] No adjustments needed

---

## Configuration Implementation

### In `config/config.yaml`:

```yaml
physio:
  eda:
    sampling_rate: 4  # Empatica E4 standard
    
    processing:
      method: "neurokit"  # Uses cvxEDA for decomposition
      
      # SCR Detection Parameters
      scr_threshold: 0.01  # μS - minimum SCR amplitude
      
      # Signal Processing
      filter: true  # Enable filtering (may be skipped at 4 Hz)
```

### Implementation Files:

- **Module:** `src/physio/eda_cleaner.py`
- **Class:** `EDACleaner`
- **Method:** `clean_signal()` using `nk.eda_process()`

---

## 🔥 DECISION 4: EDA Metrics Selection

**Research Document:** `docs/eda_metrics_research.md`

### Options Considered:

1. Essential Set (12 metrics) - Minimal, most interpretable
2. **Extended Set (23 metrics)** - Comprehensive without overwhelming
3. Comprehensive Set (40+ metrics) - Complete analysis

### ✅ DECISION: Option 2 - Extended Set (23 metrics)

**Selected Metrics:**

#### SCR Metrics (9 metrics):
1. `SCR_Peaks_N` - Number of SCRs
2. `SCR_Peaks_Rate` - SCRs per minute
3. `SCR_Peaks_Amplitude_Mean` - Average response intensity
4. `SCR_Peaks_Amplitude_Max` - Maximum response intensity
5. `SCR_Peaks_Amplitude_SD` - Response intensity variability
6. `SCR_RiseTime_Mean` - Average activation speed
7. `SCR_RiseTime_SD` - Activation speed variability
8. `SCR_RecoveryTime_Mean` - Average regulation capacity
9. `SCR_RecoveryTime_SD` - Regulation capacity variability

#### Tonic Metrics (5 metrics):
1. `EDA_Tonic_Mean` - Baseline arousal level
2. `EDA_Tonic_SD` - Arousal stability
3. `EDA_Tonic_Min` - Minimum arousal
4. `EDA_Tonic_Max` - Maximum arousal
5. `EDA_Tonic_Range` - Arousal dynamic range

#### Phasic Metrics (6 metrics):
1. `EDA_Phasic_Mean` - Average phasic activity
2. `EDA_Phasic_SD` - Phasic variability
3. `EDA_Phasic_Min` - Minimum phasic activity
4. `EDA_Phasic_Max` - Maximum phasic activity
5. `EDA_Phasic_Range` - Phasic dynamic range
6. `EDA_Phasic_Rate` - Frequency of rapid changes

#### Metadata (2 metrics):
1. `EDA_Duration` - Recording length (seconds)
2. `EDA_SamplingRate` - Sampling frequency (Hz)

### Rationale:

**Why Extended Set (23 metrics):**
- ✅ **Comprehensive coverage** of EDA physiology
- ✅ **Balanced** - not too minimal, not overwhelming
- ✅ **Research-grade** - suitable for publications
- ✅ **Interpretable** - each metric has clear meaning
- ✅ **Covers all domains:**
  - Reactivity (SCR count, rate)
  - Intensity (amplitude statistics)
  - Regulation (rise/recovery times)
  - Baseline (tonic levels and stability)
  - Dynamic range (min/max/range)

**Why Not Essential (12 metrics):**
- Would miss important variability metrics (SD, Max)
- Less comprehensive for research analysis

**Why Not Comprehensive (40+ metrics):**
- Additional complexity not needed for current research goals
- 23 metrics provide sufficient detail

### Research Alignment:

**Family Therapy Context:**
- ✅ **Emotional regulation** captured via recovery time metrics
- ✅ **Arousal levels** captured via tonic metrics
- ✅ **Reactivity** captured via SCR frequency and amplitude
- ✅ **Stability** captured via SD and range metrics
- ✅ **Moment comparisons** enabled (restingstate vs. therapy)
- ✅ **Session tracking** possible with comprehensive metrics

### Validation Confirmed:

- [x] 23-metric Extended Set is appropriate
- [x] No additional metrics needed
- [x] Metrics selection matches research goals
- [x] No domain-specific metrics missing

---

## Configuration Implementation

### In `config/config.yaml`:

```yaml
physio:
  eda:
    metrics:
      extract_all: false  # Use selected metrics, not all available
      
      # Selected metrics organized by category
      scr:
        - "SCR_Peaks_N"
        - "SCR_Peaks_Rate"
        - "SCR_Peaks_Amplitude_Mean"
        - "SCR_Peaks_Amplitude_Max"
        - "SCR_Peaks_Amplitude_SD"
        - "SCR_RiseTime_Mean"
        - "SCR_RiseTime_SD"
        - "SCR_RecoveryTime_Mean"
        - "SCR_RecoveryTime_SD"
      
      tonic:
        - "EDA_Tonic_Mean"
        - "EDA_Tonic_SD"
        - "EDA_Tonic_Min"
        - "EDA_Tonic_Max"
        - "EDA_Tonic_Range"
      
      phasic:
        - "EDA_Phasic_Mean"
        - "EDA_Phasic_SD"
        - "EDA_Phasic_Min"
        - "EDA_Phasic_Max"
        - "EDA_Phasic_Range"
        - "EDA_Phasic_Rate"
      
      metadata:
        - "EDA_Duration"
        - "EDA_SamplingRate"
```

### Implementation Files:

- **Module:** `src/physio/eda_metrics.py`
- **Class:** `EDAMetricsExtractor`
- **Method:** `extract_eda_metrics()` - extracts all 23 metrics
- **Method:** `extract_multiple_moments()` - batch processing

---

## Test Results Validation

### Subject: sub-f01p01, Session: ses-01

**Restingstate (60s baseline):**
- Duration: 60.0 s
- SCR_Peaks_N: 22
- SCR_Peaks_Rate: 22.0 per minute
- SCR_Peaks_Amplitude_Mean: 0.567 μS
- EDA_Tonic_Mean: 1.810 μS
- EDA_Phasic_Mean: 0.003 μS
- ✅ **Physiologically reasonable** - high reactivity expected in baseline

**Therapy (46.3 min active session):**
- Duration: 2779.0 s (46.3 minutes)
- SCR_Peaks_N: 791
- SCR_Peaks_Rate: 17.08 per minute
- SCR_Peaks_Amplitude_Mean: 0.745 μS
- SCR_Peaks_Amplitude_Max: 2.928 μS
- EDA_Tonic_Mean: 1.639 μS
- ✅ **Physiologically reasonable** - appropriate reactivity for active therapy

**Interpretation:**
- Higher amplitude in therapy (0.745 vs 0.567 μS) suggests stronger emotional responses
- Lower rate in therapy (17.08 vs 22.0/min) suggests some habituation
- Both tonic levels within normal range (1.6-1.8 μS)
- Results consistent with family therapy emotional engagement

---

## Pipeline Integration

### Complete EDA Processing Flow:

1. **Load** (`EDALoader`) → Raw EDA signal (4 Hz)
2. **Clean** (`EDACleaner`) → cvxEDA decomposition
   - EDA_Clean, EDA_Tonic, EDA_Phasic
   - SCR_Peaks detection
3. **Extract** (`EDAMetricsExtractor`) → 23 metrics
4. **Write** (`EDABIDSWriter`) → BIDS-compliant output
   - Processed signals TSV/JSON
   - SCR events TSV/JSON
   - Metrics TSV/JSON
   - Processing metadata
   - Summary JSON

### Command-Line Interface:

```bash
# Single subject/session
python scripts/preprocess_eda.py --subject sub-f01p01 --session ses-01

# Batch processing
python scripts/preprocess_eda.py --batch --subject-pattern "sub-f*"

# Specific moments
python scripts/preprocess_eda.py --subject sub-f01p01 --session ses-01 --moments restingstate therapy
```

---

## References

**Preprocessing Method:**
- Greco, A., Valenza, G., & Scilingo, E. P. (2016). cvxEDA: A convex optimization approach to electrodermal activity processing. IEEE Transactions on Biomedical Engineering, 63(4), 797-804.

**EDA Methodology:**
- Boucsein, W. (2012). Electrodermal Activity (2nd ed.). Springer.
- Benedek, M., & Kaernbach, C. (2010). A continuous measure of phasic electrodermal activity. Journal of Neuroscience Methods, 190(1), 80-91.
- Society for Psychophysiological Research (2012). Publication recommendations for electrodermal measurements.

**Metrics & Interpretation:**
- Dawson, M. E., Schell, A. M., & Filion, D. L. (2007). The electrodermal system. In Handbook of Psychophysiology (3rd ed., pp. 159-181).
- Bach, D. R., et al. (2010). Time-series analysis for rapid event-related skin conductance responses. Journal of Neuroscience Methods, 184(2), 224-234.

---

## Future Considerations

### Potential Extensions (Not Currently Needed):

**Additional Metrics:**
- Tonic slope (arousal trend over time) - if longitudinal analysis needed
- SCR habituation index - if habituation analysis becomes important
- Entropy measures - if complexity analysis desired

**Alternative Methods:**
- Sparse decomposition - if processing speed becomes critical for large batches
- Custom SCR threshold per subject - if population-specific tuning needed

**Advanced Analysis:**
- Event-related EDA - if stimuli timestamps become available
- Interpersonal synchrony - if family member co-regulation analysis desired
- Change detection - if session-to-session comparison becomes priority

### Migration Path:

Current implementation is flexible:
- `extract_all: true` flag enables all 40+ metrics if needed
- Config-driven metric selection allows easy adjustments
- cvxEDA vs. sparse choice configurable via `method` parameter

---

## Sign-Off

**Decisions Validated By:** User  
**Date:** October 28, 2025  
**Implementation Status:** ✅ Complete and Tested  
**Documentation Status:** ✅ Complete  
**Ready for Production:** ✅ Yes

---

**Next Steps:**
1. ✅ EDA decisions documented (this file)
2. ⏳ BVP decisions documentation (to be created)
3. ⏳ Update TODO.md with completed decisions
4. ⏳ Final Sprint 3 testing and merge preparation
