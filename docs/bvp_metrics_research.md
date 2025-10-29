# NeuroKit2 BVP Metrics Extraction - Research Findings

**Last Updated:** October 28, 2025  
**Status:** ✅ Decision Made - Extended Set (~18 metrics) Implemented  
**Implementation:** `src/physio/preprocessing/bvp_metrics.py`

## Overview

Based on research of NeuroKit2 documentation, here are the available BVP-derived metrics and analysis functions for our TherasyncPipeline project. BVP data from PPG processing can provide rich cardiovascular and autonomic nervous system insights.

**Decision Outcome:** Extended Set (~18 HRV metrics) was selected and implemented. See `docs/bvp_decisions.md` for rationale.

## Available BVP/PPG Analysis Functions

### 1. PPG-Specific Analysis Functions

- **`nk.ppg_analyze(signal, sampling_rate)`**: General PPG signal analysis
- **`nk.ppg_eventrelated(signal, events)`**: Event-related PPG analysis
- **`nk.ppg_intervalrelated(signal)`**: Interval-based PPG analysis

### 2. Heart Rate Variability (HRV) Functions

Once we have peaks from `nk.ppg_process()`, we can use the comprehensive HRV analysis:

#### Main HRV Functions:
- **`nk.hrv(peaks, sampling_rate, show=True)`**: **Complete HRV analysis** (all domains)
- **`nk.hrv_time(peaks, sampling_rate)`**: Time-domain HRV metrics
- **`nk.hrv_frequency(peaks, sampling_rate)`**: Frequency-domain HRV metrics  
- **`nk.hrv_nonlinear(peaks, sampling_rate)`**: Non-linear HRV metrics

#### Specialized HRV Functions:
- **`nk.hrv_rsa(peaks, sampling_rate)`**: Respiratory Sinus Arrhythmia
- **`nk.hrv_rqa(peaks, sampling_rate)`**: Recurrence Quantification Analysis

## Detailed BVP Metrics Categories

### A. **Time-Domain HRV Metrics** (from `nk.hrv_time()`)

**Basic Statistics:**
- `HRV_MeanNN`: Mean of RR intervals (ms)
- `HRV_SDNN`: Standard deviation of RR intervals (ms)
- `HRV_RMSSD`: Root mean square of successive RR interval differences (ms)
- `HRV_SDSD`: Standard deviation of successive RR interval differences (ms)
- `HRV_MinNN`: Minimum RR interval (ms)
- `HRV_MaxNN`: Maximum RR interval (ms)

**Advanced Time-Domain:**
- `HRV_CVNN`: Coefficient of variation of RR intervals
- `HRV_CVSD`: Coefficient of variation of successive differences
- `HRV_MedianNN`: Median RR interval (ms)
- `HRV_MadNN`: Median absolute deviation of RR intervals
- `HRV_MCVNN`: Mean coefficient of variation
- `HRV_IQRNN`: Interquartile range of RR intervals

**Percentage-Based Metrics:**
- `HRV_pNN50`: Percentage of successive RR intervals that differ by >50ms
- `HRV_pNN20`: Percentage of successive RR intervals that differ by >20ms
- `HRV_Prc20NN`: 20th percentile of RR intervals
- `HRV_Prc80NN`: 80th percentile of RR intervals

**Geometric Methods:**
- `HRV_TINN`: Triangular interpolation of NN interval histogram
- `HRV_HTI`: HRV triangular index

**Advanced Measures:**
- `HRV_SDANN`: Standard deviation of 5-minute segment means
- `HRV_SDNNI`: Mean of 5-minute segment standard deviations

### B. **Frequency-Domain HRV Metrics** (from `nk.hrv_frequency()`)

**Power Spectral Density:**
- `HRV_ULF`: Ultra low frequency power (≤0.003 Hz)
- `HRV_VLF`: Very low frequency power (0.003-0.04 Hz)
- `HRV_LF`: Low frequency power (0.04-0.15 Hz)
- `HRV_HF`: High frequency power (0.15-0.4 Hz)
- `HRV_TP`: Total power (≤0.4 Hz)

**Normalized Power:**
- `HRV_LFn`: LF power in normalized units
- `HRV_HFn`: HF power in normalized units
- `HRV_LnHF`: Natural logarithm of HF power

**Frequency Ratios:**
- `HRV_LFHF`: LF/HF ratio (autonomic balance indicator)
- `HRV_LFnu`: LF normalized units
- `HRV_HFnu`: HF normalized units

**Peak Frequencies:**
- `HRV_LFpeak`: Peak frequency in LF band
- `HRV_HFpeak`: Peak frequency in HF band

### C. **Non-Linear HRV Metrics** (from `nk.hrv_nonlinear()`)

**Poincaré Plot Analysis:**
- `HRV_SD1`: Standard deviation perpendicular to line of identity (short-term variability)
- `HRV_SD2`: Standard deviation along line of identity (long-term variability)
- `HRV_SD1SD2`: Ratio SD1/SD2
- `HRV_S`: Area of confidence ellipse
- `HRV_CSI`: Cardiac sympathetic index
- `HRV_CVI`: Cardiac vagal index
- `HRV_CSI_Modified`: Modified cardiac sympathetic index

**Entropy Measures:**
- `HRV_SampEn`: Sample entropy
- `HRV_ShanEn`: Shannon entropy
- `HRV_FuzzyEn`: Fuzzy entropy
- `HRV_MSE`: Multiscale entropy
- `HRV_CMSE`: Composite multiscale entropy
- `HRV_RCMSE`: Refined composite multiscale entropy

**Fractal Analysis:**
- `HRV_DFA_alpha1`: Detrended fluctuation analysis (short-term scaling)
- `HRV_DFA_alpha2`: Detrended fluctuation analysis (long-term scaling)
- `HRV_HFD`: Higuchi fractal dimension
- `HRV_KFD`: Katz fractal dimension
- `HRV_LZC`: Lempel-Ziv complexity

**Recurrence Analysis:**
- `HRV_REC`: Recurrence rate
- `HRV_DET`: Determinism
- `HRV_LAM`: Laminarity
- `HRV_TT`: Trapping time

### D. **PPG-Specific Metrics** (from `nk.ppg_intervalrelated()`)

- **Pulse Rate Variability (PRV)**: Similar to HRV but from pulse peaks
- **Pulse Amplitude**: Peak-to-peak amplitude variations
- **Pulse Transit Time**: If multiple PPG channels available
- **Signal Quality Indices**: Assessment of signal reliability

## Recommended Metrics Selection for Family Therapy Research

### **Essential Metrics (Basic Set)**

**Time-Domain (5 metrics):**
- `HRV_MeanNN`: Average heart rate
- `HRV_SDNN`: Overall HRV
- `HRV_RMSSD`: Parasympathetic activity
- `HRV_pNN50`: Parasympathetic activity
- `HRV_CVNN`: Normalized variability

**Frequency-Domain (4 metrics):**
- `HRV_LF`: Sympathetic + parasympathetic
- `HRV_HF`: Parasympathetic (respiratory sinus arrhythmia)
- `HRV_LFHF`: Autonomic balance
- `HRV_TP`: Total power

**Non-Linear (3 metrics):**
- `HRV_SD1`: Short-term variability
- `HRV_SD2`: Long-term variability  
- `HRV_SampEn`: Signal complexity

### **Extended Metrics (Research Set)**

Add these for more comprehensive analysis:

**Additional Time-Domain:**
- `HRV_TINN`: Geometric method
- `HRV_MinNN`, `HRV_MaxNN`: Range information

**Additional Frequency-Domain:**
- `HRV_VLF`: Very low frequency (thermoregulation, hormonal)
- `HRV_LFpeak`, `HRV_HFpeak`: Dominant frequencies

**Additional Non-Linear:**
- `HRV_DFA_alpha1`: Short-term scaling
- `HRV_CSI`, `HRV_CVI`: Cardiac autonomic indices
- `HRV_ShanEn`: Information entropy

## Implementation Approach

### Configuration for config.yaml

```yaml
physio:
  bvp:
    metrics:
      extract_all: false
      time_domain:
        - "HRV_MeanNN"
        - "HRV_SDNN" 
        - "HRV_RMSSD"
        - "HRV_pNN50"
        - "HRV_CVNN"
      frequency_domain:
        - "HRV_LF"
        - "HRV_HF"
        - "HRV_LFHF"
        - "HRV_TP"
      nonlinear:
        - "HRV_SD1"
        - "HRV_SD2"
        - "HRV_SampEn"
      # Optional: extract all available metrics
      # extract_all: true
```

### Processing Pipeline

1. **Load BVP signal** (via `BVPLoader`)
2. **Clean and process** (via `BVPCleaner` with `nk.ppg_process()`)
3. **Extract peaks** from processing results
4. **Compute HRV metrics** using extracted peaks:
   ```python
   # Get peaks from PPG processing
   peaks = processing_info['PPG_Peaks']
   
   # Compute comprehensive HRV
   hrv_results = nk.hrv(peaks, sampling_rate=64, show=False)
   
   # Or compute specific domains
   hrv_time = nk.hrv_time(peaks, sampling_rate=64)
   hrv_freq = nk.hrv_frequency(peaks, sampling_rate=64) 
   hrv_nonlinear = nk.hrv_nonlinear(peaks, sampling_rate=64)
   ```

## Key Considerations for Family Therapy

### **Physiological Relevance:**
- **HF power**: Parasympathetic activity (relaxation, emotional regulation)
- **LF/HF ratio**: Sympathovagal balance (stress vs. calm states)
- **RMSSD**: Beat-to-beat variability (emotional reactivity)
- **Sample Entropy**: System complexity (adaptive capacity)

### **Moment Comparisons:**
- **Resting State**: Baseline autonomic function
- **Therapy Sessions**: Stress responses and emotional regulation during interactions
- **Change Metrics**: Differences between resting and therapy states

## Next Steps - User Decision Required

**QUESTION**: Which BVP metrics would you like to include in the Therasync pipeline?

1. **Essential Set** (12 metrics): Basic clinical/research metrics
2. **Extended Set** (20+ metrics): Comprehensive research analysis
3. **All Available** (40+ metrics): Complete NeuroKit2 HRV suite
4. **Custom Selection**: Specific metrics based on research questions

**Additional considerations:**
- Should we compute metrics for each moment separately or compare between moments?
- Are there specific autonomic nervous system aspects of interest for family therapy?
- Do you want visualization/plotting capabilities for the metrics?

Please review these options and let me know your preference so I can implement the BVP metrics extraction module.