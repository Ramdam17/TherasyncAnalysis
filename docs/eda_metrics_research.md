# NeuroKit2 EDA Metrics Extraction - Research Findings

**Last Updated:** October 28, 2025  
**Status:** ✅ Decision Made - Extended Set (23 metrics) Implemented  
**Implementation:** `src/physio/preprocessing/eda_metrics.py`

## Overview

Based on research of NeuroKit2 documentation, here are the available EDA-derived metrics and analysis functions for our TherasyncPipeline project. EDA data provides rich insights into autonomic nervous system activity and emotional/cognitive processing.

**Decision Outcome:** Extended Set (23 EDA metrics: 9 SCR + 5 Tonic + 6 Phasic + 3 Metadata) was selected and implemented. See `docs/eda_decisions.md` for rationale.

## Available EDA Analysis Functions

### 1. EDA-Specific Analysis Functions

- **`nk.eda_analyze(signal, sampling_rate)`**: General EDA signal analysis
- **`nk.eda_eventrelated(signal, events)`**: Event-related EDA analysis
- **`nk.eda_intervalrelated(signal)`**: Interval-based EDA analysis

### 2. Component Analysis Functions

- **`nk.eda_phasic(signal, sampling_rate)`**: Decompose into tonic/phasic
- **`nk.eda_peaks(signal, sampling_rate)`**: Detect SCR peaks
- **`nk.eda_fixpeaks(peaks)`**: Fix/validate detected peaks

## Detailed EDA Metrics Categories

### A. **SCR (Skin Conductance Response) Metrics**

SCRs are discrete phasic responses reflecting specific emotional/cognitive events.

#### Basic SCR Metrics:

**1. SCR Count/Frequency:**
- `SCR_Peaks_N`: Total number of SCRs detected
- `SCR_Peaks_Rate`: SCRs per minute (peaks/duration*60)
- Interpretation: More SCRs = more reactive, more emotional processing
- Typical range: 1-30 SCRs/minute (varies by context)

**2. SCR Amplitude Metrics:**
- `SCR_Peaks_Amplitude_Mean`: Average SCR amplitude (μS)
- `SCR_Peaks_Amplitude_Max`: Maximum SCR amplitude (μS)
- `SCR_Peaks_Amplitude_Min`: Minimum SCR amplitude (μS)
- `SCR_Peaks_Amplitude_SD`: Standard deviation of amplitudes
- `SCR_Peaks_Amplitude_Sum`: Total amplitude across all SCRs
- Interpretation: Amplitude reflects intensity of response
- Typical range: 0.01-2 μS (higher = stronger response)

**3. SCR Temporal Metrics:**
- `SCR_RiseTime_Mean`: Average rise time (seconds)
- `SCR_RiseTime_SD`: Standard deviation of rise times
- `SCR_RiseTime_Max`: Maximum rise time
- `SCR_RecoveryTime_Mean`: Average recovery time (seconds)
- `SCR_RecoveryTime_SD`: Standard deviation of recovery times
- `SCR_RecoveryTime_Max`: Maximum recovery time
- Interpretation: 
  - Rise time: Speed of autonomic activation (typical 1-3s)
  - Recovery time: Autonomic regulation capacity (typical 2-10s)

**4. SCR Derived Metrics:**
- `SCR_Amplitude_Variability`: Variability in response strength
- `SCR_Latency`: Time from stimulus to SCR (if event-related)
- `SCR_Habituation`: Decrease in amplitude over time
- `SCR_Magnitude`: SCR amplitude relative to baseline

#### Advanced SCR Metrics:

**5. SCR Distribution:**
- `SCR_Peaks_Above_Threshold`: Count of significant SCRs (e.g., >0.05 μS)
- `SCR_Peaks_Distribution`: Histogram of amplitudes
- `SCR_Peaks_Skewness`: Asymmetry of amplitude distribution
- `SCR_Peaks_Kurtosis`: "Tailedness" of amplitude distribution

**6. SCR Dynamics:**
- `SCR_Acceleration`: Rate of SCR amplitude increase
- `SCR_Area`: Area under the SCR curve (AUC)
- `SCR_Half_Recovery_Time`: Time to 50% recovery

### B. **Tonic (SCL - Skin Conductance Level) Metrics**

Tonic component reflects baseline arousal and general emotional state.

#### Basic Tonic Metrics:

**1. Central Tendency:**
- `EDA_Tonic_Mean`: Average tonic level (μS)
- `EDA_Tonic_Median`: Median tonic level (μS)
- Interpretation: Overall arousal/anxiety level
- Typical range: 0.5-5 μS (population-dependent)

**2. Variability:**
- `EDA_Tonic_SD`: Standard deviation of tonic level
- `EDA_Tonic_Variance`: Variance of tonic level
- `EDA_Tonic_CV`: Coefficient of variation (SD/Mean)
- Interpretation: Stability of baseline arousal

**3. Range:**
- `EDA_Tonic_Min`: Minimum tonic level (μS)
- `EDA_Tonic_Max`: Maximum tonic level (μS)
- `EDA_Tonic_Range`: Max - Min (μS)
- `EDA_Tonic_IQR`: Interquartile range
- Interpretation: Dynamic range of arousal changes

**4. Trend:**
- `EDA_Tonic_Slope`: Linear trend over time
- `EDA_Tonic_Trend`: Direction of change (increasing/decreasing)
- Interpretation: 
  - Increasing: Escalating arousal/stress
  - Decreasing: Habituation/relaxation

#### Advanced Tonic Metrics:

**5. Tonic Dynamics:**
- `EDA_Tonic_RateOfChange`: Speed of tonic changes
- `EDA_Tonic_Acceleration`: Second derivative of tonic
- `EDA_Tonic_Fluctuations`: Number of direction changes

**6. Tonic Complexity:**
- `EDA_Tonic_SampleEntropy`: Regularity/predictability
- `EDA_Tonic_ApproximateEntropy`: Signal complexity
- `EDA_Tonic_LempelZiv`: Information content

### C. **Phasic (SCR Activity) Metrics**

Phasic component captures all rapid changes, not just discrete peaks.

#### Basic Phasic Metrics:

**1. Central Tendency:**
- `EDA_Phasic_Mean`: Average phasic activity (μS)
- `EDA_Phasic_Median`: Median phasic activity (μS)
- Interpretation: Overall level of rapid responding

**2. Variability:**
- `EDA_Phasic_SD`: Standard deviation of phasic activity
- `EDA_Phasic_Variance`: Variance of phasic activity
- `EDA_Phasic_CV`: Coefficient of variation
- Interpretation: Dynamic nature of responses

**3. Range:**
- `EDA_Phasic_Min`: Minimum phasic value (μS)
- `EDA_Phasic_Max`: Maximum phasic value (μS)
- `EDA_Phasic_Range`: Max - Min (μS)
- Interpretation: Extent of phasic reactivity

**4. Activity Level:**
- `EDA_Phasic_Rate`: Frequency of phasic changes per minute
- `EDA_Phasic_Duration`: Total time with active phasic component
- `EDA_Phasic_Percentage`: % of time with phasic activity >threshold
- Interpretation: Proportion of time actively responding

#### Advanced Phasic Metrics:

**5. Phasic Dynamics:**
- `EDA_Phasic_RMS`: Root mean square (overall energy)
- `EDA_Phasic_Energy`: Total phasic energy
- `EDA_Phasic_Power`: Average power

**6. Phasic Patterns:**
- `EDA_Phasic_BurstCount`: Number of phasic "bursts"
- `EDA_Phasic_BurstDuration`: Average burst duration
- `EDA_Phasic_InterbustInterval`: Time between bursts

### D. **Global EDA Metrics**

Overall signal characteristics combining tonic and phasic components.

**1. Signal Quality:**
- `EDA_Quality_Score`: Overall signal quality (0-1)
- `EDA_Artifacts_N`: Number of detected artifacts
- `EDA_Valid_Percentage`: % of valid signal

**2. Recording Information:**
- `EDA_Duration`: Total recording duration (seconds)
- `EDA_SamplingRate`: Sampling frequency (Hz)
- `EDA_NumSamples`: Total number of samples

**3. Tonic-Phasic Ratio:**
- `EDA_TonicPhasic_Ratio`: Tonic/Phasic proportion
- `EDA_TonicPhasic_Balance`: Balance between components
- Interpretation: Relative contribution of baseline vs. responses

## Recommended Metrics Selection for Family Therapy Research

### **Essential Metrics (Basic Set)** - 12 metrics ⭐ RECOMMENDED

This set captures the most important EDA features with minimal redundancy.

**SCR Metrics (5 metrics):**
- `SCR_Peaks_N`: Number of SCRs (reactivity count)
- `SCR_Peaks_Rate`: SCRs per minute (reactivity rate)
- `SCR_Peaks_Amplitude_Mean`: Average response intensity
- `SCR_RiseTime_Mean`: Speed of activation
- `SCR_RecoveryTime_Mean`: Regulation capacity

**Tonic Metrics (4 metrics):**
- `EDA_Tonic_Mean`: Baseline arousal level
- `EDA_Tonic_SD`: Arousal stability
- `EDA_Tonic_Range`: Dynamic arousal range
- `EDA_Tonic_Slope`: Arousal trend over time

**Phasic Metrics (2 metrics):**
- `EDA_Phasic_Mean`: Overall phasic activity
- `EDA_Phasic_Rate`: Frequency of rapid changes

**Metadata (1 metric):**
- `EDA_Duration`: Recording length

### **Extended Metrics (Research Set)** - 23 metrics

Current implementation includes these comprehensive metrics.

**SCR Metrics (9 metrics):**
- `SCR_Peaks_N`
- `SCR_Peaks_Rate`
- `SCR_Peaks_Amplitude_Mean`
- `SCR_Peaks_Amplitude_Max`
- `SCR_Peaks_Amplitude_SD`
- `SCR_RiseTime_Mean`
- `SCR_RiseTime_SD`
- `SCR_RecoveryTime_Mean`
- `SCR_RecoveryTime_SD`

**Tonic Metrics (5 metrics):**
- `EDA_Tonic_Mean`
- `EDA_Tonic_SD`
- `EDA_Tonic_Min`
- `EDA_Tonic_Max`
- `EDA_Tonic_Range`

**Phasic Metrics (6 metrics):**
- `EDA_Phasic_Mean`
- `EDA_Phasic_SD`
- `EDA_Phasic_Min`
- `EDA_Phasic_Max`
- `EDA_Phasic_Range`
- `EDA_Phasic_Rate`

**Metadata (2 metrics):**
- `EDA_Duration`
- `EDA_SamplingRate`

**Rationale:** Good balance between comprehensiveness and interpretability.

### **Comprehensive Metrics (Complete Set)** - 40+ metrics

Add all advanced metrics for exploratory research.

**Additional SCR Metrics:**
- Amplitude statistics (min, sum, variability)
- Distribution metrics (skewness, kurtosis)
- Temporal dynamics (acceleration, AUC, half-recovery)
- Threshold-based counts

**Additional Tonic Metrics:**
- Higher-order statistics (CV, IQR, percentiles)
- Trend analysis (rate of change, acceleration)
- Complexity measures (entropy, Lempel-Ziv)

**Additional Phasic Metrics:**
- Energy measures (RMS, power)
- Pattern analysis (bursts, interburst intervals)
- Activity percentages

**Additional Global Metrics:**
- Signal quality indices
- Tonic-phasic ratios
- Artifact detection

## Implementation Approach

### Configuration for config.yaml

```yaml
physio:
  eda:
    metrics:
      extract_all: false
      
      # Essential Set (12 metrics)
      scr:
        - "SCR_Peaks_N"
        - "SCR_Peaks_Rate"
        - "SCR_Peaks_Amplitude_Mean"
        - "SCR_RiseTime_Mean"
        - "SCR_RecoveryTime_Mean"
      
      tonic:
        - "EDA_Tonic_Mean"
        - "EDA_Tonic_SD"
        - "EDA_Tonic_Range"
        - "EDA_Tonic_Slope"
      
      phasic:
        - "EDA_Phasic_Mean"
        - "EDA_Phasic_Rate"
      
      metadata:
        - "EDA_Duration"
      
      # Extended Set (add these for full 23 metrics)
      scr_extended:
        - "SCR_Peaks_Amplitude_Max"
        - "SCR_Peaks_Amplitude_SD"
        - "SCR_RiseTime_SD"
        - "SCR_RecoveryTime_SD"
      
      tonic_extended:
        - "EDA_Tonic_Min"
        - "EDA_Tonic_Max"
      
      phasic_extended:
        - "EDA_Phasic_SD"
        - "EDA_Phasic_Min"
        - "EDA_Phasic_Max"
        - "EDA_Phasic_Range"
      
      metadata_extended:
        - "EDA_SamplingRate"
      
      # Optional: extract all available metrics
      # extract_all: true
```

### Processing Pipeline

1. **Load EDA signal** (via `EDALoader`)
2. **Clean and decompose** (via `EDACleaner` with `nk.eda_process()`)
3. **Extract tonic/phasic components** from processing results
4. **Detect SCR peaks** from phasic component
5. **Compute EDA metrics**:
   ```python
   # Get components from EDA processing
   tonic = processed_signals['EDA_Tonic']
   phasic = processed_signals['EDA_Phasic']
   scr_peaks = processed_signals['SCR_Peaks']
   
   # Compute SCR metrics
   scr_onsets = np.where(scr_peaks == 1)[0]
   scr_amplitudes = processed_signals.loc[scr_onsets, 'SCR_Amplitude']
   scr_metrics = {
       'SCR_Peaks_N': len(scr_onsets),
       'SCR_Peaks_Rate': len(scr_onsets) / duration * 60,
       'SCR_Peaks_Amplitude_Mean': scr_amplitudes.mean(),
       # ... etc
   }
   
   # Compute tonic metrics
   tonic_metrics = {
       'EDA_Tonic_Mean': tonic.mean(),
       'EDA_Tonic_SD': tonic.std(),
       # ... etc
   }
   
   # Compute phasic metrics
   phasic_metrics = {
       'EDA_Phasic_Mean': phasic.mean(),
       'EDA_Phasic_SD': phasic.std(),
       # ... etc
   }
   ```

## Physiological Relevance for Family Therapy

### **Emotional Regulation:**
- **SCR amplitude**: Emotional intensity
- **SCR recovery time**: Regulation capacity (faster = better regulation)
- **Tonic level**: Chronic stress/anxiety

### **Reactivity and Engagement:**
- **SCR frequency**: Emotional/cognitive processing load
- **Phasic activity**: Moment-to-moment engagement
- **Tonic stability**: Emotional stability

### **Interpersonal Synchrony (Future):**
- **Cross-correlation of SCRs**: Emotional co-regulation
- **Tonic similarity**: Shared arousal states
- **SCR timing**: Response to partner's behavior

### **Moment Comparisons:**
- **Resting State**: Baseline autonomic function
  - Lower SCR rate expected
  - Stable tonic level
  
- **Therapy Sessions**: Active emotional processing
  - Higher SCR rate during difficult topics
  - Tonic changes reflecting stress/relaxation
  - SCR patterns showing responses to interventions

### **Therapeutic Outcomes:**
- **Decreased tonic level**: Reduced anxiety over sessions
- **Improved SCR recovery**: Better emotion regulation
- **Reduced SCR rate**: Habituation/decreased reactivity
- **Increased tonic stability**: Improved emotional stability

## Metric Interpretation Guidelines

### Normal Ranges (Context-Dependent):

**Resting State:**
- Tonic: 1-5 μS (population mean ~2-3 μS)
- SCR rate: 1-5 per minute
- SCR amplitude: 0.05-0.5 μS

**Therapy/Active Engagement:**
- Tonic: 2-8 μS (elevated)
- SCR rate: 5-30 per minute
- SCR amplitude: 0.1-2 μS

**High Stress/Anxiety:**
- Tonic: >5 μS
- SCR rate: >20 per minute
- SCR amplitude: >1 μS

### Population Factors:

**Age:**
- Children: Higher reactivity, faster recovery
- Elderly: Lower overall levels, slower recovery

**Clinical Populations:**
- Anxiety: Higher tonic, more SCRs
- Depression: Lower reactivity
- PTSD: Hyperarousal, slow recovery

**Environmental Factors:**
- Temperature: Affects tonic level
- Humidity: Affects signal quality
- Time of day: Circadian variations

## Next Steps - User Decision Required

**QUESTION 1**: Which EDA metrics set would you prefer for the Therasync project?

1. **Essential Set (12 metrics)** - Most interpretable, minimal redundancy
   - Captures key EDA features
   - Easy to analyze and report
   
2. **Extended Set (23 metrics)** ⭐ CURRENTLY IMPLEMENTED
   - Comprehensive without overwhelming
   - Good for research publications
   
3. **Comprehensive Set (40+ metrics)** - Complete analysis
   - Exploratory research
   - More complex analysis required
   
4. **Custom Selection** - Specific metrics for your research questions

**QUESTION 2**: Physiological Focus

- Are you primarily interested in **arousal** (tonic) or **reactivity** (SCRs)?
- Both equally important?
- Specific aspects (e.g., emotional regulation = recovery time)?

**QUESTION 3**: Comparison Strategy

- Compare **moments** (resting vs. therapy)?
- Compare **sessions** (change over time)?
- Compare **family members** (interpersonal patterns)?
- All of the above?

**QUESTION 4**: Analysis Complexity

- Prefer **simple descriptive metrics** (means, counts)?
- Include **advanced metrics** (entropy, complexity)?
- Need for **visualization** tools?

**QUESTION 5**: Clinical Relevance

- Specific **therapeutic outcomes** to track?
- **Population characteristics** (age, clinical diagnoses)?
- **Baseline vs. change** metrics priority?

Please review these options and let me know your preferences so I can validate or adjust the current EDA metrics implementation.

## Current Implementation Status

✅ **Currently Implemented** (needs validation):
- **Extended Set (23 metrics)**
- 9 SCR metrics (count, rate, amplitude, timing)
- 5 Tonic metrics (central tendency, variability, range)
- 6 Phasic metrics (central tendency, variability, range, rate)
- 2 Metadata metrics

⏳ **Awaiting Validation**:
- Confirm 23-metric set is appropriate
- Determine if additional metrics needed
- Validate metrics selection matches research goals
- Confirm no domain-specific metrics missing

---

**References:**
- Boucsein, W. (2012). Electrodermal Activity (2nd ed.). Springer.
- Dawson, M. E., Schell, A. M., & Filion, D. L. (2007). The electrodermal system. In Handbook of Psychophysiology.
- Benedek, M., & Kaernbach, C. (2010). Decomposition of skin conductance data. Journal of Neuroscience Methods.
- Society for Psychophysiological Research (2012). Publication guidelines for EDA.
- Bach, D. R., et al. (2010). Time-series analysis for rapid event-related skin conductance responses. Journal of Neuroscience Methods.
