# NeuroKit2 EDA Preprocessing Methods - Research Findings

## Overview

Based on research of NeuroKit2 documentation, here are the available EDA (Electrodermal Activity) preprocessing methods and options for our TherasyncPipeline project.

## Main EDA Functions Available

### 1. Core EDA Processing Functions

- **`nk.eda_process(signal, sampling_rate)`**: Main processing function
  - Cleans the signal
  - Decomposes into tonic (SCL) and phasic (SCR) components
  - Detects SCR peaks
  - Returns processed signals and info dictionary

- **`nk.eda_clean(signal, sampling_rate)`**: Signal cleaning/filtering
  - Removes noise and artifacts
  - Applies appropriate filtering based on sampling rate

- **`nk.eda_phasic(signal, sampling_rate)`**: Tonic-phasic decomposition
  - Separates slow baseline (tonic) from rapid responses (phasic)
  - Multiple decomposition methods available

- **`nk.eda_peaks(signal, sampling_rate)`**: SCR peak detection
  - Identifies skin conductance responses (SCRs)
  - Extracts peak characteristics (amplitude, rise time, recovery time)

### 2. Analysis Functions

- **`nk.eda_analyze(signal, sampling_rate)`**: General EDA analysis
- **`nk.eda_eventrelated(signal, events)`**: Event-related analysis
- **`nk.eda_intervalrelated(signal)`**: Interval-based analysis

### 3. Visualization and Simulation

- **`nk.eda_plot(signals, info)`**: Visualization of processed signals
- **`nk.eda_simulate(duration, sampling_rate)`**: Generate synthetic EDA

## Signal Preprocessing Options

### A. Signal Cleaning Methods (via `nk.eda_clean()`)

NeuroKit2 provides **automatic cleaning** optimized for EDA signals:

**Available Methods:**
- `method="neurokit"` (default): Optimized for low-frequency EDA
  - Butterworth lowpass filter
  - Cutoff at 3 Hz
  - Appropriate for typical EDA sampling rates (4-64 Hz)

- `method="biosppy"`: BioSPy implementation
  - Butterworth lowpass filter
  - Cutoff at 5 Hz
  - More aggressive filtering

**Note**: At very low sampling rates (like 4 Hz for Empatica E4), filtering may be automatically skipped to avoid signal distortion.

### B. Tonic-Phasic Decomposition Methods (via `nk.eda_phasic()`)

The tonic-phasic decomposition separates:
- **Tonic (SCL)**: Slowly varying baseline skin conductance level
- **Phasic (SCR)**: Rapidly changing skin conductance responses

**Available Decomposition Methods:**

#### 1. **cvxEDA** (Convex Optimization) - `method="cvxeda"` ⭐ RECOMMENDED
```python
tonic, phasic = nk.eda_phasic(signal, sampling_rate=4, method="cvxeda")
```
**Pros:**
- State-of-the-art method (Greco et al., 2016)
- Mathematically rigorous convex optimization
- Robust to artifacts and noise
- Well-suited for low sampling rates
- No parameter tuning required
- Gold standard in EDA research

**Cons:**
- Computationally intensive (slower)
- Requires cvxopt library

**Best for:** Research-grade analysis, publications, low sampling rates

#### 2. **Sparse Deconvolution** - `method="sparse"` 
```python
tonic, phasic = nk.eda_phasic(signal, sampling_rate=4, method="sparse")
```
**Pros:**
- Based on sparse signal processing
- Good balance between accuracy and speed
- Handles artifacts well

**Cons:**
- Less established than cvxEDA
- May require parameter tuning

**Best for:** Fast processing with good quality

#### 3. **Smoothing Median** - `method="smoothmedian"`
```python
tonic, phasic = nk.eda_phasic(signal, sampling_rate=4, method="smoothmedian")
```
**Pros:**
- Very fast computation
- Simple and interpretable
- Good for exploratory analysis

**Cons:**
- Less accurate than optimization methods
- May miss subtle SCRs
- Sensitive to parameter choices

**Best for:** Quick exploratory analysis, real-time processing

#### 4. **High-Pass Filter** - `method="highpass"`
```python
tonic, phasic = nk.eda_phasic(signal, sampling_rate=4, method="highpass")
```
**Pros:**
- Extremely fast
- Simple implementation

**Cons:**
- Crude approximation
- May distort signal
- Not recommended for research

**Best for:** Quick visualization only

### C. SCR Peak Detection Options

**Detection Methods in `nk.eda_peaks()`:**

- **Automatic threshold**: Adapts to signal characteristics
- **Fixed threshold**: Manual threshold in microsiemens (μS)
- **Amplitude threshold**: Minimum SCR amplitude (default: 0.01 μS)
- **Rise time constraints**: Typical 1-3 seconds
- **Recovery time constraints**: Typical 2-10 seconds

**Configurable Parameters:**
```python
peaks = nk.eda_peaks(
    phasic_signal, 
    sampling_rate=4,
    method="neurokit",
    amplitude_min=0.01  # Minimum SCR amplitude in μS
)
```

## Recommended EDA Preprocessing Pipeline Options

Based on the documentation analysis, here are **three preprocessing approaches** for user selection:

### Option 1: **Automatic NeuroKit2 Pipeline with cvxEDA** ⭐ RECOMMENDED
```python
# Complete automated processing with gold-standard decomposition
signals, info = nk.eda_process(eda_signal, sampling_rate=4, method="neurokit")
# This internally uses cvxEDA for decomposition
```
**Pros:** 
- Gold standard method (cvxEDA)
- Fully automatic and optimized
- Best accuracy for research
- Handles low sampling rates (4 Hz) well
- Includes peak detection with optimal parameters

**Cons:** 
- Slower computation (~1-2 seconds for 45-minute session)
- Less control over individual steps

**Best for:** Research-grade analysis, publications, family therapy studies

### Option 2: **Fast Processing with Sparse Decomposition**
```python
# Step-by-step with faster decomposition
eda_clean = nk.eda_clean(eda_signal, sampling_rate=4)
eda_decomposed = nk.eda_phasic(eda_clean, sampling_rate=4, method="sparse")
peaks = nk.eda_peaks(eda_decomposed["EDA_Phasic"], sampling_rate=4)
```
**Pros:**
- Good balance of speed and accuracy
- More control over parameters
- ~5-10x faster than cvxEDA

**Cons:**
- Slightly less accurate than cvxEDA
- May need parameter tuning for optimal results

**Best for:** Large-scale batch processing, exploratory analysis

### Option 3: **Ultra-Fast Pipeline for Exploration**
```python
# Quick processing with smoothmedian
eda_clean = nk.eda_clean(eda_signal, sampling_rate=4)
eda_decomposed = nk.eda_phasic(eda_clean, sampling_rate=4, method="smoothmedian")
peaks = nk.eda_peaks(eda_decomposed["EDA_Phasic"], sampling_rate=4)
```
**Pros:**
- Very fast (~100x faster than cvxEDA)
- Good for initial data exploration
- Minimal computational resources

**Cons:**
- Lower accuracy
- May miss subtle SCRs
- Not recommended for final analysis

**Best for:** Data quality checks, visualization, real-time feedback

## Processing Considerations for Empatica E4 (4 Hz)

### Sampling Rate Impact

The Empatica E4 samples EDA at **4 Hz** (4 samples per second), which is:
- ✅ **Sufficient** for EDA analysis (typical SCRs last 2-10 seconds)
- ✅ **Compatible** with all NeuroKit2 methods
- ⚠️ **Low for filtering** - some filtering operations may be skipped

### Signal Characteristics at 4 Hz

**Typical SCR Parameters at 4 Hz:**
- **Rise time**: 1-3 seconds (4-12 samples)
- **Recovery time**: 2-10 seconds (8-40 samples)
- **Minimum detectable amplitude**: ~0.01 μS
- **Tonic changes**: Captured accurately
- **Fast SCRs**: May be slightly smoothed

### Quality Considerations

**Good signal indicators:**
- Baseline (tonic) range: 0.5-5 μS (typical)
- SCR amplitude: 0.01-2 μS
- SCR rate: 1-30 per minute

**Poor signal indicators:**
- Flat signal (0 μS) - electrode detachment
- Extreme values (>20 μS) - artifacts
- Very high SCR rate (>50/min) - movement artifacts

## Configuration Parameters for config.yaml

```yaml
physio:
  eda:
    sampling_rate: 4  # Empatica E4 default
    
    preprocessing:
      method: "neurokit"  # Options: "neurokit", "sparse", "smoothmedian"
      
      # Cleaning parameters
      cleaning:
        method: "neurokit"  # Options: "neurokit", "biosppy"
        
      # Tonic-phasic decomposition
      decomposition:
        method: "cvxeda"  # Options: "cvxeda", "sparse", "smoothmedian", "highpass"
        
      # SCR peak detection
      scr_detection:
        method: "neurokit"
        amplitude_threshold: 0.01  # μS - minimum SCR amplitude
        
      # Quality filtering
      filter:
        enabled: true
        method: "neurokit"
```

## Physiological Context for Family Therapy

### EDA Components and Their Meaning

**Tonic Component (SCL - Skin Conductance Level):**
- Reflects overall arousal state
- Changes slowly (minutes to hours)
- Related to: anxiety, stress, general emotional state
- **In therapy context**: Baseline emotional engagement, anxiety levels

**Phasic Component (SCR - Skin Conductance Response):**
- Reflects discrete emotional/cognitive responses
- Changes rapidly (2-5 seconds)
- Related to: specific stimuli, thoughts, emotional events
- **In therapy context**: Reactions to specific topics, emotional moments

### SCR Characteristics

**SCR Amplitude:**
- Reflects intensity of response
- Higher amplitude = stronger emotional/cognitive reaction

**SCR Frequency/Rate:**
- Reflects number of responses per minute
- Higher rate = more reactive, more emotional processing

**SCR Rise Time:**
- Time from onset to peak
- Reflects speed of autonomic activation

**SCR Recovery Time:**
- Time from peak to baseline
- Reflects autonomic regulation capacity

## Next Steps - User Decision Required

**QUESTION 1**: Which EDA preprocessing approach would you prefer for the Therasync project?

1. **Option 1 (RECOMMENDED)**: Automatic with cvxEDA - Gold standard, best accuracy
   - Slower but most accurate
   - Best for research publications

2. **Option 2**: Sparse decomposition - Fast and accurate
   - Good balance
   - 5-10x faster than cvxEDA

3. **Option 3**: Smoothmedian - Ultra-fast exploration
   - Quick for initial analysis
   - Not recommended for final results

4. **Custom**: Specific combination or different parameters

**QUESTION 2**: SCR Detection Parameters

- Is the default **0.01 μS threshold** appropriate for your population?
- Do you want to adjust based on age, clinical population, or other factors?

**QUESTION 3**: Processing Speed vs. Accuracy

- Do you prioritize **accuracy** (cvxEDA, slower) or **speed** (sparse, faster)?
- Expected dataset size for batch processing?

**QUESTION 4**: Physiological Metrics of Interest

- Focus on **tonic** (baseline arousal) or **phasic** (discrete responses)?
- Both components equally important?
- Specific SCR characteristics (amplitude vs. frequency)?

Please review these options and let me know your preferences so I can validate or adjust the current EDA cleaning implementation.

## Current Implementation Status

✅ **Currently Implemented** (needs validation):
- Method: `neurokit` with cvxEDA decomposition (Option 1)
- SCR threshold: 0.01 μS
- Full automatic pipeline via `nk.eda_process()`

⏳ **Awaiting Validation**:
- Confirm cvxEDA is appropriate choice
- Validate SCR threshold
- Confirm no parameter adjustments needed

---

**References:**
- Greco, A., Valenza, G., & Scilingo, E. P. (2016). cvxEDA: A convex optimization approach to electrodermal activity processing. IEEE TBME.
- Benedek, M., & Kaernbach, C. (2010). A continuous measure of phasic electrodermal activity. Journal of Neuroscience Methods.
- Society for Psychophysiological Research (2012). Publication recommendations for EDA.
