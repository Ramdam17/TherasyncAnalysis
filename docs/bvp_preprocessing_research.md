# NeuroKit2 BVP Preprocessing Methods - Research Findings

**Last Updated:** October 28, 2025  
**Status:** ✅ Decision Made - Option 1 (Automatic) Implemented  
**Implementation:** `src/physio/preprocessing/bvp_cleaner.py`

## Overview

Based on research of NeuroKit2 documentation, here are the available BVP (Blood Volume Pulse) preprocessing methods and options for our TherasyncPipeline project.

**Decision Outcome:** Option 1 (Automatic NeuroKit2 Pipeline) was selected and implemented. See `docs/bvp_decisions.md` for rationale.

## Main BVP Functions Available

### 1. Core BVP Processing Functions

- **`nk.ppg_process(signal, sampling_rate)`**: Main processing function
  - Cleans the signal
  - Detects peaks (pulse waves)  
  - Extracts physiological features
  - Returns processed signals and info dictionary

- **`nk.ppg_clean(signal, sampling_rate)`**: Signal cleaning/filtering
  - Removes noise and artifacts
  - Applies appropriate filtering

- **`nk.ppg_peaks(signal, sampling_rate)`**: Peak detection
  - Identifies pulse wave peaks
  - Critical for heart rate calculation

- **`nk.ppg_findpeaks(signal)`**: Alternative peak finding method

### 2. Analysis Functions

- **`nk.ppg_analyze(signal, sampling_rate)`**: General PPG analysis
- **`nk.ppg_eventrelated(signal, events)`**: Event-related analysis
- **`nk.ppg_intervalrelated(signal)`**: Interval-based analysis

### 3. Visualization and Simulation

- **`nk.ppg_plot(signals, info)`**: Visualization of processed signals
- **`nk.ppg_simulate(duration, sampling_rate, heart_rate)`**: Generate synthetic PPG

## Signal Preprocessing Options

### A. Signal Cleaning Methods (via `nk.ppg_clean()`)

NeuroKit2 provides automatic cleaning optimized for PPG signals:
- **Noise filtering**: Removes high-frequency noise
- **Baseline correction**: Handles signal drift
- **Artifact removal**: Eliminates movement artifacts

### B. General Signal Processing (can be applied to BVP)

From `nk.signal_*` functions that can be used with BVP data:

#### 1. **Detrending Methods** (`nk.signal_detrend()`)
- `method="polynomial"`: Polynomial detrending (specify order)
- `method="tarvainen2002"`: Advanced detrending method
- `method="locreg"`: Local regression detrending
- Linear drift removal (order=0): Mean removal

#### 2. **Filtering Methods** (`nk.signal_filter()`)
- **Low-pass filtering**: `nk.signal_filter(signal, lowcut=None, highcut=X)`
- **High-pass filtering**: `nk.signal_filter(signal, lowcut=X, highcut=None)`
- **Band-pass filtering**: `nk.signal_filter(signal, lowcut=X, highcut=Y)`
- **Butterworth filters**: Default filter type

#### 3. **Standardization Methods** (`nk.standardize()`)
- **Global standardization**: Z-score normalization across entire signal
- **Windowed standardization**: `nk.standardize(signal, window=samples)`
  - Rolling standardization with specified window size
  - Example: `window=sampling_rate*2` for 2-second windows

## Recommended BVP Preprocessing Pipeline Options

Based on the documentation analysis, here are **three preprocessing approaches** for user selection:

### Option 1: **Automatic NeuroKit2 Pipeline** (Recommended for beginners)
```python
# Complete automated processing
signals, info = nk.ppg_process(bvp_signal, sampling_rate=64)
cleaned_bvp = signals["PPG_Clean"]
```
**Pros**: Fully automatic, optimized for PPG signals, includes peak detection
**Cons**: Less control over individual preprocessing steps

### Option 2: **Manual Pipeline with Standard Methods**
```python
# Step-by-step preprocessing with more control
bvp_clean = nk.ppg_clean(bvp_signal, sampling_rate=64)
bvp_detrended = nk.signal_detrend(bvp_clean, method="polynomial", order=2)
bvp_filtered = nk.signal_filter(bvp_detrended, lowcut=0.5, highcut=8.0)
```
**Pros**: Full control over each step, customizable parameters
**Cons**: Requires more parameter tuning

### Option 3: **Windowed Standardization Pipeline**
```python
# Combines NeuroKit cleaning with windowed standardization
bvp_clean = nk.ppg_clean(bvp_signal, sampling_rate=64)
bvp_standardized = nk.standardize(bvp_clean, window=128)  # 2-second windows at 64Hz
```
**Pros**: Good for signals with varying baseline, robust to artifacts
**Cons**: May affect absolute signal amplitude information

## Filtering Recommendations for BVP (64Hz sampling rate)

Based on physiological considerations:

### Frequency Bands for BVP Analysis
- **Heart rate range**: 0.5 - 4.0 Hz (30-240 BPM)
- **Typical adult range**: 0.8 - 2.5 Hz (48-150 BPM) 
- **Noise above**: >8 Hz (movement artifacts, electrical noise)

### Suggested Filter Parameters
- **Low-pass cutoff**: 8 Hz (removes high-frequency noise)
- **High-pass cutoff**: 0.5 Hz (removes very low-frequency drift)
- **Band-pass**: 0.5-8 Hz for clean heart rate analysis

## Configuration Parameters for config.yaml

```yaml
physio:
  bvp:
    preprocessing:
      method: "automatic"  # Options: "automatic", "manual", "windowed"
      
      # Automatic method parameters
      ppg_process:
        sampling_rate: 64
        
      # Manual method parameters  
      cleaning:
        enabled: true
      detrending:
        method: "polynomial"  # Options: "polynomial", "tarvainen2002", "locreg"
        order: 2  # For polynomial method
      filtering:
        enabled: true
        lowcut: 0.5   # Hz - removes drift
        highcut: 8.0  # Hz - removes noise
        
      # Windowed standardization parameters
      standardization:
        enabled: false
        window_seconds: 2  # Will be converted to samples (64*2=128)
```

## Next Steps - User Decision Required

**QUESTION**: Which BVP preprocessing approach would you prefer for the Therasync project?

1. **Option 1**: Automatic NeuroKit2 pipeline (simplest, recommended)
2. **Option 2**: Manual pipeline with configurable filtering and detrending  
3. **Option 3**: Windowed standardization approach
4. **Custom**: Combination of methods or different approach

**Additional questions**:
- Do you want to preserve raw signal amplitude information or is normalized data sufficient?
- Are there specific frequency bands of interest for your family therapy analysis?
- Should the preprocessing be configurable per study or standardized?

Please review these options and let me know your preference so I can implement the selected approach in the BVP cleaning module.