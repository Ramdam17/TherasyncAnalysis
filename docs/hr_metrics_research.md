# HR Metrics Research - TherasyncPipeline

**Authors**: Lena Adel, Remy Ramadour  
**Date**: October 28, 2025  
**Version**: v0.3.0 (Modular Architecture)  
**Status:** ✅ IMPLEMENTED & PRODUCTION-READY  
**Implementation:** `src/physio/preprocessing/hr_*.py`

## Overview

This document explores Heart Rate (HR) metrics for the Therasync project, focusing on **basic HR descriptive statistics and trends** rather than Heart Rate Variability (HRV) metrics, which are already covered by the BVP pipeline.

**Important Note**: HRV metrics (RMSSD, pNN50, frequency domain, etc.) are already extracted by the BVP pipeline (18 HRV metrics). This HR pipeline focuses on complementary metrics derived from the direct HR signal.

## HR vs HRV Distinction

### HR Metrics (This Pipeline)
- Direct analysis of instantaneous heart rate values (BPM)
- Descriptive statistics of HR signal
- HR trends and patterns over time
- HR stability and response characteristics

### HRV Metrics (Already in BVP Pipeline)
- Analysis of R-R interval variability
- Time-domain: RMSSD, pNN50, SDNN, etc.
- Frequency-domain: LF, HF, LF/HF ratio, etc.
- Non-linear: SD1, SD2, Sample Entropy, etc.

## Available HR Data

From Empatica E4 devices:
- **Sampling Rate**: 1 Hz (1 sample per second)
- **Unit**: BPM (beats per minute)
- **Signal Type**: Instantaneous heart rate
- **Typical Range**: 50-150 BPM (rest to moderate activity)

## Proposed HR Metrics Categories

### 1. Descriptive Statistics (7 metrics)

Basic statistical measures of HR signal:

| Metric | Description | Clinical Relevance |
|--------|-------------|-------------------|
| `HR_Mean` | Mean heart rate (BPM) | Overall cardiovascular demand |
| `HR_Median` | Median heart rate (BPM) | Central tendency, robust to outliers |
| `HR_SD` | Standard deviation of HR | HR variability (different from HRV) |
| `HR_Min` | Minimum heart rate (BPM) | Resting/recovery capacity |
| `HR_Max` | Maximum heart rate (BPM) | Peak cardiovascular response |
| `HR_Range` | HR range (Max - Min) | Response amplitude |
| `HR_IQR` | Interquartile range (Q3 - Q1) | Central 50% variability |

### 2. Trend Analysis (5 metrics)

Changes in HR over time:

| Metric | Description | Clinical Relevance |
|--------|-------------|-------------------|
| `HR_Slope` | Linear trend (BPM/minute) | HR adaptation over session |
| `HR_Initial` | HR in first 10% of session | Initial state |
| `HR_Final` | HR in last 10% of session | Final state |
| `HR_Change` | Final - Initial HR | Net change |
| `HR_Peak_Time` | Time of maximum HR | Response timing |

### 3. Stability Metrics (4 metrics)

HR consistency and fluctuation patterns:

| Metric | Description | Clinical Relevance |
|--------|-------------|-------------------|
| `HR_Stability` | 1 / (1 + HR_SD) | Inverse measure of variability |
| `HR_RMSSD_Simple` | RMS of successive differences | Beat-to-beat changes (simplified) |
| `HR_CV` | Coefficient of variation (SD/Mean) | Normalized variability |
| `HR_MAD` | Median absolute deviation | Robust variability measure |

### 4. Response Patterns (6 metrics)

Physiological response characteristics:

| Metric | Description | Clinical Relevance |
|--------|-------------|-------------------|
| `HR_Elevated_Percent` | % time above resting + 20 BPM | Activation periods |
| `HR_Recovery_Rate` | HR decline rate after peaks | Autonomic recovery |
| `HR_Acceleration_Mean` | Mean HR acceleration | Response dynamics |
| `HR_Deceleration_Mean` | Mean HR deceleration | Recovery dynamics |
| `HR_Plateaus_Count` | Number of stable periods (±3 BPM) | Steady states |
| `HR_Peaks_Count` | Number of HR peaks (>10 BPM above baseline) | Response episodes |

### 5. Contextual Metrics (3 metrics)

Session and data quality information:

| Metric | Description | Clinical Relevance |
|--------|-------------|-------------------|
| `HR_Duration` | Signal duration (seconds) | Session length |
| `HR_Samples` | Number of samples | Data completeness |
| `HR_Quality` | Data quality score (0-1) | Reliability indicator |

## Total: 25 HR Metrics

**Summary by category**:
- Descriptive Statistics: 7 metrics
- Trend Analysis: 5 metrics  
- Stability Metrics: 4 metrics
- Response Patterns: 6 metrics
- Contextual Metrics: 3 metrics

## Implementation Approach

### 1. Basic Statistics
```python
# Simple pandas/numpy operations
hr_mean = data['hr'].mean()
hr_std = data['hr'].std()
hr_min, hr_max = data['hr'].min(), data['hr'].max()
```

### 2. Trend Analysis
```python
# Linear regression for trend
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(data['time'], data['hr'])
```

### 3. Peak Detection
```python
# Using scipy.signal for peak detection
from scipy.signal import find_peaks
peaks, properties = find_peaks(data['hr'], prominence=10, distance=60)  # 1 min apart
```

### 4. Response Patterns
```python
# Calculate acceleration/deceleration
hr_diff = data['hr'].diff()
acceleration = hr_diff[hr_diff > 0].mean()
deceleration = hr_diff[hr_diff < 0].mean()
```

## Clinical Interpretation

### Resting vs Therapy Comparison

**Expected patterns**:
- **Resting**: Lower mean HR, smaller range, more stable
- **Therapy**: Higher mean HR, larger range, more variable

**Key indicators**:
- `HR_Mean`: Basic cardiovascular demand
- `HR_Range`: Response amplitude  
- `HR_Stability`: Emotional regulation
- `HR_Slope`: Adaptation/habituation

### Therapeutic Insights

1. **Stress Response**: Higher HR_Max, larger HR_Range
2. **Emotional Regulation**: Higher HR_Stability over time
3. **Engagement**: Moderate HR elevation, responsive patterns
4. **Recovery**: Good HR_Recovery_Rate after responses

## Validation Strategy

### 1. Physiological Plausibility
- HR values in normal range (40-180 BPM)
- Reasonable HR_Range for context
- Consistent with session type

### 2. Cross-Validation with HRV
- Compare HR_Mean with HRV baseline
- Validate HR_Stability against HRV complexity measures
- Check consistency with BVP-derived patterns

### 3. Inter-Subject Variability
- Expect individual differences in HR patterns
- Family-level similarities possible
- Session-to-session consistency within subjects

## Technical Considerations

### Data Quality
- Handle missing values (interpolation for gaps <5s)
- Detect outliers (HR outside 40-180 BPM)
- Quality scoring based on completeness and plausibility

### Computational Efficiency
- All metrics computable with basic numpy/pandas
- No complex signal processing required
- Fast execution (< 0.1s per subject/session)

### BIDS Compliance
- Save metrics in TSV format with JSON metadata
- Include units, descriptions, and computation parameters
- Follow BIDS derivatives naming conventions

## Configuration

Proposed config.yaml structure:
```yaml
physio:
  hr:
    sampling_rate: 1  # Hz
    processing:
      method: "basic_stats"
      outlier_threshold: [40, 180]  # BPM range
      interpolation_max_gap: 5  # seconds
      peak_prominence: 10  # BPM
      stable_threshold: 3  # BPM for plateaus
    metrics:
      extract_all: false
      selected_metrics:
        descriptive:
          - "HR_Mean"
          - "HR_SD"
          - "HR_Min"
          - "HR_Max"
          - "HR_Range"
        trends:
          - "HR_Slope"
          - "HR_Change"
        stability:
          - "HR_Stability"
          - "HR_CV"
        response:
          - "HR_Elevated_Percent"
          - "HR_Peaks_Count"
        contextual:
          - "HR_Duration"
          - "HR_Quality"
```

## Next Steps

1. **User Review**: Select subset of 10-15 most relevant metrics
2. **Implementation**: Create HRMetricsExtractor class
3. **Testing**: Validate on real data (5 subjects)
4. **Integration**: Add to HR pipeline and BIDS writer
5. **Documentation**: Add to API reference and troubleshooting guide

## Literature Context

While HRV is extensively studied for autonomic function, basic HR metrics provide complementary information:

- **HR_Mean**: Reflects metabolic demand and arousal level
- **HR_Stability**: May indicate emotional regulation capacity  
- **HR_Response_Patterns**: Show engagement and stress reactivity
- **HR_Trends**: Capture adaptation and habituation

These metrics are particularly relevant for family therapy analysis where:
- Mean HR may reflect session engagement
- HR stability may indicate emotional regulation
- Response patterns may show synchrony between family members

---

**Status**: Research complete, ready for user review and metric selection.  
**Total Metrics Available**: 25 HR metrics (complementary to 18 HRV metrics from BVP)  
**Implementation Complexity**: Low (basic statistics and signal processing)  
**Expected Performance**: < 0.1s per subject/session