# Troubleshooting Guide - TherasyncPipeline

**Authors**: Lena Adel, Remy Ramadour  
**Version**: 0.1.0  
**Last Updated**: October 28, 2025

This guide helps resolve common issues when using the TherasyncPipeline.

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Data Loading Problems](#data-loading-problems)
3. [Processing Errors](#processing-errors)
   - [BVP Processing Errors](#bvp-processing-errors)
     - [Peak Detection Fails](#peak-detection-fails)
     - [Memory Errors](#memory-errors)
     - [Sampling Rate Mismatch](#sampling-rate-mismatch)
   - [EDA Signal Processing Errors](#eda-signal-processing-errors)
     - [cvxEDA Convergence Failure](#cvxeda-convergence-failure)
     - [SCR Detection Issues](#scr-detection-issues)
     - [Low/Negative EDA Values](#lownegative-eda-values)
     - [Sampling Rate Warning](#sampling-rate-warning)
4. [Quality Warnings](#quality-warnings)
   - [BVP Quality Warnings](#bvp-quality-warnings)
     - [Low Signal Quality Warning](#low-signal-quality-warning)
     - [ParserWarning](#parserwarning)
   - [EDA Quality Warnings](#eda-quality-warnings)
     - [Unusual SCR Rates](#unusual-scr-rates)
     - [Atypical Tonic EDA Levels](#atypical-tonic-eda-levels)
5. [Output Issues](#output-issues)
6. [Performance Problems](#performance-problems)
7. [Common Warnings Explained](#common-warnings-explained)

---

## Installation Issues

### Poetry Installation Fails

**Problem**: `poetry install` fails with dependency conflicts

```bash
Error: The current project's Python requirement (>=3.8,<4.0) is not compatible
```

**Solutions**:

1. **Check Python version**:
```bash
python --version  # Should be 3.8 or higher
```

2. **Update Poetry**:
```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry self update
```

3. **Clear Poetry cache**:
```bash
poetry cache clear pypi --all
poetry install
```

4. **Use specific Python version**:
```bash
poetry env use python3.12
poetry install
```

---

### Missing System Dependencies

**Problem**: NeuroKit2 or scipy installation fails

```bash
ERROR: Failed building wheel for scipy
```

**Solution** (Ubuntu/Debian):
```bash
sudo apt-get update
sudo apt-get install -y python3-dev build-essential gfortran libopenblas-dev liblapack-dev
poetry install
```

**Solution** (macOS):
```bash
brew install openblas lapack
poetry install
```

---

### Import Errors After Installation

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solutions**:

1. **Use PYTHONPATH**:
```bash
PYTHONPATH=. poetry run python scripts/preprocess_bvp.py --help
```

2. **Install in editable mode**:
```bash
poetry install
```

3. **Check virtual environment**:
```bash
poetry env info  # Shows active environment
poetry shell     # Activate environment
```

---

## Data Loading Problems

### File Not Found Errors

**Problem**: `FileNotFoundError: No BVP files found for sub-f01p01/ses-01`

**Diagnosis**:
```bash
# Check data structure
tree data/raw/sub-f01p01/ses-01/ -L 2

# Expected structure:
# data/raw/sub-f01p01/ses-01/physio/
#   ├── sub-f01p01_ses-01_task-restingstate_recording-bvp.tsv
#   ├── sub-f01p01_ses-01_task-restingstate_recording-bvp.json
#   ├── sub-f01p01_ses-01_task-therapy_recording-bvp.tsv
#   └── sub-f01p01_ses-01_task-therapy_recording-bvp.json
```

**Solutions**:

1. **Check config paths**:
```yaml
# config/config.yaml
paths:
  sourcedata: "data/raw"  # Must match your data location
```

2. **Verify file naming**:
   - Files must follow BIDS naming: `{subject}_{session}_task-{moment}_recording-bvp.{tsv,json}`
   - Subject must include `sub-` prefix
   - Session must include `ses-` prefix

3. **Check file permissions**:
```bash
ls -la data/raw/sub-f01p01/ses-01/physio/
# Files should be readable (r-- in permissions)
```

---

### Missing JSON Sidecar

**Problem**: `FileNotFoundError: JSON sidecar not found: *.json`

**Solution**:

Every TSV file must have a matching JSON sidecar with metadata:

```json
{
  "SamplingFrequency": 64,
  "StartTime": 0.0,
  "Columns": ["time", "bvp"],
  "Units": ["seconds", "AU"],
  "TaskName": "restingstate",
  "RecordingType": "bvp"
}
```

Check that both files exist:
```bash
ls -1 data/raw/sub-f01p01/ses-01/physio/*task-restingstate*
# Should show both .tsv and .json
```

---

### Data Format Errors

**Problem**: `ValueError: Invalid data structure`

**Diagnosis**:
```python
import pandas as pd

# Load TSV manually
data = pd.read_csv('path/to/file.tsv', sep='\t')
print(data.columns)  # Should be ['time', 'bvp']
print(data.dtypes)   # Both should be numeric
print(data.head())
```

**Required Format**:
```
time	bvp
0.000	-0.123
0.016	0.456
0.031	-0.789
```

**Common Issues**:
- ❌ Wrong separator (commas instead of tabs)
- ❌ Missing column names
- ❌ Non-numeric values
- ❌ Empty rows

**Fix**:
```python
# Correct TSV format
data.to_csv('fixed_file.tsv', sep='\t', index=False, float_format='%.6f')
```

---

## Processing Errors

### BVP Processing Errors

#### Peak Detection Fails

**Problem**: `ValueError: No peaks detected in signal`

**Causes**:
1. Signal too short (<10 seconds)
2. Signal quality too poor
3. Incorrect sampling rate

**Solutions**:

1. **Check signal length**:
```python
duration = data['time'].max()
print(f"Duration: {duration:.1f} seconds")
# Minimum: ~10 seconds for reliable peak detection
```

2. **Verify sampling rate**:
```yaml
# config/config.yaml
physio:
  bvp:
    sampling_rate: 64  # Must match actual data
```

3. **Try different detection method**:
```yaml
physio:
  bvp:
    processing:
      method: "elgendi"  # Try: "msptd", "bishop", "adaptive"
```

4. **Visualize signal**:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(data['time'], data['bvp'])
plt.xlabel('Time (s)')
plt.ylabel('BVP (AU)')
plt.title('Raw BVP Signal')
plt.show()
# Look for clear pulsatile pattern
```

---

#### Memory Errors

**Problem**: `MemoryError` or system freezes during processing

**Solution** for long recordings:

1. **Process moments separately**:
```bash
# Instead of processing all at once
poetry run python scripts/preprocess_bvp.py --subject sub-f01p01 --session ses-01 --moment restingstate
poetry run python scripts/preprocess_bvp.py --subject sub-f01p01 --session ses-01 --moment therapy
```

2. **Increase system swap**:
```bash
# Check available memory
free -h

# For very long recordings (>60 min), ensure sufficient RAM
```

3. **Chunk processing** (for future implementation):
   - Process in 10-minute windows
   - Combine results afterward

---

#### Sampling Rate Mismatch

**Problem**: `ValueError: Sampling rate mismatch. Expected 64 Hz, got 32 Hz`

**Solution**:

1. **Update config to match data**:
```yaml
physio:
  bvp:
    sampling_rate: 32  # Match your device's sampling rate
```

2. **Check JSON metadata**:
```json
{
  "SamplingFrequency": 32  # Must match actual data
}
```

3. **Verify actual sampling rate**:
```python
import pandas as pd

data = pd.read_csv('file.tsv', sep='\t')
time_diff = data['time'].diff().median()
actual_rate = 1 / time_diff
print(f"Actual sampling rate: {actual_rate:.1f} Hz")
```

---

### EDA Signal Processing Errors

#### cvxEDA Convergence Failure

**Problem**: `RuntimeError: cvxEDA optimization did not converge`

**Causes**:
1. Signal contains NaN or infinite values
2. Signal extremely noisy or corrupted
3. Insufficient signal length (<10 seconds)
4. Numerical instability with extreme values

**Solutions**:

1. **Check for invalid values**:
```python
import pandas as pd
import numpy as np

data = pd.read_csv('EDA.tsv', sep='\t')
print(f"NaN values: {data['eda'].isna().sum()}")
print(f"Infinite values: {np.isinf(data['eda']).sum()}")
print(f"Signal range: [{data['eda'].min():.3f}, {data['eda'].max():.3f}] μS")
```

2. **Verify signal length**:
```python
duration = len(data) / 4  # 4 Hz sampling rate
print(f"Duration: {duration:.1f} seconds")
# Minimum: 10 seconds for cvxEDA
```

3. **Try fallback method** (if NeuroKit2 cvxEDA fails):
```yaml
# config/config.yaml
physio:
  eda:
    processing:
      method: "neurokit"  # Uses median filter decomposition as fallback
```

4. **Inspect signal visually**:
```python
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.plot(data['eda'])
plt.xlabel('Sample')
plt.ylabel('EDA (μS)')
plt.title('Raw EDA Signal')
plt.show()
# Look for extreme spikes, flat lines, or artifacts
```

---

#### SCR Detection Issues

**Problem 1**: `WARNING: Very few SCRs detected (0 peaks in 60s)`

**Causes**:
- Subject extremely calm/relaxed
- SCR threshold too high
- Poor electrode contact (flat signal)

**Solutions**:

1. **Check signal quality**:
```python
# View raw signal statistics
data = pd.read_csv('EDA.tsv', sep='\t')
print(f"Mean: {data['eda'].mean():.3f} μS")
print(f"Std: {data['eda'].std():.3f} μS")
print(f"Range: [{data['eda'].min():.3f}, {data['eda'].max():.3f}] μS")

# Flat signal (std < 0.01) suggests poor contact
```

2. **Adjust SCR threshold**:
```yaml
# config/config.yaml
physio:
  eda:
    processing:
      scr_threshold: 0.01  # Lower to detect smaller SCRs (default: 0.01)
```

3. **Visualize detected SCRs**:
```python
# Check processed output
events = pd.read_csv('sub-f01p01_ses-01_task-restingstate_desc-scr_events.tsv', sep='\t')
print(f"SCRs detected: {len(events)}")
print(events[['onset', 'amplitude', 'rise_time']].head())
```

**Problem 2**: `WARNING: Excessive SCRs detected (>50 SCRs/min)`

**Causes**:
- Motion artifacts creating false peaks
- Electrical interference
- SCR threshold too low
- Subject highly anxious

**Solutions**:

1. **Increase SCR threshold**:
```yaml
physio:
  eda:
    processing:
      scr_threshold: 0.02  # Increase to reduce false positives
```

2. **Check for artifacts**:
```python
# Look at SCR amplitude distribution
events = pd.read_csv('*_desc-scr_events.tsv', sep='\t')
events['amplitude'].hist(bins=30)
plt.xlabel('SCR Amplitude (μS)')
plt.ylabel('Count')
plt.title('SCR Amplitude Distribution')
plt.show()
# Many very small amplitudes (<0.01 μS) suggest noise
```

---

#### Low/Negative EDA Values

**Problem**: `WARNING: EDA signal contains negative values`

**Cause**: Poor electrode contact or data preprocessing artifact

**Solutions**:

1. **Check electrode quality**:
   - Ensure proper skin prep (clean, dry)
   - Use isotonic electrode gel
   - Check electrode placement (palmar surfaces)

2. **Offset correction** (if needed):
```python
# Add offset to make signal positive
data['eda'] = data['eda'] - data['eda'].min() + 0.01
```

3. **Verify data integrity**:
```bash
# Check raw source data
cat data/sourcedata/sub-f01p01/ses-01/physio/EDA.tsv | head -20
# Should show reasonable values (typically 0.5-20 μS)
```

---

#### Sampling Rate Warning

**Problem**: `NeuroKitWarning: EDA signal is sampled at very low frequency. Skipping filtering.`

**Context**: 
- This is **EXPECTED** for Empatica E4 data (4 Hz)
- Not an error, just informational
- Pipeline still processes correctly

**Why it appears**:
- NeuroKit2 recommends >10 Hz for optimal filtering
- 4 Hz is low but acceptable for EDA
- Bandpass filtering is skipped to avoid artifacts

**No action needed** - pipeline handles this automatically.

---

## Quality Warnings

### BVP Quality Warnings

#### Low Signal Quality Warning

**Warning**: `Low signal quality for BVP therapy: mean quality 0.699 < threshold 0.8`

**Understanding**:
- Quality score ranges 0-1 (higher = better)
- Threshold default: 0.8
- **Important**: Scores 0.65-0.75 are typical for long recordings (>30 min) and remain usable

**Quality Score Interpretation**:

| Range | Quality | Action |
|-------|---------|--------|
| 0.8-1.0 | Excellent | Proceed normally |
| 0.7-0.8 | Good | Proceed normally |
| 0.65-0.7 | Acceptable | ✓ Normal for long sessions |
| 0.5-0.65 | Marginal | Review manually |
| <0.5 | Poor | Investigate signal issues |

**When to Worry**:
- ❌ Quality < 0.5: Likely sensor issues
- ❌ Rapid quality drops: Movement artifacts
- ❌ No peaks detected: Complete signal loss

**When NOT to Worry**:
- ✅ Quality 0.65-0.75 for >30 min recordings: **Normal**
- ✅ Gradual quality decrease: Natural variation
- ✅ Many peaks detected: Signal still usable

**Solutions**:

1. **Adjust threshold for long recordings**:
```yaml
physio:
  bvp:
    processing:
      quality_threshold: 0.7  # More permissive for long sessions
```

2. **Review quality over time**:
```python
import matplotlib.pyplot as plt

# Plot quality scores
plt.figure(figsize=(12, 4))
plt.plot(processed['PPG_Quality'])
plt.axhline(y=0.8, color='r', linestyle='--', label='Threshold')
plt.xlabel('Sample')
plt.ylabel('Quality Score')
plt.legend()
plt.show()
```

3. **Check peak detection results**:
```python
# Even with lower quality, peaks may be reliable
num_peaks = processed['PPG_Peaks'].sum()
duration = len(processed) / 64  # At 64 Hz
peak_rate = num_peaks / (duration / 60)  # Peaks per minute

print(f"Detected {num_peaks} peaks in {duration:.1f}s")
print(f"Heart rate: {peak_rate:.1f} BPM")
# Expected: 50-100 BPM at rest
```

---

### ParserWarning

**Warning**: `ParserWarning: Falling back to the 'python' engine because the 'c' engine does not support regex separators`

**Solution**: Already fixed in v0.1.0

If you still see this:
```python
# In bvp_loader.py or bvp_bids_writer.py
# Change:
data = pd.read_csv(file, sep='\\t')  # Wrong (escaped)

# To:
data = pd.read_csv(file, sep='\t')   # Correct (actual tab)
```

---

## Output Issues

### Permission Denied

**Problem**: `PermissionError: [Errno 13] Permission denied: 'data/derivatives/...'`

**Solutions**:

1. **Check directory permissions**:
```bash
ls -ld data/derivatives/
# Should be writable: drwxrwxr-x
```

2. **Fix permissions**:
```bash
chmod -R u+w data/derivatives/
```

3. **Check ownership**:
```bash
ls -l data/derivatives/
# Owner should be your user
sudo chown -R $USER:$USER data/derivatives/
```

---

### Existing Output Files

**Problem**: Processing fails because output files already exist

**Solution**:

1. **Clean outputs before reprocessing**:
```bash
# Clean specific subject/session
poetry run python scripts/clean_outputs.py --subject sub-f01p01 --session ses-01 --force

# Or clean all
poetry run python scripts/clean_outputs.py --all --force
```

2. **Manual cleanup**:
```bash
rm -rf data/derivatives/therasync-bvp/sub-f01p01/ses-01/
```

---

### Incomplete Outputs

**Problem**: Some output files missing after processing

**Diagnosis**:
```bash
# Check what was created
ls -1 data/derivatives/therasync-bvp/sub-f01p01/ses-01/physio/

# Expected files (per moment):
# - *_task-{moment}_desc-processed_recording-bvp.tsv
# - *_task-{moment}_desc-processed_recording-bvp.json
# - *_task-{moment}_desc-processing_recording-bvp.json

# Plus session-level:
# - *_desc-bvpmetrics_physio.tsv
# - *_desc-bvpmetrics_physio.json
# - *_desc-summary_recording-bvp.json
```

**Solutions**:

1. **Check logs**:
```bash
tail -n 50 log/bvp_preprocessing.log
# Look for errors during writing
```

2. **Verify processing completed**:
```python
# Check summary file
import json

with open('data/derivatives/.../desc-summary_recording-bvp.json') as f:
    summary = json.load(f)
    
print(summary['processing_complete'])  # Should be True
print(summary['moments_processed'])     # Should list all moments
```

3. **Re-run processing**:
```bash
# Clean and reprocess
poetry run python scripts/clean_outputs.py --subject sub-f01p01 --session ses-01 --force
poetry run python scripts/preprocess_bvp.py --subject sub-f01p01 --session ses-01 --verbose
```

---

### EDA Quality Warnings

#### Unusual SCR Rates

**Warning**: Subject has unusually high or low SCR rates compared to typical ranges

**Normal SCR Rates**:
- **Resting State**: 1-20 SCRs/min (typical), up to 30 during mild stress
- **Therapy Sessions**: 2-20 SCRs/min (variable based on emotional engagement)
- **High Arousal**: 20-40 SCRs/min (anxiety, excitement)
- **Very Low**: <5 SCRs/min (very calm, meditation)

**From Real Data Testing**:
- Family f01p01: 22-27 SCRs/min rest, 12.81-17.08 SCRs/min therapy
- Family f02p01: 11-21 SCRs/min rest, 2.24-7.42 SCRs/min therapy

**Interpretation**:

1. **High variability between subjects is NORMAL**:
   - Individual differences in autonomic reactivity
   - Different therapeutic contexts and engagement
   - Family f02p01 showing lower arousal is within normal range

2. **When to investigate**:
   - ❌ Exactly 0 SCRs: Poor electrode contact
   - ❌ >50 SCRs/min: Motion artifacts or noise
   - ❌ Sudden rate changes: Data quality issues
   - ✅ Consistent low rates (5-15/min): Calm subject
   - ✅ Moderate rates (15-25/min): Normal arousal
   - ✅ Variable rates across sessions: Expected

**Solutions**:

1. **Compare within-subject**:
```bash
# Check consistency across sessions for same subject
cat data/derivatives/therasync-eda/sub-f01p01/*/physio/*_desc-edametrics_physio.tsv
```

2. **Visualize SCR distribution**:
```python
import pandas as pd
import matplotlib.pyplot as plt

metrics = pd.read_csv('*_desc-edametrics_physio.tsv', sep='\t')
metrics['scr_count'].hist(bins=10)
plt.xlabel('Number of SCRs')
plt.ylabel('Frequency')
plt.title('SCR Distribution Across Moments')
plt.show()
```

3. **Check tonic levels for context**:
```python
# Low SCR rate with high tonic level suggests sustained arousal
# High SCR rate with low tonic suggests reactive arousal
metrics[['moment', 'scr_count', 'scr_per_min', 'tonic_mean']].sort_values('scr_per_min')
```

---

#### Atypical Tonic EDA Levels

**Warning**: Tonic EDA levels outside typical physiological range

**Normal Phasic Tonic Levels** (after cvxEDA decomposition):
- **Typical**: 0.01-0.5 μS (phasic component only)
- **Very Low**: <0.01 μS (minimal phasic activity)
- **Moderate**: 0.1-0.3 μS (normal arousal)
- **High**: 0.3-0.5 μS (elevated arousal)

**Note**: These are **phasic tonic** values, not absolute skin conductance levels (which are typically 1-20 μS).

**From Real Data Testing**:
- All values in range 0.002-0.476 μS: ✅ Physiologically reasonable
- Higher during therapy vs rest in most cases: ✅ Expected pattern

**Interpretation**:

1. **Very low values (<0.01 μS)**:
   - Poor electrode contact (check signal visually)
   - Very calm/relaxed subject
   - Possible preprocessing artifact

2. **Normal values (0.01-0.5 μS)**:
   - ✅ Proceed with analysis
   - Reflects phasic arousal component

3. **Extreme values (>1.0 μS in phasic component)**:
   - Check raw signal for artifacts
   - Verify cvxEDA decomposition worked correctly

**Solutions**:

1. **Visualize tonic component**:
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load processed signal
signal = pd.read_csv('*_desc-processed_recording-eda.tsv', sep='\t')

plt.figure(figsize=(12, 6))
plt.plot(signal['time'], signal['eda_tonic'], label='Tonic')
plt.plot(signal['time'], signal['eda_phasic'], label='Phasic', alpha=0.5)
plt.xlabel('Time (s)')
plt.ylabel('EDA (μS)')
plt.legend()
plt.title('Tonic and Phasic Components')
plt.show()
```

2. **Compare raw and processed**:
```python
# Check if decomposition worked
print(f"Raw mean: {signal['eda_raw'].mean():.3f} μS")
print(f"Clean mean: {signal['eda_clean'].mean():.3f} μS")
print(f"Tonic mean: {signal['eda_tonic'].mean():.3f} μS")
print(f"Phasic mean: {signal['eda_phasic'].mean():.3f} μS")
```

3. **Review electrode quality**:
```bash
# Check if low tonic levels are consistent across moments
grep tonic_mean data/derivatives/therasync-eda/sub-*/ses-*/physio/*_desc-edametrics_physio.tsv
```

---

## Performance Problems

### Slow Processing

**Problem**: Processing takes much longer than expected

**Expected Times** (per minute of data):
- Load: <0.1s
- Clean: ~0.5s
- Metrics: ~0.2s
- Write: <0.1s

**Diagnosis**:

1. **Enable verbose logging**:
```bash
poetry run python scripts/preprocess_bvp.py --subject sub-f01p01 --session ses-01 --verbose
# Check which step is slow
```

2. **Profile processing**:
```python
import time

start = time.time()
processed = cleaner.clean_signal(raw_data)
duration = time.time() - start

print(f"Processing took {duration:.2f}s for {len(raw_data)/64:.1f}s of data")
```

**Solutions**:

1. **Reduce metric extraction**:
```yaml
physio:
  bvp:
    metrics:
      extract_all: false  # Extract only selected metrics
      selected_metrics:
        time_domain: ["HRV_MeanNN", "HRV_RMSSD"]  # Fewer metrics
```

2. **Process moments separately**:
```bash
# Faster than all at once for very long recordings
poetry run python scripts/preprocess_bvp.py --subject sub-f01p01 --session ses-01 --moment restingstate
```

3. **Check system resources**:
```bash
htop  # Monitor CPU/memory usage
# Ensure no other heavy processes running
```

---

### High Memory Usage

**Problem**: System uses excessive memory during processing

**Solutions**:

1. **Process one moment at a time**:
```bash
poetry run python scripts/preprocess_bvp.py --subject sub-f01p01 --session ses-01 --moment therapy
```

2. **Close other applications**

3. **Monitor memory**:
```bash
watch -n 1 free -h
```

---

## Common Warnings Explained

### UserWarning: Mean of empty slice

**Warning**: `RuntimeWarning: Mean of empty slice`

**Cause**: No valid quality scores in signal segment

**Impact**: Usually harmless, metrics still computed

**Solution**: If frequent, review signal quality

---

### FutureWarning: DataFrame.applymap

**Warning**: `FutureWarning: DataFrame.applymap has been deprecated`

**Cause**: pandas version difference in NeuroKit2

**Impact**: None (functionality works)

**Solution**: Ignore or update NeuroKit2:
```bash
poetry update neurokit2
```

---

### DeprecationWarning: np.find_common_type

**Warning**: `DeprecationWarning: np.find_common_type is deprecated`

**Cause**: NumPy/SciPy compatibility

**Impact**: None (functionality works)

**Solution**: Update dependencies:
```bash
poetry update numpy scipy
```

---

## Getting Help

### Enable Debug Logging

```bash
# Maximum verbosity
poetry run python scripts/preprocess_bvp.py --subject sub-f01p01 --session ses-01 --verbose

# Check full log
cat log/bvp_preprocessing.log
```

### Collect Diagnostic Information

```bash
# System info
python --version
poetry --version
poetry show  # List installed packages

# Data structure
tree data/raw/sub-f01p01/ -L 3

# Config
cat config/config.yaml

# Recent logs
tail -n 100 log/bvp_preprocessing.log
```

### Report Issues

When reporting issues, include:

1. **Error message** (full traceback)
2. **Command used**
3. **Data structure** (tree output)
4. **Config file** (if custom)
5. **Log file** (last 50 lines)
6. **System info** (Python version, OS)

**GitHub Issues**: https://github.com/Ramdam17/TherasyncAnalysis/issues

---

## Quick Reference

### Most Common Issues

| Issue | Quick Fix |
|-------|-----------|
| Import errors | `PYTHONPATH=. poetry run ...` |
| File not found | Check `paths.sourcedata` in config |
| Low BVP quality warning | Normal for long recordings if >0.65 |
| Few/many SCRs detected | Check electrode contact, adjust threshold |
| cvxEDA convergence error | Check signal for NaN/inf values |
| Low sampling rate warning | Expected for E4 (4 Hz), no action needed |
| Memory error | Process moments separately |
| Permission denied | `chmod -R u+w data/derivatives/` |
| Existing outputs | `poetry run python scripts/clean_outputs.py --all --force` |

### Useful Commands

#### BVP Pipeline
```bash
# Clean and reprocess BVP
poetry run python scripts/clean_outputs.py --derivatives --force
poetry run python scripts/preprocess_bvp.py --subject sub-f01p01 --session ses-01 --verbose

# Check BVP outputs
tree data/derivatives/therasync-bvp/sub-f01p01/ses-01/

# View BVP metrics
column -t -s $'\t' data/derivatives/therasync-bvp/sub-f01p01/ses-01/physio/*bvpmetrics*.tsv

# Check BVP logs
tail -f log/bvp_preprocessing.log
```

#### EDA Pipeline
```bash
# Clean and reprocess EDA
poetry run python scripts/clean_outputs.py --derivatives --force
PYTHONPATH=. poetry run python scripts/preprocess_eda.py --subject sub-f01p01 --session ses-01 --verbose

# Check EDA outputs
tree data/derivatives/therasync-eda/sub-f01p01/ses-01/
find data/derivatives/therasync-eda -name "*metrics*" | sort

# View EDA metrics (all subjects)
for file in data/derivatives/therasync-eda/*/*/physio/*_desc-edametrics_physio.tsv; do
  echo "=== $(basename $file) ===";
  cat "$file" | column -t -s $'\t';
done

# Check single subject EDA metrics
column -t -s $'\t' data/derivatives/therasync-eda/sub-f01p01/ses-01/physio/*edametrics*.tsv

# View SCR events
head -20 data/derivatives/therasync-eda/sub-f01p01/ses-01/physio/*_desc-scr_events.tsv

# Check processing summary
cat data/derivatives/therasync-eda/sub-f01p01/ses-01/physio/*_desc-summary_recording-eda.json | jq
```

---

**Authors**: Lena Adel, Remy Ramadour  
**Last Updated**: October 28, 2025  
**Version**: 0.1.0
