# Therasync Pipeline - Available Resources Inventory

This document tracks all available resources (data files, images, configs, metadata) that can be used for building interactive viewers, dashboards, and visualizations.

**Authors**: Lena Adel, Remy Ramadour  
**Last Updated**: November 11, 2025  
**Pipeline Version**: v1.0.0 (Production Ready)  
**Status**: All preprocessing pipelines complete (BVP, EDA, HR) with harmonized architecture and visualization suite

---

## 📊 Table of Contents

1. [Configuration Files](#configuration-files)
2. [Raw Data](#raw-data)
3. [Processed Data (Derivatives)](#processed-data-derivatives)
4. [Epoched Data](#epoched-data)
5. [Metadata & Descriptors](#metadata--descriptors)
6. [Logs](#logs)
7. [Documentation](#documentation)
8. [Future Resources](#future-resources)

---

## Configuration Files

### Main Configuration
- **Path**: `config/config.yaml`
- **Format**: YAML
- **Purpose**: Main pipeline configuration (paths, processing parameters, metrics selection)
- **Key Sections**:
  - Study metadata
  - Data paths
  - Moment/task definitions
  - BVP processing parameters
  - Selected HRV metrics
  - BIDS configuration
  - Logging settings
- **Viewer Use**: Display pipeline settings, show which metrics were computed

### Example Configuration
- **Path**: `config/example_config.yaml`
- **Format**: YAML
- **Purpose**: Documented example with all available options
- **Viewer Use**: Reference documentation, configuration builder UI

---

## Raw Data

### Data Structure
```
data/
├── sourcedata/          # Raw BIDS data
│   ├── sub-f*shared/    # Family session recordings (shared data)
│   │   └── ses-*/
│   │       ├── moi_tables/      # Moment of Interest timing tables
│   │       ├── transcripts/     # Session transcripts
│   │       └── video/           # Video recordings
│   │
│   └── sub-f*p*/        # Individual participant data
│       └── ses-*/
│           └── physio/
│               ├── *_recording-bvp.{tsv,json}   # Blood Volume Pulse
│               ├── *_recording-eda.{tsv,json}   # Electrodermal Activity
│               ├── *_recording-hr.{tsv,json}    # Heart Rate
│               ├── *_recording-temp.{tsv,json}  # Temperature
│               └── *_recording-acc.{tsv,json}   # Accelerometer
│
└── derivatives/         # Processed outputs
    ├── preprocessing/
    │   └── sub-{subject}/
    │       └── ses-{session}/
    │           ├── bvp/         # BVP preprocessing outputs (13 files)
    │           ├── eda/         # EDA preprocessing outputs (13 files)
    │           └── hr/          # HR preprocessing outputs (14 files)
    │
    └── visualizations/  # Visualization suite (6 plots per session)
```

### Available Physiological Signals (Raw)

#### BVP (Blood Volume Pulse)
- **Files**: `*_task-{moment}_recording-bvp.tsv` + `.json`
- **Sampling Rate**: 64 Hz
- **Columns**: `time`, `bvp`
- **Units**: seconds, arbitrary units (AU)
- **Moments**: `restingstate`, `therapy`
- **Viewer Use**: 
  - Time-series plots of raw BVP signal
  - Quality indicators overlay
  - Detected peaks visualization

#### EDA (Electrodermal Activity)
- **Files**: `*_task-{moment}_recording-eda.tsv` + `.json`
- **Sampling Rate**: 4 Hz
- **Status**: ✅ Pipeline complete (Phase 2)
- **Viewer Use**: Skin conductance visualization, arousal levels, phasic/tonic components

#### HR (Heart Rate)
- **Files**: `*_task-{moment}_recording-hr.tsv` + `.json`
- **Sampling Rate**: 1 Hz (derived from BVP)
- **Status**: ✅ Pipeline complete (Phase 2)
- **Viewer Use**: Real-time heart rate display, trends, per-moment analysis

#### Temperature
- **Files**: `*_task-{moment}_recording-temp.tsv` + `.json`
- **Sampling Rate**: 4 Hz
- **Status**: ⏳ Not yet processed
- **Future Use**: Temperature trends, correlations

#### Accelerometer
- **Files**: `*_task-{moment}_recording-acc.tsv` + `.json`
- **Sampling Rate**: 32 Hz
- **Status**: ⏳ Not yet processed
- **Future Use**: Movement detection, activity levels

### Moment/Session Metadata
- **Files**: `*_recording-*.json` (JSON sidecars for each recording)
- **Contains**:
  - `SamplingFrequency`: Sampling rate in Hz
  - `StartTime`: Recording start time
  - `Columns`: Column names in TSV
  - `Units`: Units for each column
  - `TaskName`: Moment/task identifier
  - `RecordingType`: Signal type
  - `AcquisitionTime`: Unix timestamp
  - `FamilyID`: Family identifier
- **Viewer Use**: Display session info, timing synchronization

---

## Processed Data (Derivatives)

### Preprocessing Pipeline Outputs

All three modalities (BVP, EDA, HR) follow harmonized architecture with consistent file patterns and naming conventions.

#### Dataset-Level
- **Paths**: 
  - `data/derivatives/therasync-bvp/`
  - `data/derivatives/therasync-eda/`
  - `data/derivatives/therasync-hr/`
- **File**: `dataset_description.json`
- **Format**: JSON
- **Contains**:
  - Dataset name and version
  - BIDS version
  - Generation information
  - Pipeline metadata
- **Viewer Use**: Pipeline provenance, version tracking

#### Subject/Session Level
- **Paths**: `data/derivatives/therasync-{modality}/{subject}/{session}/physio/`

##### 1. Processed Signals
- **Files**: `*_task-{moment}_desc-processed_recording-bvp.{tsv,json}`
- **Format**: TSV + JSON sidecar
- **Columns**:
  - `time`: Time in seconds
  - `PPG_Clean`: Cleaned BVP signal
  - `PPG_Rate`: Instantaneous heart rate (BPM)
- **Viewer Use**:
  - **Primary**: Time-series visualization of cleaned signals
  - **Secondary**: Compare raw vs. processed signals
  - **Interactive**: Zoom, pan, select regions
  - **Overlay**: Peak detection markers, quality indicators

##### 2. Processing Metadata
- **Files**: `*_task-{moment}_desc-processing_recording-bvp.json`
- **Format**: JSON
- **Contains**:
  - Processing method and parameters
  - Peak detection results (indices, count)
  - Quality metrics (mean, std)
  - Signal statistics (amplitude, range)
  - Processing timestamp
- **Viewer Use**:
  - Quality control dashboard
  - Processing parameters display
  - Signal quality indicators
  - Peak detection statistics

##### 3. HRV Metrics
- **Files**: `*_desc-bvpmetrics_physio.{tsv,json}`
- **Format**: TSV + JSON sidecar
- **Columns** (20 metrics per moment):
  - **Identifiers**: `moment`
  
  - **Time-Domain HRV (5 metrics)**:
    - `HRV_MeanNN`: Mean RR interval (ms) - Average heart period
    - `HRV_SDNN`: Standard deviation of RR intervals (ms) - Overall HRV
    - `HRV_RMSSD`: Root mean square of successive differences (ms) - Parasympathetic activity
    - `HRV_pNN50`: Percentage of RR intervals >50ms different (%) - Parasympathetic tone
    - `HRV_CVNN`: Coefficient of variation - Normalized variability
  
  - **Frequency-Domain HRV (4 metrics)**:
    - `HRV_LF`: Low frequency power (ms²) - Sympathetic + parasympathetic
    - `HRV_HF`: High frequency power (ms²) - Parasympathetic (RSA)
    - `HRV_LFHF`: LF/HF ratio - Autonomic balance
    - `HRV_TP`: Total power (ms²) - Overall autonomic activity
  
  - **Nonlinear HRV (3 metrics)**:
    - `HRV_SD1`: Poincaré plot SD1 (ms) - Short-term variability
    - `HRV_SD2`: Poincaré plot SD2 (ms) - Long-term variability
    - `HRV_SampEn`: Sample entropy - Signal complexity/regularity
  
  - **Signal Quality Metrics (8 metrics)**:
    - `BVP_NumPeaks`: Number of detected peaks
    - `BVP_Duration`: Signal duration (seconds)
    - `BVP_PeakRate`: Detected peak rate (peaks/second)
    - `BVP_MeanQuality`: Mean signal quality (0-1)
    - `BVP_QualityStd`: Quality standard deviation
    - `BVP_MeanAmplitude`: Mean peak amplitude
    - `BVP_StdAmplitude`: Amplitude standard deviation
    - `BVP_RangeAmplitude`: Amplitude range (max-min)

- **Viewer Use**:
  - **Dashboards**: Key metrics display (cards, gauges)
  - **Comparisons**: Moment-to-moment comparisons (restingstate vs. therapy)
  - **Time-Domain**: Bar charts, trend lines
  - **Frequency-Domain**: Spectral visualizations, pie charts (LF/HF ratio)
  - **Nonlinear**: Poincaré plots, entropy displays
  - **Quality**: Quality indicators, thresholds, warnings
  - **Tables**: Sortable, filterable metric tables
  - **Export**: CSV/JSON for further analysis

##### 4. RR Intervals
- **Files**: `*_task-{moment}_desc-rrintervals_physio.{tsv,json}`
- **Format**: TSV + JSON sidecar (2 files per moment, 4 files per session)
- **Purpose**: Peak-to-peak intervals from BVP signal for detailed HRV analysis
- **Columns** (4 per row):
  - `time_peak_start`: Timestamp of interval start peak (seconds)
  - `time_peak_end`: Timestamp of interval end peak (seconds)
  - `rr_interval_ms`: Duration between peaks in milliseconds
  - `is_valid`: Validity flag (1 = valid, 0 = invalid based on physiological thresholds)

- **Validation**:
  - **Valid Range**: 300-2000 ms (30-200 BPM)
  - **Configurable**: Thresholds defined in `config.yaml`
  - **Preservation**: All intervals saved (invalid marked with `is_valid=0`)
  - **Typical Valid Rate**: 95-98% for clean signals

- **Viewer Use**:
  - **Time-Series**: Plot RR intervals over time (tachogram)
  - **Poincaré Plots**: RR(n) vs. RR(n+1) scatter plots
  - **Histograms**: Distribution of RR intervals
  - **Quality Filtering**: Toggle display of invalid intervals
  - **Advanced HRV**: Custom time-domain and frequency-domain analysis
  - **Export**: Filtered intervals for external HRV tools

##### 5. Session Summary
- **Files**: `*_desc-summary_recording-bvp.json`
- **Format**: JSON
- **Contains**:
  - Processing summary for all moments
  - Success/failure status per moment
  - Aggregate statistics
  - Processing timestamp
- **Viewer Use**:
  - Session overview dashboard
  - Quality control summary
  - Processing status indicators

---

### Epoched Data

#### Directory Structure
- **Path**: `data/derivatives/epoched/`
- **Organization**: Same as preprocessing (BIDS-compliant)
```
epoched/
└── sub-{subject}/
    └── ses-{session}/
        ├── bvp/   # 4 files (2 therapy + 2 restingstate)
        ├── eda/   # 4 files (2 therapy + 2 restingstate)
        └── hr/    # 4 files (2 therapy + 2 restingstate)
```

#### Epoched Files
- **Files**: Same as preprocessed files with added epoch columns
- **Format**: TSV + JSON sidecar
- **Total**: 401 files across 51 sessions (8 files per session)
- **File Types**:
  - RR intervals: `*_desc-rrintervals_physio.{tsv,json}`
  - Processed recordings: `*_desc-processed_recording-{modality}.{tsv,json}`

#### Epoch Columns

All epoched files contain 3 additional columns with JSON-formatted epoch ID lists:

##### 1. Fixed Windows with Overlap
- **Column Name**: `epoch_fixed_duration{duration}s_overlap{overlap}s`
- **Example**: `epoch_fixed_duration30s_overlap5s`
- **Method**: Non-overlapping windows (duration=30s, step=25s due to 5s overlap)
- **Sample Assignment**: Samples can belong to 1-2 epochs
- **Format**: JSON list (e.g., `"[0]"`, `"[0, 1]"`, `"[1, 2]"`)
- **Use Cases**:
  - DPPA (Dynamic Pupillometric Physiological Analysis)
  - Time-window analysis with context preservation
  - Averaging over overlapping windows

##### 2. N-Split (Equal Division)
- **Column Name**: `epoch_nsplit{n_epochs}`
- **Example**: `epoch_nsplit120`
- **Method**: Signal divided into N equal-duration epochs (no overlap)
- **Sample Assignment**: Each sample belongs to exactly 1 epoch
- **Format**: JSON list with single ID (e.g., `"[0]"`, `"[1]"`, `"[119]"`)
- **Use Cases**:
  - Equal-duration comparisons
  - Temporal progression analysis
  - Statistical analyses requiring independent windows

##### 3. Sliding Windows
- **Column Name**: `epoch_sliding_duration{duration}s_step{step}s`
- **Example**: `epoch_sliding_duration30s_step5s`
- **Method**: Overlapping windows (duration=30s, step=5s)
- **Sample Assignment**: Samples can belong to many epochs (typically 6 with step=5s)
- **Format**: JSON list (e.g., `"[0, 1, 2, 3, 4, 5]"`)
- **Configuration**: step=5s reduces epoch gaps to 3.9%
- **Use Cases**:
  - Fine-grained temporal analysis
  - Moving averages
  - Event-triggered epoch selection
  - Robustness to epoch boundary effects

#### Special Case: Restingstate
- **Rule**: All restingstate samples assigned to epoch `[0]` regardless of method
- **Rationale**: Restingstate is a baseline reference without temporal subdivisions
- **Format**: Always `"[0]"` for all three epoch columns

#### Epoch ID Format

**Consistent JSON List Format**:
- All epoch columns use JSON-formatted lists: `"[0]"`, `"[0, 1, 2]"`
- Type: Always `str` (object dtype in pandas)
- Parsing: `ast.literal_eval(value)` → Python list

**Example Data**:
```tsv
time    epoch_fixed_duration30s_overlap5s    epoch_nsplit120    epoch_sliding_duration30s_step5s
0.77    [0]                                   [0]                [0]
25.45   [0, 1]                                [0]                [0, 1, 2, 3, 4, 5]
50.06   [1, 2]                                [1]                [2, 3, 4, 5, 6, 7, 8, 9, 10]
```

**Parsing Example**:
```python
import pandas as pd
import ast

df = pd.read_csv('epoched_file.tsv', sep='\t')
# Parse epoch IDs from JSON list format
epoch_ids = ast.literal_eval(df['epoch_sliding_duration30s_step5s'].iloc[0])
# Result: [0, 1, 2, 3, 4, 5] (Python list of integers)
```

#### Viewer Use Cases

- **Epoch Selection**:
  - Filter data by epoch ID
  - Select epochs around events
  - Compare epochs across sessions

- **Epoch Aggregation**:
  - Average metrics per epoch
  - Temporal trends across epochs
  - Epoch-based statistics

- **Epoch Shuffling**:
  - Randomize epoch order for statistical analysis
  - Bootstrap resampling within epochs
  - Cross-validation with epoch stratification

- **Visualization**:
  - Color-code samples by epoch
  - Plot epoch boundaries
  - Heatmaps of epoch-based metrics
  - Time-series with epoch overlays

- **Quality Control**:
  - Identify epochs with missing data (gap analysis)
  - Detect epochs with poor signal quality
  - Epoch-level quality flags

#### Configuration
- **File**: `config/config.yaml`
- **Section**: `epoching`
- **Customizable**:
  - Fixed window duration and overlap
  - N-split epoch count
  - Sliding window duration and step
  - Enable/disable individual methods
  - File patterns to epoch (include/exclude)

---

## Metadata & Descriptors

### BIDS Metadata
All derivative files include JSON sidecars with:
- Column descriptions
- Units of measurement
- Processing methods
- Quality metrics
- Temporal information
- Provenance

**Viewer Use**: 
- Tooltips and help text
- Unit displays
- Method documentation
- Quality thresholds

### Processing Provenance
- **Where**: Embedded in JSON sidecars
- **Contains**:
  - Software version
  - Processing date/time
  - Parameters used
  - Input file references
- **Viewer Use**: 
  - Reproducibility information
  - Processing timeline
  - Version tracking

---

## Logs

### Location
- **Path**: `log/`
- **Files**: 
  - `bvp_preprocessing.log`: Latest BVP processing log
  - `therasyncpipeline.log`: General pipeline log
  - Rotated logs with timestamps

### Content
- Processing steps and timing
- Warnings and errors
- Quality control information
- File I/O operations
- Configuration loaded

### Viewer Use
- **Debugging**: Error displays, warnings
- **Monitoring**: Processing progress, real-time updates
- **Audit**: Processing history, timeline
- **Quality**: Quality warnings visualization

---

## Documentation

### Available Documentation

#### User Documentation
1. **README.md**: Project overview and quick start
2. **QUICKREF.md**: Quick reference card for common commands
3. **docs/testing_guide.md**: Comprehensive testing workflows
4. **docs/sprint_2_summary.md**: Sprint 2 completion summary
5. **docs/bvp_preprocessing_research.md**: BVP processing research notes
6. **docs/bvp_metrics_research.md**: HRV metrics selection rationale

#### Technical Documentation
- **pyproject.toml**: Dependencies and project metadata
- **config/example_config.yaml**: Fully documented configuration

### Viewer Use
- **Help System**: Contextual help, tooltips
- **Method Descriptions**: Algorithm explanations
- **Metric Definitions**: What each HRV metric means
- **Tutorials**: Interactive walkthroughs

---

## Visualization Suite

### Available Visualizations (v1.0.0)

All visualizations are automatically generated during preprocessing and saved as PNG files in `data/derivatives/visualizations/{subject}/{session}/`.

#### 1. Signal Quality Plots (2 plots per session)
- **Files**: `*_task-{moment}_desc-quality_recording-bvp.png`
- **Moments**: `restingstate`, `therapy`
- **Contains**:
  - Raw BVP signal
  - Cleaned signal overlay
  - Quality indicators
  - Detected peaks markers
- **Viewer Use**: Quality assessment, signal validation, preprocessing verification

#### 2. HRV Metrics Comparison (1 plot per session)
- **File**: `*_desc-hrvcomparison_physio.png`
- **Contains**:
  - Time-domain metrics (MeanNN, SDNN, RMSSD, pNN50)
  - Frequency-domain metrics (LF, HF, LF/HF ratio)
  - Nonlinear metrics (SD1, SD2, SampEn)
  - Moment-to-moment comparison (restingstate vs. therapy)
- **Viewer Use**: Physiological state comparison, autonomic balance assessment

#### 3. EDA Signal Plots (2 plots per session)
- **Files**: `*_task-{moment}_desc-quality_recording-eda.png`
- **Moments**: `restingstate`, `therapy`
- **Contains**:
  - Raw EDA signal
  - Cleaned signal
  - Phasic component
  - Tonic component
  - SCR peaks markers
- **Viewer Use**: Arousal assessment, skin conductance analysis

#### 4. Heart Rate Plots (1 plot per session)
- **File**: `*_desc-hrcomparison_physio.png`
- **Contains**:
  - Per-moment HR statistics
  - Mean HR comparison (restingstate vs. therapy)
  - HR variability indicators
- **Viewer Use**: Heart rate trends, moment comparison

**Total**: 6 visualizations per session
**Production Status**: 306/306 plots generated successfully (100% success rate)

---

## Future Resources

### Planned for Future Versions
- ⏳ Epoched analysis (30-second windows with 1-second steps)
- ⏳ Dynamic HRV metrics over time
- ⏳ Cross-signal correlations (BVP-EDA, BVP-HR)
- ⏳ Family synchrony metrics
- ⏳ Interactive visualizations (HTML/JavaScript)
- ⏳ Statistical analysis results
- ⏳ Group comparisons
- ⏳ Moment of Interest (MOI) aligned data

### Additional Planned Visualizations
- ⏳ Spectral density plots (frequency domain)
- ⏳ Poincaré plots (nonlinear analysis)
- ⏳ Heatmaps (metrics across subjects/sessions)
- ⏳ Correlation matrices
- ⏳ Real-time streaming dashboards

---

## Data Access Patterns for Viewer

### Example Access Patterns

#### 1. Load Session Overview
```javascript
// Get session list
const sessions = await fetch('data/derivatives/therasync-bvp/subjects.json');

// Get specific session metrics
const metrics = await fetch('data/derivatives/therasync-bvp/sub-f01p01/ses-01/physio/sub-f01p01_ses-01_desc-bvpmetrics_physio.tsv');
const metadata = await fetch('data/derivatives/therasync-bvp/sub-f01p01/ses-01/physio/sub-f01p01_ses-01_desc-bvpmetrics_physio.json');
```

#### 2. Display Time-Series
```javascript
// Load processed signal
const signal = await d3.tsv('data/derivatives/therasync-bvp/sub-f01p01/ses-01/physio/sub-f01p01_ses-01_task-restingstate_desc-processed_recording-bvp.tsv');

// Load processing info for peak markers
const processing = await fetch('data/derivatives/therasync-bvp/sub-f01p01/ses-01/physio/sub-f01p01_ses-01_task-restingstate_desc-processing_recording-bvp.json');
```

#### 3. Compare Moments
```javascript
// Load metrics for both moments
const metrics = await d3.tsv('data/derivatives/therasync-bvp/sub-f01p01/ses-01/physio/sub-f01p01_ses-01_desc-bvpmetrics_physio.tsv');

// Filter by moment
const restingstate = metrics.filter(d => d.moment === 'restingstate');
const therapy = metrics.filter(d => d.moment === 'therapy');
```

#### 4. Quality Dashboard
```javascript
// Load all processing metadata
const sessions = ['sub-f01p01/ses-01', 'sub-f01p01/ses-02'];
const quality = await Promise.all(
  sessions.map(s => fetch(`data/derivatives/therasync-bvp/${s}/physio/*_desc-summary_recording-bvp.json`))
);
```

---

## Resource Organization Summary

### By Data Type

| Type | Count | Location | Status |
|------|-------|----------|--------|
| Configuration | 2 | `config/` | ✅ Available |
| Raw Physiological Signals | 5 types | `data/sourcedata/sub-*/ses-*/physio/` | ✅ Available |
| Processed Signals (BVP/EDA/HR) | 3 modalities | `data/derivatives/therasync-*/` | ✅ Available |
| HRV Metrics | 20 metrics | `data/derivatives/therasync-bvp/` | ✅ Available |
| EDA Metrics | 12 metrics | `data/derivatives/therasync-eda/` | ✅ Available |
| HR Metrics | Per-moment stats | `data/derivatives/therasync-hr/` | ✅ Available |
| Visualizations | 6 per session | `data/derivatives/visualizations/` | ✅ Available |
| Processing Metadata | Per moment | `data/derivatives/therasync-*/` | ✅ Available |
| Logs | Multiple | `log/` | ✅ Available |
| Documentation | 10 files | `docs/` + root | ✅ Available |

### By Viewer Component

| Viewer Component | Required Resources | Status |
|-----------------|-------------------|--------|
| Session Browser | Subject/session lists, summaries | ✅ Ready |
| Signal Viewer | Processed signals TSV, processing JSON | ✅ Ready |
| Metrics Dashboard | Metrics TSV, metadata JSON | ✅ Ready |
| Quality Monitor | Processing JSONs, summary JSON, logs | ✅ Ready |
| Moment Comparison | Metrics TSV filtered by moment | ✅ Ready |
| Visualization Gallery | PNG files in derivatives/visualizations/ | ✅ Ready |
| Configuration Display | config.yaml | ✅ Ready |
| Documentation Help | docs/*.md | ✅ Ready |

---

## Notes for Viewer Development

### Recommended Libraries
- **D3.js**: Time-series plots, interactive visualizations
- **Plotly.js**: Complex charts, 3D plots, heatmaps
- **Chart.js**: Simple charts, gauges, dashboards
- **DataTables**: Sortable/filterable metric tables
- **PapaParse**: TSV/CSV parsing

### Design Considerations
1. **Performance**: Large signals (~180k samples for therapy) - use downsampling/windowing
2. **Responsiveness**: Mobile-friendly layouts
3. **Interactivity**: Zoom, pan, select, filter
4. **Export**: Allow CSV/JSON/PNG exports
5. **Real-time**: Support streaming if needed (future)

### Priority Visualizations (v1.0.0)
1. ✅ Session list/browser
2. ✅ Key metrics dashboard (cards)
3. ✅ Time-series signal viewer
4. ✅ Moment comparison (bar charts)
5. ✅ Quality control dashboard
6. ✅ Automated visualization generation (PNG exports)

---

**This document will be updated as new pipelines are added and new resources become available.**
