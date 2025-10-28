# Therasync Pipeline - Available Resources Inventory

This document tracks all available resources (data files, images, configs, metadata) that can be used for building interactive viewers, dashboards, and visualizations.

**Authors**: Lena Adel, Remy Ramadour  
**Last Updated**: October 27, 2025  
**Pipeline Version**: 0.1.0  
**Status**: Sprint 2 - BVP Processing Complete

---

## 📊 Table of Contents

1. [Configuration Files](#configuration-files)
2. [Raw Data](#raw-data)
3. [Processed Data (Derivatives)](#processed-data-derivatives)
4. [Metadata & Descriptors](#metadata--descriptors)
5. [Logs](#logs)
6. [Documentation](#documentation)
7. [Future Resources](#future-resources)

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
data/raw/
├── sourcedata/          # Family session recordings (shared data)
│   └── sub-f*shared/
│       └── ses-*/
│           ├── moi_tables/      # Moment of Interest timing tables
│           ├── transcripts/     # Session transcripts
│           └── video/           # Video recordings
│
└── sub-f*p*/           # Individual participant data
    └── ses-*/
        └── physio/
            ├── *_recording-bvp.{tsv,json}   # Blood Volume Pulse
            ├── *_recording-eda.{tsv,json}   # Electrodermal Activity
            ├── *_recording-hr.{tsv,json}    # Heart Rate
            ├── *_recording-temp.{tsv,json}  # Temperature
            └── *_recording-acc.{tsv,json}   # Accelerometer
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
- **Status**: ⏳ Pipeline not yet implemented (Sprint 3)
- **Future Use**: Skin conductance visualization, arousal levels

#### HR (Heart Rate)
- **Files**: `*_task-{moment}_recording-hr.tsv` + `.json`
- **Sampling Rate**: 1 Hz (derived from BVP)
- **Status**: ⏳ Available but not processed yet
- **Future Use**: Real-time heart rate display, trends

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

### BVP Pipeline Outputs

#### Dataset-Level
- **Path**: `data/derivatives/therasync-bvp/`
- **File**: `dataset_description.json`
- **Format**: JSON
- **Contains**:
  - Dataset name and version
  - BIDS version
  - Generation information
  - Pipeline metadata
- **Viewer Use**: Pipeline provenance, version tracking

#### Subject/Session Level
- **Path**: `data/derivatives/therasync-bvp/{subject}/{session}/physio/`

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

##### 4. Session Summary
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

## Future Resources

### Planned for Sprint 3+ (EDA Processing)
- ⏳ Processed EDA signals (phasic, tonic components)
- ⏳ EDA metrics (SCR frequency, amplitude, rise time)
- ⏳ Arousal indicators

### Planned for Future Sprints
- ⏳ Epoched analysis (30-second windows with 1-second steps)
- ⏳ Dynamic HRV metrics over time
- ⏳ Cross-signal correlations (BVP-EDA, BVP-HR)
- ⏳ Family synchrony metrics
- ⏳ Visualization plots (PNG/SVG exports)
- ⏳ Statistical analysis results
- ⏳ Group comparisons
- ⏳ Moment of Interest (MOI) aligned data

### Planned Visualizations (to be generated)
- ⏳ Time-series plots (raw and processed signals)
- ⏳ HRV distribution plots
- ⏳ Spectral density plots (frequency domain)
- ⏳ Poincaré plots (nonlinear analysis)
- ⏳ Quality control plots
- ⏳ Heatmaps (metrics across subjects/sessions)
- ⏳ Correlation matrices

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
| Raw Physiological Signals | 5 types | `data/raw/sub-*/ses-*/physio/` | ✅ Available |
| Processed BVP Signals | 2 moments | `data/derivatives/therasync-bvp/` | ✅ Available |
| HRV Metrics | 20 metrics | `data/derivatives/therasync-bvp/` | ✅ Available |
| Processing Metadata | Per moment | `data/derivatives/therasync-bvp/` | ✅ Available |
| Logs | Multiple | `log/` | ✅ Available |
| Documentation | 6 files | `docs/` | ✅ Available |

### By Viewer Component

| Viewer Component | Required Resources | Status |
|-----------------|-------------------|--------|
| Session Browser | Subject/session lists, summaries | ✅ Ready |
| Signal Viewer | Processed signals TSV, processing JSON | ✅ Ready |
| Metrics Dashboard | Metrics TSV, metadata JSON | ✅ Ready |
| Quality Monitor | Processing JSONs, summary JSON, logs | ✅ Ready |
| Moment Comparison | Metrics TSV filtered by moment | ✅ Ready |
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

### Priority Visualizations (Phase 1)
1. ✅ Session list/browser
2. ✅ Key metrics dashboard (cards)
3. ✅ Time-series signal viewer
4. ✅ Moment comparison (bar charts)
5. ⏳ Quality control dashboard

---

**This document will be updated as new pipelines are added and new resources become available.**
