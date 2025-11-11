# API Reference - TherasyncPipeline

**Authors**: Lena Adel, Remy Ramadour  
**Version**: 1.0.0 (Production Ready - Phase 2 Harmonization Complete)  
**Last Updated**: November 11, 2025

This document provides comprehensive API documentation for all modules in the TherasyncPipeline project.

**Note**: All BIDS writers (BVP, EDA, HR) have been harmonized in Phase 2 to use identical code patterns and helper method signatures for consistency and maintainability.

---

## Table of Contents

1. [Core Modules](#core-modules)
   - [ConfigLoader](#configloader)
2. [Preprocessing Modules](#preprocessing-modules)
   - **BVP (Blood Volume Pulse)**
     - [BVPLoader](#bvploader)
     - [BVPCleaner](#bvpcleaner)
     - [BVPMetrics](#bvpmetrics)
     - [BVPBIDSWriter](#bvpbidswriter)
   - **EDA (Electrodermal Activity)**
     - [EDALoader](#edaloader)
     - [EDACleaner](#edacleaner)
     - [EDAMetricsExtractor](#edametricsextractor)
     - [EDABIDSWriter](#edabidswriter)
   - **HR (Heart Rate)**
     - [HRLoader](#hrloader)
     - [HRCleaner](#hrcleaner)
     - [HRMetrics](#hrmetrics)
     - [HRBIDSWriter](#hrbidswriter)
3. [Scripts](#scripts)
   - [preprocess_bvp.py](#preprocess_bvppy)
   - [preprocess_eda.py](#preprocess_edapy)
   - [preprocess_hr.py](#preprocess_hrpy)
   - [generate_visualizations.py](#generate_visualizationspy)
   - [run_all_preprocessing.py](#run_all_preprocessingpy)
   - [run_all_visualizations.py](#run_all_visualizationspy)
   - [clean_outputs.py](#clean_outputspy)
4. [Visualization Modules](#visualization-modules)
   - [VisualizationDataLoader](#visualizationdataloader)
   - [VisualizationConfig](#visualizationconfig)
   - [SignalPlotter](#signalplotter)
   - [HRVPlotter](#hrvplotter)
   - [EDAPlotter](#edaplotter)
5. [Batch Processing](#batch-processing)
   - [BatchPreprocessor](#batchpreprocessor)
   - [BatchVisualizer](#batchvisualizer)

---

## Core Modules

### ConfigLoader

**Module**: `src.core.config_loader.py`

Central configuration management system with YAML loading, validation, and dot-notation access.

#### Class: `ConfigLoader`

```python
from src.core.config_loader import ConfigLoader

config = ConfigLoader(config_path='config/config.yaml')
```

**Constructor Parameters**:
- `config_path` (str, optional): Path to YAML configuration file. Default: `'config/config.yaml'`

**Attributes**:
- `config_path` (Path): Path object to configuration file
- `config` (dict): Loaded configuration dictionary

**Methods**:

##### `load_config() -> dict`
Loads and validates YAML configuration file.

```python
config_dict = config.load_config()
```

**Returns**: Dictionary containing all configuration settings

**Raises**:
- `FileNotFoundError`: If config file doesn't exist
- `yaml.YAMLError`: If YAML parsing fails
- `jsonschema.ValidationError`: If config doesn't match schema

**Schema Requirements**:
```yaml
paths:
  sourcedata: str  # Path to raw data
  derivatives: str  # Path to processed outputs
  logs: str        # Path to log files
study:
  name: str
  version: str
moments: list[dict]  # Task/moment definitions
physio: dict         # Processing parameters
```

##### `get(key: str, default: Any = None) -> Any`
Retrieves configuration value using dot notation.

```python
# Access nested values
sampling_rate = config.get('physio.bvp.sampling_rate')  # Returns 64
method = config.get('physio.bvp.processing.method')     # Returns 'elgendi'
missing = config.get('nonexistent.key', default=None)   # Returns None
```

**Parameters**:
- `key` (str): Dot-separated path to config value (e.g., `'physio.bvp.sampling_rate'`)
- `default` (Any): Value to return if key not found. Default: `None`

**Returns**: Configuration value at specified path, or default if not found

##### `validate_config(config_dict: dict) -> bool`
Validates configuration against JSON schema.

```python
is_valid = config.validate_config(config_dict)
```

**Parameters**:
- `config_dict` (dict): Configuration dictionary to validate

**Returns**: `True` if valid

**Raises**: `jsonschema.ValidationError` if validation fails

**Example Usage**:

```python
from src.core.config_loader import ConfigLoader

# Initialize and auto-load
config = ConfigLoader('config/config.yaml')

# Access configuration
data_path = config.get('paths.sourcedata')  # 'data/raw'
moments = config.get('moments')              # List of moment dicts
threshold = config.get('physio.bvp.processing.quality_threshold')  # 0.8

# Check if key exists
if config.get('custom_setting') is not None:
    # Use custom setting
    pass
```

---

## Preprocessing Modules

All preprocessing modules follow a consistent initialization pattern:
- Accept `config_path: Optional[Union[str, Path]]` parameter
- Create ConfigLoader internally
- Located in `src/physio/preprocessing/`

### BVPLoader

**Module**: `src.physio.preprocessing.bvp_loader`

Loads Blood Volume Pulse (BVP) data from BIDS-compliant TSV/JSON files with moment-based segmentation.

#### Class: `BVPLoader`

```python
from src.physio.preprocessing.bvp_loader import BVPLoader

loader = BVPLoader(config_path='config/config.yaml')
```

**Constructor Parameters**:
- `config_path` (Optional[Union[str, Path]]): Path to configuration file. If None, uses default config.

**Attributes**:
- `config` (ConfigLoader): Configuration loader instance
- `sourcedata_path` (Path): Path to raw data directory
- `sampling_rate` (int): Expected BVP sampling rate (default: 64 Hz)

**Methods**:

##### `load_subject_session(subject: str, session: str, moment: str = None) -> Tuple[pd.DataFrame, dict]`

Loads BVP data for a specific subject/session, optionally filtered by moment.

```python
# Load all moments
data, metadata = loader.load_subject_session('sub-f01p01', 'ses-01')

# Load specific moment
data, metadata = loader.load_subject_session('sub-f01p01', 'ses-01', moment='restingstate')
```

**Parameters**:
- `subject` (str): Subject ID (e.g., `'sub-f01p01'`)
- `session` (str): Session ID (e.g., `'ses-01'`)
- `moment` (str, optional): Moment/task name (e.g., `'restingstate'`, `'therapy'`). If None, loads all moments.

**Returns**: 
- `data` (pd.DataFrame): BVP data with columns `['time', 'bvp']`
- `metadata` (dict): Combined metadata from JSON sidecars

**Raises**:
- `FileNotFoundError`: If data files don't exist
- `ValueError`: If data validation fails

**DataFrame Structure**:
```
   time    bvp
0  0.000  -0.123
1  0.016   0.456
2  0.031  -0.789
```

##### `find_bvp_files(subject: str, session: str) -> List[Tuple[Path, Path]]`

Finds all BVP TSV/JSON file pairs for a subject/session.

```python
file_pairs = loader.find_bvp_files('sub-f01p01', 'ses-01')
# Returns: [(tsv_path_1, json_path_1), (tsv_path_2, json_path_2), ...]
```

**Parameters**:
- `subject` (str): Subject ID
- `session` (str): Session ID

**Returns**: List of tuples `(tsv_path, json_path)` for each recording

**Raises**: `FileNotFoundError` if no BVP files found

##### `_load_single_recording(tsv_file: Path, json_file: Path) -> Tuple[pd.DataFrame, dict]`

Internal method to load a single TSV/JSON pair.

**Example Usage**:

```python
from src.physio.bvp_loader import BVPLoader

loader = BVPLoader()

# Load all moments for analysis
all_data, all_meta = loader.load_subject_session('sub-f01p01', 'ses-01')
print(f"Loaded {len(all_data)} samples at {all_meta['SamplingFrequency']} Hz")

# Load only baseline
baseline_data, baseline_meta = loader.load_subject_session(
    'sub-f01p01', 'ses-01', moment='restingstate'
)

# Get time range
duration = baseline_data['time'].max()
print(f"Baseline duration: {duration:.1f} seconds")
```

---

### BVPCleaner

**Module**: `src.physio.preprocessing.bvp_cleaner`

Cleans and processes BVP signals using NeuroKit2 with peak detection and quality assessment.

#### Class: `BVPCleaner`

```python
from src.physio.bvp_cleaner import BVPCleaner

cleaner = BVPCleaner()
```

**Constructor Parameters**:
- `config` (ConfigLoader, optional): Configuration object. If None, loads default config.

**Attributes**:
- `config` (ConfigLoader): Configuration loader instance
- `sampling_rate` (int): BVP sampling rate (Hz)
- `processing_config` (dict): Processing parameters from config
- `method` (str): Peak detection method (default: `'elgendi'`)
- `quality_threshold` (float): Minimum acceptable quality score (0-1 scale, default: 0.8)

**Methods**:

##### `clean_signal(bvp_data: pd.DataFrame, moment: str = None) -> pd.DataFrame`

Processes raw BVP signal and extracts clean signal with peaks.

```python
# Clean BVP signal
processed = cleaner.clean_signal(raw_data, moment='restingstate')

# Access results
clean_signal = processed['PPG_Clean']
peaks = processed['PPG_Peaks']
quality = processed['PPG_Quality']
heart_rate = processed['PPG_Rate']
```

**Parameters**:
- `bvp_data` (pd.DataFrame): Raw BVP data with columns `['time', 'bvp']`
- `moment` (str, optional): Moment name for logging purposes

**Returns**: pd.DataFrame with columns:
- `PPG_Clean` (float): Cleaned BVP signal
- `PPG_Peaks` (int): Binary peak indicators (1 = peak, 0 = no peak)
- `PPG_Rate` (float): Instantaneous heart rate (BPM)
- `PPG_Quality` (float): Signal quality score (0-1, higher = better)

**Processing Pipeline**:
1. Validates input data (time and bvp columns)
2. Applies NeuroKit2 `ppg_process()` with specified method
3. Detects peaks using Elgendi algorithm (optimized for PPG)
4. Computes quality scores using template matching
5. Logs warnings if quality < threshold

**Quality Score Interpretation**:
- `0.8-1.0`: Excellent quality
- `0.7-0.8`: Good quality
- `0.65-0.7`: Acceptable (typical for long recordings)
- `< 0.65`: Poor quality, review manually

##### `get_clean_signal(processed_signals: pd.DataFrame) -> pd.Series`

Extracts cleaned signal from processed results.

```python
clean_bvp = cleaner.get_clean_signal(processed_signals)
```

**Parameters**:
- `processed_signals` (pd.DataFrame): Output from `clean_signal()`

**Returns**: pd.Series containing cleaned BVP values

##### `get_peaks(processed_signals: pd.DataFrame) -> np.ndarray`

Extracts peak indices from processed results.

```python
peak_indices = cleaner.get_peaks(processed_signals)
# Returns: array([123, 189, 256, 324, ...])  # Sample indices of peaks
```

**Parameters**:
- `processed_signals` (pd.DataFrame): Output from `clean_signal()`

**Returns**: numpy array of integer indices where peaks occur

##### `get_peak_metadata(processed_signals: pd.DataFrame, moment: str = None) -> dict`

Computes comprehensive peak statistics and quality metrics.

```python
metadata = cleaner.get_peak_metadata(processed_signals, moment='therapy')
```

**Returns**: Dictionary with keys:
```python
{
    'num_peaks': int,              # Total peaks detected
    'peak_indices': List[int],     # Peak sample indices
    'mean_quality': float,         # Average quality score
    'std_quality': float,          # Quality score std dev
    'processing_method': str,      # 'elgendi'
    'quality_method': str,         # 'templatematch'
    'sampling_rate': int,          # Hz
    'quality_threshold': float,    # Configured threshold
    'moment': str                  # Moment name (if provided)
}
```

**Example Usage**:

```python
from src.physio.bvp_cleaner import BVPCleaner
from src.physio.bvp_loader import BVPLoader

# Load and clean
loader = BVPLoader()
cleaner = BVPCleaner()

raw_data, _ = loader.load_subject_session('sub-f01p01', 'ses-01', moment='restingstate')
processed = cleaner.clean_signal(raw_data, moment='restingstate')

# Extract results
clean_signal = cleaner.get_clean_signal(processed)
peaks = cleaner.get_peaks(processed)
metadata = cleaner.get_peak_metadata(processed, moment='restingstate')

print(f"Detected {metadata['num_peaks']} peaks")
print(f"Quality: {metadata['mean_quality']:.3f}")
print(f"Heart rate range: {processed['PPG_Rate'].min():.1f}-{processed['PPG_Rate'].max():.1f} BPM")
```

---

### BVPMetrics

**Module**: `src.physio.preprocessing.bvp_metrics`

Extracts Heart Rate Variability (HRV) metrics from processed BVP signals.

#### Class: `BVPMetrics`

```python
from src.physio.bvp_metrics import BVPMetrics

metrics_extractor = BVPMetrics()
```

**Constructor Parameters**:
- `config` (ConfigLoader, optional): Configuration object. If None, loads default config.

**Attributes**:
- `config` (ConfigLoader): Configuration loader instance
- `sampling_rate` (int): BVP sampling rate
- `selected_metrics` (dict): Configured metrics to extract

**Methods**:

##### `extract_hrv_metrics(processed_signals: pd.DataFrame, moment: str = None) -> pd.DataFrame`

Extracts HRV metrics from processed BVP signal.

```python
# Extract metrics for one moment
metrics_df = metrics_extractor.extract_hrv_metrics(processed_signals, moment='restingstate')

# DataFrame has 1 row with all metrics
print(metrics_df.columns)
# ['moment', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_pNN50', ...]
```

**Parameters**:
- `processed_signals` (pd.DataFrame): Output from `BVPCleaner.clean_signal()`
- `moment` (str, optional): Moment name to include in output

**Returns**: pd.DataFrame with 1 row containing all metrics

**Extracted Metrics**:

**Time-Domain HRV (5 metrics)**:
- `HRV_MeanNN` (ms): Mean of normal-to-normal (NN) intervals
- `HRV_SDNN` (ms): Standard deviation of NN intervals (overall variability)
- `HRV_RMSSD` (ms): Root mean square of successive differences (parasympathetic activity)
- `HRV_pNN50` (%): Percentage of successive NN intervals >50ms (parasympathetic tone)
- `HRV_CVNN`: Coefficient of variation of NN intervals (normalized variability)

**Frequency-Domain HRV (4 metrics)**:
- `HRV_LF` (ms²): Low frequency power (0.04-0.15 Hz, sympathetic + parasympathetic)
- `HRV_HF` (ms²): High frequency power (0.15-0.4 Hz, parasympathetic/RSA)
- `HRV_LFHF`: Ratio of LF to HF (autonomic balance indicator)
- `HRV_TP` (ms²): Total power (overall autonomic activity)

**Nonlinear HRV (3 metrics)**:
- `HRV_SD1` (ms): Poincaré plot SD1 (short-term variability)
- `HRV_SD2` (ms): Poincaré plot SD2 (long-term variability)
- `HRV_SampEn`: Sample entropy (signal complexity/regularity, lower = more regular)

**Signal Quality Metrics (8 metrics)**:
- `BVP_NumPeaks`: Number of detected peaks
- `BVP_Duration` (s): Signal duration
- `BVP_PeakRate` (peaks/s): Average peak detection rate
- `BVP_MeanQuality`: Mean quality score (0-1)
- `BVP_QualityStd`: Standard deviation of quality scores
- `BVP_MeanAmplitude`: Mean peak amplitude
- `BVP_StdAmplitude`: Standard deviation of amplitudes
- `BVP_RangeAmplitude`: Range (max - min) of amplitudes

##### `extract_multiple_moments(moment_data: Dict[str, pd.DataFrame]) -> pd.DataFrame`

Extracts metrics for multiple moments at once.

```python
# Dictionary of moment name -> processed signals
moments = {
    'restingstate': processed_rest,
    'therapy': processed_therapy
}

# Extract all at once
all_metrics = metrics_extractor.extract_multiple_moments(moments)

# DataFrame has 2 rows (one per moment)
print(all_metrics[['moment', 'HRV_MeanNN', 'HRV_RMSSD']])
#         moment  HRV_MeanNN  HRV_RMSSD
# 0  restingstate      1201.4      85.2
# 1       therapy       963.7      42.1
```

**Parameters**:
- `moment_data` (Dict[str, pd.DataFrame]): Dictionary mapping moment names to processed signals

**Returns**: pd.DataFrame with one row per moment

##### `get_metric_descriptions() -> dict`

Returns detailed descriptions of all available metrics.

```python
descriptions = metrics_extractor.get_metric_descriptions()

print(descriptions['HRV_RMSSD'])
# {'name': 'Root Mean Square of Successive Differences',
#  'unit': 'ms',
#  'domain': 'time',
#  'description': 'Reflects parasympathetic (vagal) activity',
#  'interpretation': 'Higher values indicate better autonomic regulation'}
```

**Returns**: Dictionary with metric names as keys and description dicts as values

**Example Usage**:

```python
from src.physio.bvp_loader import BVPLoader
from src.physio.bvp_cleaner import BVPCleaner
from src.physio.bvp_metrics import BVPMetrics

# Initialize pipeline
loader = BVPLoader()
cleaner = BVPCleaner()
metrics = BVPMetrics()

# Process one moment
raw_data, _ = loader.load_subject_session('sub-f01p01', 'ses-01', moment='restingstate')
processed = cleaner.clean_signal(raw_data, moment='restingstate')
hrv_metrics = metrics.extract_hrv_metrics(processed, moment='restingstate')

# Compare multiple moments
moments_data = {}
for moment in ['restingstate', 'therapy']:
    raw, _ = loader.load_subject_session('sub-f01p01', 'ses-01', moment=moment)
    moments_data[moment] = cleaner.clean_signal(raw, moment=moment)

comparison = metrics.extract_multiple_moments(moments_data)

# Analyze differences
print("HRV Comparison:")
print(comparison[['moment', 'HRV_MeanNN', 'HRV_RMSSD', 'HRV_HF', 'HRV_LFHF']])
```

---

### BVPBIDSWriter

**Module**: `src.physio.preprocessing.bvp_bids_writer`

Writes processed BVP data in BIDS-compliant format with comprehensive metadata.

#### Class: `BVPBIDSWriter`

```python
from src.physio.bvp_bids_writer import BVPBIDSWriter

writer = BVPBIDSWriter()
```

**Constructor Parameters**:
- `config` (ConfigLoader, optional): Configuration object. If None, loads default config.

**Attributes**:
- `config` (ConfigLoader): Configuration loader instance
- `derivatives_path` (Path): Path to derivatives output directory
- `pipeline_name` (str): Pipeline name for BIDS structure (default: `'therasync-bvp'`)

**Methods**:

##### `write_processed_signals(subject: str, session: str, moment: str, processed_signals: pd.DataFrame, original_metadata: dict) -> Path`

Saves processed BVP signal to BIDS-compliant TSV/JSON files.

```python
output_path = writer.write_processed_signals(
    subject='sub-f01p01',
    session='ses-01',
    moment='restingstate',
    processed_signals=processed_data,
    original_metadata=original_meta
)
# Saves to: data/derivatives/therasync-bvp/sub-f01p01/ses-01/physio/
#   sub-f01p01_ses-01_task-restingstate_desc-processed_recording-bvp.tsv
#   sub-f01p01_ses-01_task-restingstate_desc-processed_recording-bvp.json
```

**Parameters**:
- `subject` (str): Subject ID (e.g., `'sub-f01p01'`)
- `session` (str): Session ID (e.g., `'ses-01'`)
- `moment` (str): Moment/task name (e.g., `'restingstate'`)
- `processed_signals` (pd.DataFrame): Output from `BVPCleaner.clean_signal()`
- `original_metadata` (dict): Original metadata from raw data

**Returns**: Path to saved TSV file

**Output Files**:
1. **TSV file**: Tab-separated values with columns `['time', 'PPG_Clean', 'PPG_Rate']`
2. **JSON sidecar**: Metadata including processing parameters, column descriptions, units

##### `write_processing_metadata(subject: str, session: str, moment: str, peak_metadata: dict) -> Path`

Saves processing metadata (peaks, quality) to JSON file.

```python
meta_path = writer.write_processing_metadata(
    subject='sub-f01p01',
    session='ses-01',
    moment='restingstate',
    peak_metadata=peak_meta
)
# Saves to: sub-f01p01_ses-01_task-restingstate_desc-processing_recording-bvp.json
```

**Parameters**:
- `subject` (str): Subject ID
- `session` (str): Session ID
- `moment` (str): Moment name
- `peak_metadata` (dict): Output from `BVPCleaner.get_peak_metadata()`

**Returns**: Path to saved JSON file

**Output Content**:
```json
{
  "num_peaks": 46,
  "peak_indices": [123, 189, 256, ...],
  "mean_quality": 0.81,
  "std_quality": 0.15,
  "processing_method": "elgendi",
  "quality_method": "templatematch",
  "sampling_rate": 64,
  "quality_threshold": 0.8,
  "processing_timestamp": "2025-10-28T12:34:56"
}
```

##### `write_hrv_metrics(subject: str, session: str, metrics_df: pd.DataFrame) -> Path`

Saves HRV metrics to BIDS-compliant TSV/JSON files.

```python
metrics_path = writer.write_hrv_metrics(
    subject='sub-f01p01',
    session='ses-01',
    metrics_df=hrv_metrics
)
# Saves to: sub-f01p01_ses-01_desc-bvpmetrics_physio.tsv
#           sub-f01p01_ses-01_desc-bvpmetrics_physio.json
```

**Parameters**:
- `subject` (str): Subject ID
- `session` (str): Session ID
- `metrics_df` (pd.DataFrame): Output from `BVPMetrics.extract_hrv_metrics()` or `extract_multiple_moments()`

**Returns**: Path to saved TSV file

**Output Files**:
1. **TSV file**: All HRV metrics (20 columns) with one row per moment
2. **JSON sidecar**: Comprehensive metadata with column descriptions, units, domains

##### `write_session_summary(subject: str, session: str, summary_data: dict) -> Path`

Saves session-level summary of all processing.

```python
summary = {
    'moments_processed': ['restingstate', 'therapy'],
    'total_duration': 2839.0,
    'total_peaks': 2929,
    'mean_quality': 0.755,
    'processing_complete': True
}

summary_path = writer.write_session_summary('sub-f01p01', 'ses-01', summary)
# Saves to: sub-f01p01_ses-01_desc-summary_recording-bvp.json
```

**Parameters**:
- `subject` (str): Subject ID
- `session` (str): Session ID
- `summary_data` (dict): Summary statistics dictionary

**Returns**: Path to saved JSON file

##### `create_dataset_description() -> None`

Creates/updates `dataset_description.json` at derivatives root.

```python
writer.create_dataset_description()
# Creates: data/derivatives/therasync-bvp/dataset_description.json
```

**Output Content**:
```json
{
  "Name": "Therasync BVP Processing",
  "BIDSVersion": "1.6.0",
  "DatasetType": "derivative",
  "GeneratedBy": [{
    "Name": "TherasyncPipeline",
    "Version": "0.1.0",
    "CodeURL": "https://github.com/Ramdam17/TherasyncAnalysis"
  }],
  "SourceDatasets": [{
    "URL": "local",
    "Version": "0.1.0"
  }]
}
```

**Example Usage**:

```python
from src.physio.bvp_loader import BVPLoader
from src.physio.bvp_cleaner import BVPCleaner
from src.physio.bvp_metrics import BVPMetrics
from src.physio.bvp_bids_writer import BVPBIDSWriter

# Initialize pipeline
loader = BVPLoader()
cleaner = BVPCleaner()
metrics = BVPMetrics()
writer = BVPBIDSWriter()

# Create dataset description (once per project)
writer.create_dataset_description()

# Process and save one moment
subject, session, moment = 'sub-f01p01', 'ses-01', 'restingstate'

raw_data, original_meta = loader.load_subject_session(subject, session, moment=moment)
processed = cleaner.clean_signal(raw_data, moment=moment)
peak_meta = cleaner.get_peak_metadata(processed, moment=moment)
hrv_metrics = metrics.extract_hrv_metrics(processed, moment=moment)

# Save all outputs
writer.write_processed_signals(subject, session, moment, processed, original_meta)
writer.write_processing_metadata(subject, session, moment, peak_meta)
writer.write_hrv_metrics(subject, session, hrv_metrics)

print(f"✓ Saved outputs to: data/derivatives/therasync-bvp/{subject}/{session}/physio/")
```

---

## EDA (Electrodermal Activity) Modules

### EDALoader

**Module**: `src.physio.preprocessing.eda_loader`

Loads and validates Electrodermal Activity (EDA) data from Empatica E4 devices in BIDS format.

#### Class: `EDALoader`

```python
from src.physio.eda_loader import EDALoader

loader = EDALoader()
```

**Constructor Parameters**:
- `config` (ConfigLoader, optional): Configuration instance. If `None`, creates new instance.

**Attributes**:
- `sourcedata_path` (Path): Path to raw data directory
- `sampling_rate` (int): EDA sampling rate (default: 4 Hz for Empatica E4)

**Methods**:

##### `load_subject_session(subject: str, session: str, moment: Optional[str] = None) -> Tuple[pd.DataFrame, dict]`

Loads EDA data for a subject/session, optionally filtered by moment.

```python
# Load all moments
data, metadata = loader.load_subject_session('sub-f01p01', 'ses-01')

# Load specific moment
data, metadata = loader.load_subject_session('sub-f01p01', 'ses-01', moment='restingstate')
```

**Parameters**:
- `subject` (str): Subject ID (e.g., `'sub-f01p01'`)
- `session` (str): Session ID (e.g., `'ses-01'`)
- `moment` (str, optional): Task/moment name. If `None`, loads all moments.

**Returns**:
- `data` (pd.DataFrame): EDA signal with columns `['time', 'eda']`
- `metadata` (dict): Combined metadata from JSON sidecars

**Example DataFrame**:
```
      time       eda
0     0.00     1.234
1     0.25     1.256
2     0.50     1.278
...
```

**Metadata Keys**:
- `SamplingFrequency`: 4.0 (Hz)
- `StartTime`: Start time in seconds
- `TaskName`: Moment/task name
- `RecordingType`: 'EDA'
- `Units`: ['s', 'μS']
- `DeviceManufacturer`: 'Empatica'
- `DeviceModel`: 'E4'

---

### EDACleaner

**Module**: `src.physio.preprocessing.eda_cleaner`

Decomposes EDA signal into tonic (slow baseline) and phasic (fast SCR) components using cvxEDA.

#### Class: `EDACleaner`

```python
from src.physio.eda_cleaner import EDACleaner

cleaner = EDACleaner()
```

**Constructor Parameters**:
- `config` (ConfigLoader, optional): Configuration instance

**Attributes**:
- `method` (str): Decomposition method (default: `'neurokit'` uses cvxEDA)
- `sampling_rate` (int): Sampling rate in Hz (default: 4)
- `scr_threshold` (float): Minimum SCR amplitude in μS (default: 0.01)

**Methods**:

##### `clean_signal(eda_data: pd.DataFrame, moment: Optional[str] = None) -> pd.DataFrame`

Process raw EDA signal and decompose into tonic/phasic components.

```python
raw_data, _ = loader.load_subject_session('sub-f01p01', 'ses-01', moment='restingstate')
processed = cleaner.clean_signal(raw_data, moment='restingstate')
```

**Parameters**:
- `eda_data` (pd.DataFrame): Raw EDA with columns `['time', 'eda']`
- `moment` (str, optional): Moment name for logging

**Returns**: DataFrame with processed signals

**Output Columns**:
- `time` (float): Time in seconds
- `EDA_Raw` (float): Original signal (μS)
- `EDA_Clean` (float): Cleaned signal (μS)
- `EDA_Tonic` (float): Slow-varying baseline component (μS)
- `EDA_Phasic` (float): Fast-varying response component (μS)
- `SCR_Peaks` (int): Binary indicator of SCR peaks (1=peak, 0=no peak)
- `SCR_Amplitude` (float): Amplitude of detected SCRs (μS, 0 if no peak)
- `SCR_RiseTime` (float): Rise time of SCRs (seconds, 0 if no peak)
- `SCR_RecoveryTime` (float): Recovery time of SCRs (seconds, 0 if no peak)

**Processing Steps**:
1. Clean raw signal (artifact removal)
2. Decompose using cvxEDA into tonic + phasic
3. Detect SCR peaks in phasic component (threshold: 0.01 μS)
4. Compute SCR characteristics (amplitude, rise/recovery times)

**Example**:
```python
processed = cleaner.clean_signal(raw_data, moment='therapy')
print(f"Detected {processed['SCR_Peaks'].sum()} SCRs")
print(f"Mean tonic level: {processed['EDA_Tonic'].mean():.3f} μS")
```

---

### EDAMetricsExtractor

**Module**: `src.physio.preprocessing.eda_metrics`

Extracts comprehensive EDA metrics from processed signals.

#### Class: `EDAMetricsExtractor`

```python
from src.physio.eda_metrics import EDAMetricsExtractor

extractor = EDAMetricsExtractor()
```

**Constructor Parameters**:
- `config` (ConfigLoader, optional): Configuration instance

**Attributes**:
- `selected_metrics` (list): Metrics to compute (from config)

**Methods**:

##### `extract_eda_metrics(processed_signals: pd.DataFrame, moment: Optional[str] = None) -> pd.DataFrame`

Extracts EDA metrics from processed signals.

```python
processed = cleaner.clean_signal(raw_data, moment='restingstate')
metrics = extractor.extract_eda_metrics(processed, moment='restingstate')
```

**Parameters**:
- `processed_signals` (pd.DataFrame): Output from `EDACleaner.clean_signal()`
- `moment` (str, optional): Moment name (added to output)

**Returns**: DataFrame with one row containing all metrics

**Metric Categories** (23 metrics total):

1. **SCR Metrics** (9):
   - `SCR_Peaks_N`: Number of SCR peaks detected
   - `SCR_Peaks_Rate`: SCR rate per minute
   - `SCR_Peaks_Amplitude_Mean`: Mean SCR amplitude (μS)
   - `SCR_Peaks_Amplitude_Max`: Maximum SCR amplitude (μS)
   - `SCR_Peaks_Amplitude_Sum`: Sum of all SCR amplitudes (μS)
   - `SCR_RiseTime_Mean`: Mean SCR rise time (seconds)
   - `SCR_RecoveryTime_Mean`: Mean SCR recovery time (seconds)
   - `SCR_Peaks_Amplitude_Std`: Standard deviation of SCR amplitudes
   - `SCR_Peaks_Amplitude_Min`: Minimum SCR amplitude (μS)

2. **Tonic Component Metrics** (5):
   - `EDA_Tonic_Mean`: Mean tonic level (μS)
   - `EDA_Tonic_Std`: Standard deviation of tonic level
   - `EDA_Tonic_Min`: Minimum tonic level (μS)
   - `EDA_Tonic_Max`: Maximum tonic level (μS)
   - `EDA_Tonic_Range`: Range of tonic level (μS)

3. **Phasic Component Metrics** (6):
   - `EDA_Phasic_Mean`: Mean phasic activity (μS)
   - `EDA_Phasic_Std`: Standard deviation of phasic activity
   - `EDA_Phasic_Min`: Minimum phasic value (μS)
   - `EDA_Phasic_Max`: Maximum phasic value (μS)
   - `EDA_Phasic_Range`: Range of phasic activity (μS)
   - `EDA_Phasic_Rate`: Rate of change of phasic component

4. **Metadata** (3):
   - `moment`: Moment/task name
   - `EDA_Duration`: Signal duration (seconds)
   - `EDA_SamplingRate`: Sampling rate (Hz)

**Example Output**:
```
  moment  SCR_Peaks_N  SCR_Peaks_Rate  SCR_Peaks_Amplitude_Mean  EDA_Tonic_Mean  ...
0  rest            22            22.0                     0.156           1.234  ...
```

##### `extract_multiple_moments(processed_results: Dict[str, pd.DataFrame]) -> pd.DataFrame`

Extracts metrics for multiple moments in batch.

```python
processed_results = {
    'restingstate': processed_rest,
    'therapy': processed_therapy
}

all_metrics = extractor.extract_multiple_moments(processed_results)
```

**Parameters**:
- `processed_results` (dict): Dict mapping moment names to processed DataFrames

**Returns**: DataFrame with one row per moment

---

### EDABIDSWriter

**Module**: `src.physio.preprocessing.eda_bids_writer`

Writes processed EDA data and metrics in BIDS-compliant format.

#### Class: `EDABIDSWriter`

```python
from src.physio.eda_bids_writer import EDABIDSWriter

writer = EDABIDSWriter()
```

**Constructor Parameters**:
- `config` (ConfigLoader, optional): Configuration instance

**Attributes**:
- `derivatives_path` (Path): Path to derivatives directory
- `pipeline_name` (str): Pipeline name (default: `'physio_preprocessing'`)
- `pipeline_dir` (Path): Full pipeline output directory

**Methods**:

##### `save_processed_data(subject_id: str, session_id: str, processed_results: Dict[str, pd.DataFrame], session_metrics: pd.DataFrame, processing_metadata: Optional[Dict] = None) -> Dict[str, List[str]]`

Saves all processed EDA data and metrics for a session.

```python
# Prepare data
processed_results = {
    'restingstate': processed_rest,
    'therapy': processed_therapy
}

session_metrics = extractor.extract_multiple_moments(processed_results)

# Save everything
output_files = writer.save_processed_data(
    subject_id='sub-f01p01',
    session_id='ses-01',
    processed_results=processed_results,
    session_metrics=session_metrics,
    processing_metadata={'method': 'cvxEDA', 'threshold': 0.01}
)

print(f"Created {sum(len(files) for files in output_files.values())} files")
```

**Parameters**:
- `subject_id` (str): Subject ID (e.g., `'sub-f01p01'`)
- `session_id` (str): Session ID (e.g., `'ses-01'`)
- `processed_results` (dict): Dict of moment → processed DataFrame
- `session_metrics` (pd.DataFrame): Output from `extract_multiple_moments()`
- `processing_metadata` (dict, optional): Additional processing info

**Returns**: Dictionary with lists of created file paths

**Output Structure**:
```python
{
    'processed_signals': [  # TSV + JSON per moment
        'sub-f01p01_ses-01_task-restingstate_physio.tsv.gz',
        'sub-f01p01_ses-01_task-restingstate_physio.json',
        'sub-f01p01_ses-01_task-therapy_physio.tsv.gz',
        'sub-f01p01_ses-01_task-therapy_physio.json'
    ],
    'scr_events': [  # TSV + JSON per moment (if SCRs detected)
        'sub-f01p01_ses-01_task-restingstate_events.tsv',
        'sub-f01p01_ses-01_task-restingstate_events.json',
        ...
    ],
    'metrics': [  # Session-level metrics
        'sub-f01p01_ses-01_desc-edametrics_physio.tsv',
        'sub-f01p01_ses-01_desc-edametrics_physio.json'
    ],
    'metadata': [  # Moment-specific metadata
        'sub-f01p01_ses-01_task-restingstate_desc-metadata.json',
        ...
    ],
    'summary': [  # Session summary
        'sub-f01p01_ses-01_desc-summary.json'
    ]
}
```

**File Types**:

1. **Processed Signals** (`_physio.tsv.gz` + `.json`):
   - Columns: `time`, `EDA_Raw`, `EDA_Clean`, `EDA_Tonic`, `EDA_Phasic`, `SCR_Peaks`, `SCR_Amplitude`, `SCR_RiseTime`, `SCR_RecoveryTime`
   - Compressed TSV with gzip
   - JSON sidecar with column descriptions, units

2. **SCR Events** (`_events.tsv` + `.json`):
   - Columns: `onset`, `duration`, `amplitude`, `rise_time`, `recovery_time`
   - One row per detected SCR
   - BIDS events format

3. **Metrics** (`_desc-edametrics_physio.tsv` + `.json`):
   - All 23 metrics in one row per moment
   - Session-level file (all moments combined)

4. **Metadata** (`_desc-metadata.json`):
   - Per-moment processing details
   - Quality metrics, parameters used

5. **Summary** (`_desc-summary.json`):
   - Session-level overview
   - Total SCRs, durations, processing status

##### `create_dataset_description() -> None`

Creates `dataset_description.json` for BIDS compliance (call once per project).

```python
writer.create_dataset_description()
```

**Example Complete Workflow**:

```python
from src.physio.eda_loader import EDALoader
from src.physio.eda_cleaner import EDACleaner
from src.physio.eda_metrics import EDAMetricsExtractor
from src.physio.eda_bids_writer import EDABIDSWriter

# Initialize pipeline
loader = EDALoader()
cleaner = EDACleaner()
extractor = EDAMetricsExtractor()
writer = EDABIDSWriter()

# Process one subject/session
subject, session = 'sub-f01p01', 'ses-01'

# Process each moment
processed_results = {}
for moment in ['restingstate', 'therapy']:
    raw_data, _ = loader.load_subject_session(subject, session, moment=moment)
    processed = cleaner.clean_signal(raw_data, moment=moment)
    processed_results[moment] = processed

# Extract metrics for all moments
session_metrics = extractor.extract_multiple_moments(processed_results)

# Save everything
output_files = writer.save_processed_data(
    subject_id=subject,
    session_id=session,
    processed_results=processed_results,
    session_metrics=session_metrics
)

print(f"✓ Saved {sum(len(f) for f in output_files.values())} files")
print(f"✓ Output: data/derivatives/physio_preprocessing/{subject}/{session}/physio/")
```

---

## Scripts

### preprocess_bvp.py

**Location**: `scripts/physio/preprocessing/preprocess_bvp.py`

Command-line interface for BVP preprocessing pipeline.

**Usage**:

```bash
# Basic usage (no prefix needed for subject/session)
poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject f01p01 --session 01

# With verbose logging
poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject f01p01 --session 01 --verbose

# Custom config
poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject f01p01 --session 01 --config config/custom.yaml
```

**Note**: No PYTHONPATH needed - scripts handle path setup internally.

**Arguments**:

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--subject` | str | Yes | - | Subject ID (e.g., `sub-f01p01`) |
| `--session` | str | Yes | - | Session ID (e.g., `ses-01`) |
| `--moment` | str | No | All moments | Specific moment to process |
| `--config` | str | No | `config/config.yaml` | Path to config file |
| `--verbose` | flag | No | False | Enable verbose logging |

**Pipeline Steps**:
1. Load configuration
2. Load raw BVP data (all moments or specific moment)
3. Clean signals and detect peaks
4. Extract HRV metrics
5. Save BIDS-compliant outputs:
   - Processed signals (TSV + JSON)
   - Processing metadata (JSON)
   - HRV metrics (TSV + JSON)
   - Session summary (JSON)
6. Create/update dataset description

**Output Structure**:
```
data/derivatives/preprocessing/
└── sub-f01p01/
    └── ses-01/
        └── bvp/
            ├── sub-f01p01_ses-01_task-restingstate_desc-bvp-processed_physio.tsv.gz
            ├── sub-f01p01_ses-01_task-restingstate_desc-bvp-processed_physio.json
            ├── sub-f01p01_ses-01_task-restingstate_desc-bvp-metrics_physio.tsv
            ├── sub-f01p01_ses-01_task-restingstate_desc-bvp-metrics_physio.json
            └── ... (9 files total)
```

**Exit Codes**:
- `0`: Success
- `1`: Error (check logs)

**Logging**:
- **Default**: INFO level to console and `log/preprocessing_bvp_YYYYMMDD_HHMMSS.log`
- **Verbose**: DEBUG level with detailed processing information

---

### preprocess_eda.py

**Location**: `scripts/physio/preprocessing/preprocess_eda.py`

Command-line interface for EDA preprocessing pipeline.

**Usage**:

```bash
# Basic usage (no prefix needed for subject/session)
poetry run python scripts/physio/preprocessing/preprocess_eda.py --subject f01p01 --session 01

# With verbose logging
poetry run python scripts/physio/preprocessing/preprocess_eda.py --subject f01p01 --session 01 --verbose
```

**Output Structure**:
```
data/derivatives/preprocessing/
└── sub-f01p01/
    └── ses-01/
        └── eda/
            ├── sub-f01p01_ses-01_task-restingstate_desc-eda-processed_physio.tsv.gz
            ├── sub-f01p01_ses-01_task-restingstate_desc-scr_events.tsv
            ├── sub-f01p01_ses-01_task-restingstate_desc-eda-metrics_physio.tsv
            └── ... (13 files total)
```

---

### preprocess_hr.py

**Location**: `scripts/physio/preprocessing/preprocess_hr.py`

Command-line interface for HR preprocessing pipeline.

**Usage**:

```bash
# Basic usage (no prefix needed for subject/session)
poetry run python scripts/physio/preprocessing/preprocess_hr.py --subject f01p01 --session 01

# With verbose logging
poetry run python scripts/physio/preprocessing/preprocess_hr.py --subject f01p01 --session 01 --verbose
```

**Output Structure**:
```
data/derivatives/preprocessing/
└── sub-f01p01/
    └── ses-01/
        └── hr/
            ├── sub-f01p01_ses-01_task-restingstate_desc-hr-processed_physio.tsv.gz
            ├── sub-f01p01_ses-01_task-restingstate_desc-hr-metrics_physio.tsv
            └── ... (7 files total)
```

---

### clean_outputs.py

**Location**: `scripts/utils/clean_outputs.py`

Utility script for cleaning derivatives, logs, and cache files between test iterations.

**Usage**:

```bash
# Clean specific subject/session
poetry run python scripts/utils/clean_outputs.py --subject f01p01 --session 01

# Clean specific modality
poetry run python scripts/utils/clean_outputs.py --subject f01p01 --session 01 --modality bvp

# Clean all outputs (with confirmation)
poetry run python scripts/utils/clean_outputs.py --all
```

**Arguments**:

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--subject` | str | No | All subjects | Specific subject to clean (no prefix) |
| `--session` | str | No | All sessions | Specific session to clean (no prefix) |
| `--modality` | str | No | All modalities | Specific modality (bvp, eda, hr) |
| `--all` | flag | No | False | Clean all outputs |
| `--dry-run` | flag | No | False | Preview what would be deleted |

**What Gets Cleaned**:

| Target | Location | Description |
|--------|----------|-------------|
| Preprocessing outputs | `data/derivatives/preprocessing/` | All processed outputs (BVP, EDA, HR) |
| Logs | `log/*.log` | Processing log files |

**Safety Features**:
- Confirmation prompt for destructive operations
- Dry run option to preview deletions
- Preserves raw data (never touches `data/sourcedata/`)
- Detailed deletion reporting

**Example Output**:
```
=== Cleanup Summary ===
📁 Derivatives: sub-f01p01/ses-01/bvp (9 files)
📁 Derivatives: sub-f01p01/ses-01/eda (13 files)
� Derivatives: sub-f01p01/ses-01/hr (7 files)
🗑️  Cache: 229 directories

Total to delete: 232 directories, 10 files

Proceed with deletion? [y/N]: y

✓ Deleted 232 items
```

---

### preprocess_eda.py

**Location**: `scripts/preprocess_eda.py`

Command-line interface for EDA preprocessing pipeline.

**Usage**:

```bash
# Basic usage
poetry run python scripts/preprocess_eda.py --subject sub-f01p01 --session ses-01

# With verbose logging
poetry run python scripts/preprocess_eda.py --subject sub-f01p01 --session ses-01 --verbose

# Custom config
poetry run python scripts/preprocess_eda.py --subject sub-f01p01 --session ses-01 --config config/custom.yaml

# Process specific moment only
poetry run python scripts/preprocess_eda.py --subject sub-f01p01 --session ses-01 --moment restingstate

# Batch processing
poetry run python scripts/preprocess_eda.py --subject sub-f01p01 --session ses-01 --all-moments
```

**Arguments**:

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--subject` | str | Yes | - | Subject ID (e.g., `sub-f01p01`) |
| `--session` | str | Yes | - | Session ID (e.g., `ses-01`) |
| `--moment` | str | No | Auto-detect | Specific moment to process |
| `--all-moments` | flag | No | False | Process all available moments |
| `--config` | str | No | `config/config.yaml` | Path to config file |
| `--verbose` | flag | No | False | Enable DEBUG-level logging |

**Processing Pipeline**:
1. Load raw EDA data (4 Hz from Empatica E4)
2. Clean signal and decompose (cvxEDA: tonic + phasic)
3. Detect SCR peaks (threshold: 0.01 μS)
4. Extract 23 EDA metrics (SCR, tonic, phasic)
5. Save BIDS-compliant outputs:
   - Processed signals (TSV + JSON)
   - SCR events (TSV + JSON)
   - Session metrics (TSV + JSON)
   - Moment metadata (JSON)
   - Session summary (JSON)
6. Create/update dataset description

**Output Structure**:
```
data/derivatives/physio_preprocessing/
├── dataset_description.json
└── sub-f01p01/
    └── ses-01/
        └── physio/
            ├── sub-f01p01_ses-01_task-restingstate_physio.tsv.gz
            ├── sub-f01p01_ses-01_task-restingstate_physio.json
            ├── sub-f01p01_ses-01_task-restingstate_events.tsv
            ├── sub-f01p01_ses-01_task-restingstate_events.json
            ├── sub-f01p01_ses-01_task-restingstate_desc-metadata.json
            ├── sub-f01p01_ses-01_task-therapy_physio.tsv.gz
            ├── sub-f01p01_ses-01_task-therapy_physio.json
            ├── sub-f01p01_ses-01_task-therapy_events.tsv
            ├── sub-f01p01_ses-01_task-therapy_events.json
            ├── sub-f01p01_ses-01_task-therapy_desc-metadata.json
            ├── sub-f01p01_ses-01_desc-edametrics_physio.tsv
            ├── sub-f01p01_ses-01_desc-edametrics_physio.json
            └── sub-f01p01_ses-01_desc-summary.json
```

**Example Output**:
```
=== EDA Preprocessing Pipeline ===

Subject: sub-f01p01
Session: ses-01
Moments to process: restingstate, therapy

Processing moment: restingstate
  ✓ Loaded 240 samples (60.0s at 4 Hz)
  ✓ Decomposed into tonic + phasic components
  ✓ Detected 22 SCRs (mean amplitude: 0.156 μS)
  ✓ Extracted 23 metrics
  ✓ Saved 5 output files

Processing moment: therapy
  ✓ Loaded 11064 samples (2766.0s at 4 Hz)
  ✓ Decomposed into tonic + phasic components
  ✓ Detected 791 SCRs (mean amplitude: 0.672 μS)
  ✓ Extracted 23 metrics
  ✓ Saved 5 output files

✓ Session metrics saved (2 moments)
✓ Processing complete!

Total SCRs detected: 813
Total duration: 2826.0 seconds
Output: data/derivatives/physio_preprocessing/sub-f01p01/ses-01/physio/
```

**Exit Codes**:
- `0`: Success
- `1`: Error (check logs)

**Logging**:
- **Default**: INFO level to console and `log/eda_preprocessing.log`
- **Verbose**: DEBUG level with detailed processing information, NeuroKit2 diagnostics

**Typical Processing Time**:
- Restingstate (60s): ~2 seconds
- Therapy session (45 min): ~15-20 seconds
- cvxEDA decomposition is computationally intensive for long signals

---



### Main Configuration Structure

```yaml
# Study metadata
study:
  name: str                    # Study name
  version: str                 # Version number

# Data paths
paths:
  sourcedata: str              # Raw data directory (e.g., "data/raw")
  derivatives: str             # Processed outputs (e.g., "data/derivatives")
  logs: str                    # Log files (e.g., "log")

# Moment/task definitions
moments:
  - name: str                  # Moment identifier (e.g., "restingstate")
    description: str           # Human-readable description

# Physiological processing parameters
physio:
  bvp:
    sampling_rate: int         # Expected sampling rate (Hz)
    processing:
      method: str              # Peak detection method ("elgendi")
      method_quality: str      # Quality assessment ("templatematch")
      lowcut_freq: float       # High-pass filter (Hz)
      highcut_freq: float      # Low-pass filter (Hz)
      quality_threshold: float # Quality threshold (0-1)
    metrics:
      extract_all: bool        # Extract all metrics or selected only
      selected_metrics:        # Metrics to extract
        time_domain: List[str]
        frequency_domain: List[str]
        nonlinear: List[str]

# BIDS configuration
bids:
  pipeline_name: str           # Derivatives pipeline name
  dataset_description:
    Name: str
    BIDSVersion: str
    DatasetType: str

# Logging configuration
logging:
  level: str                   # Default log level ("INFO", "DEBUG")
  format: str                  # Log message format
  file_handler:
    enabled: bool
    filename: str
    max_bytes: int
    backup_count: int
```

### Example Config Values

See `config/example_config.yaml` for a fully documented example with all available options.

---

## Error Handling

### Common Exceptions

#### `FileNotFoundError`
Raised when expected data files are missing.

```python
try:
    data, meta = loader.load_subject_session('sub-f01p01', 'ses-01')
except FileNotFoundError as e:
    print(f"Data not found: {e}")
    # Check that data exists in: data/raw/sub-f01p01/ses-01/physio/
```

#### `ValueError`
Raised when data validation fails.

```python
try:
    processed = cleaner.clean_signal(invalid_data)
except ValueError as e:
    print(f"Invalid data structure: {e}")
    # Check that data has required columns: 'time', 'bvp'
```

#### `jsonschema.ValidationError`
Raised when configuration doesn't match schema.

```python
try:
    config = ConfigLoader('invalid_config.yaml')
except jsonschema.ValidationError as e:
    print(f"Config validation failed: {e.message}")
    # Check config against schema in ConfigLoader.CONFIG_SCHEMA
```

### Logging Best Practices

```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate log levels
logger.debug("Detailed processing information")  # Verbose mode only
logger.info("Normal processing steps")           # Standard output
logger.warning("Non-fatal issues")               # Quality warnings
logger.error("Errors that prevent processing")   # Critical failures

# Include context in log messages
logger.info(f"Processing {subject}/{session}/{moment}: {len(data)} samples")
logger.warning(f"Low quality for {moment}: {quality:.3f} < {threshold}")
```

---

## Performance Considerations

### Memory Usage

- **BVP signals**: ~1.5 MB per minute at 64 Hz
- **Processed outputs**: ~3-4x larger due to additional columns
- **Long sessions (>30 min)**: Consider chunking for very large datasets

### Processing Speed

Typical processing times on standard hardware:

| Operation | Duration (1 min signal) | Duration (30 min signal) |
|-----------|------------------------|--------------------------|
| Load data | <0.1s | ~0.5s |
| Clean signal | ~0.5s | ~15s |
| Extract HRV metrics | ~0.2s | ~5s |
| Write outputs | <0.1s | ~0.5s |
| **Total** | **~1s** | **~20s** |

### Optimization Tips

1. **Batch processing**: Process multiple subjects in parallel
2. **Selective moments**: Use `--moment` flag to process only needed moments
3. **Quality threshold**: Adjust for your needs (higher = faster rejection of poor data)
4. **Metric selection**: Extract only needed metrics (set `extract_all: false`)

---

## Visualization Modules

### VisualizationDataLoader

**Module**: `src.visualization.data_loader.py`

Loads preprocessed BVP, EDA, and HR data for visualization.

#### Class: `VisualizationDataLoader`

```python
from src.visualization.data_loader import VisualizationDataLoader

# Initialize
loader = VisualizationDataLoader()

# Load data for one subject/session
data = loader.load_subject_session(
    subject="f01p01",
    session="01",
    config_name="therasync"
)

# List all available subjects/sessions
subjects = loader.list_available_subjects(config_name="therasync")
```

**Methods**:

##### `load_subject_session(subject, session, config_name="therasync")`

Load preprocessed data for a single subject/session.

**Parameters**:
- `subject` (str): Subject ID (e.g., "f01p01")
- `session` (str): Session number (e.g., "01")
- `config_name` (str): Configuration name

**Returns**: Dictionary with keys:
- `bvp`: BVP signals and metrics
  - `signals`: Dict of DataFrames per moment
  - `metrics`: DataFrame with HRV metrics
- `eda`: EDA signals and metrics
  - `signals`: Dict of DataFrames per moment
  - `scr_events`: Dict of DataFrames per moment
  - `metrics`: DataFrame with EDA metrics
- `hr`: HR signals and metrics
  - `signals`: DataFrame with combined HR data
  - `metrics`: DataFrame with HR metrics

##### `list_available_subjects(config_name="therasync")`

List all subjects/sessions with preprocessed data.

**Returns**: List of tuples `[(subject, session), ...]`

---

### VisualizationConfig

**Module**: `src.visualization.config.py`

Configuration for plot styling and parameters.

#### Class: `VisualizationConfig`

```python
from src.visualization.config import VisualizationConfig

# Initialize from YAML
config = VisualizationConfig(config_path="config/config.yaml")

# Access plot settings
dpi = config.dpi  # 300
colors = config.colors  # Color palette
figure_sizes = config.figure_sizes  # {'dashboard': (14, 10), ...}
```

**Attributes**:
- `dpi`: Plot resolution (default: 300)
- `figure_format`: Output format (default: "png")
- `colors`: Color palette for different signal types
- `figure_sizes`: Predefined figure dimensions
- `font_sizes`: Font size configuration

---

### SignalPlotter

**Module**: `src.visualization.plotters.signal_plots.py`

Generate multi-signal visualizations.

#### Class: `SignalPlotter`

```python
from src.visualization.plotters.signal_plots import SignalPlotter

# Initialize
plotter = SignalPlotter(config_path="config/config.yaml")

# Generate dashboard
output_path = plotter.plot_dashboard(
    data=data,
    subject="f01p01",
    session="01",
    output_dir="data/derivatives/visualization/sub-f01p01/ses-01/figures"
)

# Generate timeline
output_path = plotter.plot_timeline(
    hr_signals=data['hr']['signals'],
    output_dir="data/derivatives/visualization/sub-f01p01/ses-01/figures"
)
```

**Methods**:

##### `plot_dashboard(data, subject, session, output_dir)`

Create multi-signal dashboard with BVP, EDA, HR.

**Generates**: `01_dashboard_multisignals.png`

##### `plot_timeline(hr_signals, output_dir)`

Create HR dynamics timeline with variability bands.

**Generates**: `06_hr_dynamics_timeline.png`

---

### HRVPlotter

**Module**: `src.visualization.plotters.hrv_plots.py`

Generate HRV analysis visualizations.

#### Class: `HRVPlotter`

```python
from src.visualization.plotters.hrv_plots import HRVPlotter

# Initialize
plotter = HRVPlotter(config_path="config/config.yaml")

# Generate Poincaré plot
output_path = plotter.plot_poincare(
    bvp_data=data['bvp'],
    output_dir="data/derivatives/visualization/sub-f01p01/ses-01/figures"
)

# Generate autonomic balance
output_path = plotter.plot_autonomic_balance(
    bvp_data=data['bvp'],
    output_dir="data/derivatives/visualization/sub-f01p01/ses-01/figures"
)
```

**Methods**:

##### `plot_poincare(bvp_data, output_dir)`

Create Poincaré plot showing HRV non-linear dynamics (SD1/SD2).

**Generates**: `02_poincare_hrv.png`

##### `plot_autonomic_balance(bvp_data, output_dir)`

Create autonomic balance timeline (LF/HF ratio evolution).

**Generates**: `03_autonomic_balance.png`

---

### EDAPlotter

**Module**: `src.visualization.plotters.eda_plots.py`

Generate EDA/arousal visualizations.

#### Class: `EDAPlotter`

```python
from src.visualization.plotters.eda_plots import EDAPlotter

# Initialize
plotter = EDAPlotter(config_path="config/config.yaml")

# Generate arousal profile
output_path = plotter.plot_arousal_profile(
    eda_data=data['eda'],
    output_dir="data/derivatives/visualization/sub-f01p01/ses-01/figures"
)

# Generate SCR distribution
output_path = plotter.plot_scr_distribution(
    eda_data=data['eda'],
    output_dir="data/derivatives/visualization/sub-f01p01/ses-01/figures"
)
```

**Methods**:

##### `plot_arousal_profile(eda_data, output_dir)`

Create EDA arousal profile with tonic/phasic decomposition and SCR events.

**Generates**: `04_eda_arousal_profile.png`

##### `plot_scr_distribution(eda_data, output_dir)`

Create SCR amplitude distribution histogram with statistics.

**Generates**: `05_scr_distribution.png`

---

## Batch Processing

### BatchPreprocessor

**Module**: `scripts.batch.run_all_preprocessing.py`

Automated preprocessing of all subjects/sessions.

#### Class: `BatchPreprocessor`

```python
from scripts.batch.run_all_preprocessing import BatchPreprocessor

# Initialize
preprocessor = BatchPreprocessor(
    config_path="config/config.yaml",
    dry_run=False,
    skip_existing=True,
    subjects=None  # All subjects
)

# Run batch processing
preprocessor.run_batch()
```

**Parameters**:
- `config_path`: Path to YAML configuration
- `dry_run`: Preview without execution (default: False)
- `skip_existing`: Skip already processed sessions (default: False)
- `subjects`: List of subject IDs to process (default: None = all)

**Features**:
- Scans `data/raw/` for all `sub-*/ses-*` combinations
- Executes BVP → EDA → HR sequentially (dependency-aware)
- Timeout: 600 seconds per script
- Comprehensive error tracking and logging
- Keyboard interrupt handling (Ctrl+C shows partial results)

**Outputs**:
- Preprocessed data in `data/derivatives/preprocessing/`
- Timestamped log: `log/batch_preprocessing_YYYYMMDD_HHMMSS.log`

---

### BatchVisualizer

**Module**: `scripts.batch.run_all_visualizations.py`

Automated visualization generation for all preprocessed sessions.

#### Class: `BatchVisualizer`

```python
from scripts.batch.run_all_visualizations import BatchVisualizer

# Initialize
visualizer = BatchVisualizer(
    config_path="config/config.yaml",
    dry_run=False,
    plots=None,  # All plots (1-6)
    subjects=None  # All subjects
)

# Run batch visualization
visualizer.run_batch()
```

**Parameters**:
- `config_path`: Path to YAML configuration
- `dry_run`: Preview without generation (default: False)
- `plots`: List of plot numbers to generate (default: None = all)
- `subjects`: List of subject IDs to visualize (default: None = all)

**Features**:
- Auto-detects preprocessed sessions from `data/derivatives/preprocessing/`
- Generates 6 PNG files per session
- Per-plot error tracking and statistics
- Keyboard interrupt handling

**Outputs**:
- Visualizations in `data/derivatives/visualization/sub-*/ses-*/figures/`
- Timestamped log: `log/batch_visualization_YYYYMMDD_HHMMSS.log`

**Plot Numbering**:
1. Multi-signal dashboard
2. Poincaré HRV plot
3. Autonomic balance timeline
4. EDA arousal profile
5. SCR distribution
6. HR dynamics timeline

---

## Version History

### v0.5.0 (2025-11-11)

**Sprint 5 Completed**:
- ✅ Visualization pipeline: 6 core plots per session
- ✅ VisualizationDataLoader: Load preprocessed data
- ✅ Signal/HRV/EDA plotters: Publication-ready figures
- ✅ Batch preprocessing: Automated BVP→EDA→HR pipeline
- ✅ Batch visualization: Automated plot generation
- ✅ Comprehensive logging with horodated files
- ✅ Real data validation: 50/51 sessions (98% success)
- ✅ Generated 300 visualizations
- ✅ Complete documentation update

### v0.3.0 (2025-10-28)

**Modular Architecture Refactoring**:
- ✅ Restructured entire codebase
- ✅ Created `src/physio/preprocessing/` structure
- ✅ Unified module signatures across BVP/EDA/HR
- ✅ All 34 tests passing

### v0.1.0 (2025-10-28)

**Sprint 2 Completed**:
- ✅ BVPLoader: BIDS-compliant data loading
- ✅ BVPCleaner: Signal processing with NeuroKit2
- ✅ BVPMetrics: 12 essential HRV metrics
- ✅ BVPBIDSWriter: BIDS-compliant output formatting
- ✅ CLI scripts: `preprocess_bvp.py`, `clean_outputs.py`
- ✅ Comprehensive testing suite
- ✅ Full documentation

---

## Support & Contribution

**Authors**: Lena Adel, Remy Ramadour  
**Repository**: https://github.com/Ramdam17/TherasyncAnalysis  
**License**: MIT

For issues, questions, or contributions, please open an issue on GitHub.

---

**Last Updated**: November 11, 2025
