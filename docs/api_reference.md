# API Reference - TherasyncPipeline

**Authors**: Lena Adel, Remy Ramadour  
**Version**: 1.0.0 (Production Ready - Phase 2 Harmonization Complete)  
**Last Updated**: November 11, 2025

This document provides comprehensive API documentation for all modules in the TherasyncPipeline project, auto-generated from source code to ensure accuracy.

**Note**: All BIDS writers (BVP, EDA, HR) have been harmonized in Phase 2 to use identical code patterns and helper method signatures for consistency and maintainability.

---

## 📋 Table of Contents

1. [Core Modules](#core-modules)
   - [ConfigLoader](#class-configloader)
2. [BVP Preprocessing](#bvp-preprocessing)
   - [BVPLoader](#class-bvploader)
   - [BVPCleaner](#class-bvpcleaner)
   - [BVPMetricsExtractor](#class-bvpmetricsextractor)
   - [BVPBIDSWriter](#class-bvpbidswriter)
3. [EDA Preprocessing](#eda-preprocessing)
   - [EDALoader](#class-edaloader)
   - [EDACleaner](#class-edacleaner)
   - [EDAMetricsExtractor](#class-edametricsextractor)
   - [EDABIDSWriter](#class-edabidswriter)
4. [HR Preprocessing](#hr-preprocessing)
   - [HRLoader](#class-hrloader)
   - [HRCleaner](#class-hrcleaner)
   - [HRMetricsExtractor](#class-hrmetricsextractor)
   - [HRBIDSWriter](#class-hrbidswriter)
5. [Epoching](#epoching)
   - [EpochAssigner](#class-epochassigner)
   - [EpochBIDSWriter](#class-epochbidswriter)
6. [Version History](#version-history)
7. [Support & Contribution](#support--contribution)

---

## Core Modules

#### Class: `ConfigLoader`

**Module**: `src.core.config_loader`

Configuration loader with schema validation and environment variable support.

This class handles loading YAML configuration files, validating them against
a schema, and providing easy access to configuration parameters.

**Constructor**:

```python
ConfigLoader(self, config_path: Union[str, pathlib.Path, NoneType] = None)
```

**Methods**:

##### `get(self, key: str, default: Any = None) -> Any`

Get a configuration value using dot notation.

Args:
    key: Configuration key in dot notation (e.g., 'physio.bvp.sampling_rate').
    default: Default value if key is not found.
    
Returns:
    Configuration value or default.

##### `get_bids_config(self) -> Dict[str, Any]`

Get BIDS configuration settings.

Returns:
    BIDS configuration dictionary.

##### `get_moment_names(self) -> list`

Get the list of moment names.

Returns:
    List of moment names.

##### `get_moments(self) -> list`

Get the list of configured moments/tasks.

Returns:
    List of moment configurations.

##### `get_paths(self) -> Dict[str, str]`

Get path configurations.

Returns:
    Dictionary of configured paths.

##### `get_physio_config(self, signal_type: str) -> Dict[str, Any]`

Get physiological signal configuration.

Args:
    signal_type: Type of signal ('bvp', 'eda', 'hr').
    
Returns:
    Configuration dictionary for the specified signal type.

##### `load_config(self) -> Dict[str, Any]`

Load and validate the configuration file.

Returns:
    Loaded and validated configuration dictionary.
    
Raises:
    ConfigError: If configuration file is invalid or missing.

##### `save_config(self, output_path: Union[str, pathlib.Path, NoneType] = None) -> None`

Save the current configuration to a file.

Args:
    output_path: Path to save the configuration. If None, overwrites original.


---

## BVP Preprocessing

#### Class: `BVPLoader`

**Module**: `src.physio.preprocessing.bvp_loader`

Load and validate BVP data files from BIDS-formatted Empatica recordings.

This class handles loading BVP data with associated metadata, validates data
integrity, and segments data according to configured moments (e.g., resting_state, therapy).

**Constructor**:

```python
BVPLoader(self, config_path: Union[str, pathlib.Path, NoneType] = None)
```

**Methods**:

##### `get_available_data(self, subject_pattern: str = '*') -> Dict[str, Dict[str, List[str]]]`

Scan for available BVP data files in the sourcedata directory.

Args:
    subject_pattern: Pattern to match subject directories (default: all subjects)
    
Returns:
    Nested dictionary: {subject: {session: [moments]}}

##### `load_moment_data(self, subject_id: str, session_id: str, moment: str) -> Tuple[pandas.core.frame.DataFrame, Dict]`

Load BVP data for a specific moment (task).

Args:
    subject_id: Subject identifier (e.g., 'sub-f01p01')
    session_id: Session identifier (e.g., 'ses-01')
    moment: Moment name (e.g., 'restingstate', 'therapy')
    
Returns:
    Tuple of (data_dataframe, metadata_dict)
    
Raises:
    FileNotFoundError: If TSV or JSON files are not found
    ValueError: If data validation fails

##### `load_subject_session_data(self, subject_id: str, session_id: str, moments: Optional[List[str]] = None) -> Dict[str, Dict]`

Load all BVP data for a subject and session across specified moments.

Args:
    subject_id: Subject identifier (e.g., 'sub-f01p01')
    session_id: Session identifier (e.g., 'ses-01')
    moments: List of moment names to load. If None, loads all configured moments.
    
Returns:
    Dictionary with moment names as keys and loaded data as values.
    Each moment contains: {'data': DataFrame, 'metadata': Dict}
    
Raises:
    FileNotFoundError: If required data files are not found
    ValueError: If data validation fails

#### Class: `BVPCleaner`

**Module**: `src.physio.preprocessing.bvp_cleaner`

Clean and process BVP data using NeuroKit2 PPG processing pipeline.

This class implements the BVP preprocessing approach established in the original
Therasync project, using nk.ppg_process with elgendi method for peak detection
and templatematch for quality assessment.

**Constructor**:

```python
BVPCleaner(self, config_path: Union[str, pathlib.Path, NoneType] = None)
```

**Methods**:

##### `apply_additional_filtering(self, signal: Union[pandas.core.series.Series, numpy.ndarray], sampling_rate: int) -> numpy.ndarray`

Apply additional filtering to BVP signal if needed.

This method provides optional additional filtering using the configured
frequency bounds, separate from the main nk.ppg_process pipeline.

Args:
    signal: BVP signal to filter
    sampling_rate: Sampling rate in Hz
    
Returns:
    Filtered signal array

##### `get_clean_signal(self, processed_signals: pandas.core.frame.DataFrame) -> pandas.core.series.Series`

Extract the cleaned BVP signal from processed results.

Args:
    processed_signals: DataFrame from process_signal()
    
Returns:
    Series containing the cleaned BVP signal

##### `get_peaks(self, processing_info: Dict) -> numpy.ndarray`

Extract detected peaks from processing results.

Args:
    processing_info: Info dictionary from process_signal()
    
Returns:
    Array of peak indices

##### `get_quality_scores(self, processed_signals: pandas.core.frame.DataFrame) -> Optional[pandas.core.series.Series]`

Extract signal quality scores if available.

Args:
    processed_signals: DataFrame from process_signal()
    
Returns:
    Series containing quality scores, or None if not available

##### `process_moment_signals(self, moment_data: Dict[str, Union[pandas.core.frame.DataFrame, Dict]]) -> Dict[str, Tuple[pandas.core.frame.DataFrame, Dict]]`

Process BVP signals for all moments in a dataset.

Args:
    moment_data: Dictionary with moment names as keys and data/metadata as values.
                Expected format: {moment: {'data': DataFrame, 'metadata': Dict}}
    
Returns:
    Dictionary with processed signals and info for each moment.
    Format: {moment: (processed_signals, processing_info)}

##### `process_signal(self, bvp_signal: Union[pandas.core.series.Series, numpy.ndarray, List], sampling_rate: Optional[int] = None, moment: str = 'unknown') -> Tuple[pandas.core.frame.DataFrame, Dict[str, Any]]`

Process a BVP signal using NeuroKit2 PPG processing pipeline.

This method follows the established Therasync approach:
1. Apply nk.ppg_process with elgendi method and templatematch quality
2. Extract cleaned signal and processing information
3. Validate results and apply quality checks

Args:
    bvp_signal: Raw BVP signal data
    sampling_rate: Sampling rate in Hz. If None, uses config default.
    moment: Name of the moment/task being processed (for logging)
    
Returns:
    Tuple of (processed_signals_dataframe, processing_info_dict)
    
Raises:
    ValueError: If signal is empty or processing fails
    RuntimeError: If NeuroKit2 processing encounters errors

#### Class: `BVPMetricsExtractor`

**Module**: `src.physio.preprocessing.bvp_metrics`

Extract HRV and cardiovascular metrics from processed BVP data.

This class implements the essential HRV metrics extraction using NeuroKit2,
supporting both session-level analysis and future epoched analysis capabilities.

**Constructor**:

```python
BVPMetricsExtractor(self, config_path: Union[str, pathlib.Path, NoneType] = None)
```

**Methods**:

##### `compare_moments(self, session_metrics: Dict[str, Dict[str, float]], baseline_moment: str = 'restingstate', comparison_moment: str = 'therapy') -> Dict[str, float]`

Compare metrics between two moments (e.g., resting vs therapy).

Args:
    session_metrics: Output from extract_session_metrics()
    baseline_moment: Name of baseline moment
    comparison_moment: Name of comparison moment
    
Returns:
    Dictionary of differences (comparison - baseline)

##### `extract_epoched_metrics(self, processed_signals: pandas.core.frame.DataFrame, processing_info: Dict, moment: str) -> pandas.core.frame.DataFrame`

Extract HRV metrics from sliding windows (future implementation).

This method will implement the 30-second sliding window approach
with 1-second steps for dynamic HRV analysis.

Args:
    processed_signals: Processed signals DataFrame
    processing_info: Processing information dictionary
    moment: Moment name
    
Returns:
    DataFrame with time-series of HRV metrics

##### `extract_session_metrics(self, processed_results: Dict[str, Tuple[pandas.core.frame.DataFrame, Dict]]) -> Dict[str, Dict[str, float]]`

Extract HRV metrics for entire sessions/moments.

Args:
    processed_results: Output from BVPCleaner.process_moment_signals()
                      Format: {moment: (processed_signals, processing_info)}
    
Returns:
    Dictionary with extracted metrics for each moment.
    Format: {moment: {metric_name: value}}

##### `extract_rr_intervals(self, peaks: numpy.ndarray, sampling_rate: float, moment: str) -> pandas.core.frame.DataFrame`

Extract RR intervals (peak-to-peak) from detected BVP peaks.

Calculates time intervals between consecutive peaks and validates them against
physiological thresholds. All intervals are preserved with a validity flag.

Args:
    peaks: Array of peak indices from BVP signal processing
    sampling_rate: Sampling rate in Hz (typically 64 Hz for BVP)
    moment: Moment name (e.g., 'restingstate', 'therapy')
    
Returns:
    DataFrame with columns:
        - time_peak_start: Timestamp of interval start peak (seconds)
        - time_peak_end: Timestamp of interval end peak (seconds)
        - rr_interval_ms: Duration between peaks (milliseconds)
        - is_valid: Validity flag (1 = valid, 0 = invalid)
        
Notes:
    - Valid range defaults: 300-2000 ms (30-200 BPM)
    - Configurable via config.yaml: physio.bvp.rr_intervals
    - Invalid intervals marked with is_valid=0 (not filtered out)
    - First peak has no prior interval (starts at second peak)

Example:
    >>> peaks = processing_info['peaks']
    >>> rr_intervals = extractor.extract_rr_intervals(peaks, 64.0, 'therapy')
    >>> valid_intervals = rr_intervals[rr_intervals['is_valid'] == 1]
    >>> print(f"Valid rate: {len(valid_intervals) / len(rr_intervals) * 100:.1f}%")

##### `get_configured_metrics_list(self) -> List[str]`

Get list of all configured metrics that will be extracted.

Returns:
    List of metric names

##### `get_metrics_summary(self, session_metrics: Dict[str, Dict[str, float]]) -> pandas.core.frame.DataFrame`

Convert session metrics to a summary DataFrame.

Args:
    session_metrics: Output from extract_session_metrics()
    
Returns:
    DataFrame with moments as rows and metrics as columns

#### Class: `BVPBIDSWriter`

**Module**: `src.physio.preprocessing.bvp_bids_writer`

Save processed BVP data and metrics in BIDS-compliant format.

This class handles saving processed signals, extracted metrics, and metadata
following BIDS derivatives specifications for physiological data.

**Constructor**:

```python
BVPBIDSWriter(self, config_path: Union[str, pathlib.Path, NoneType] = None)
```

**Methods**:

##### `create_group_summary(self, subjects_data: Dict[str, Dict[str, Dict[str, float]]], output_filename: str = 'group_bvp_metrics.tsv') -> str`

Create group-level summary of BVP metrics across subjects.

Args:
    subjects_data: Nested dict {subject_id: {session_id: session_metrics}}
    output_filename: Name of output file
    
Returns:
    Path to created group summary file

##### `save_processed_data(self, subject_id: str, session_id: str, processed_results: Dict[str, pandas.core.frame.DataFrame], session_metrics: Optional[pandas.core.frame.DataFrame] = None, processing_metadata: Optional[Dict] = None) -> Dict[str, List[pathlib.Path]]`

Save processed BVP data and metrics in BIDS format.

Args:
    subject_id: Subject identifier WITH prefix (e.g., 'sub-f01p01')
    session_id: Session identifier WITH prefix (e.g., 'ses-01')
    processed_results: Dictionary mapping moment names to processed DataFrames
                     Expected columns: time, PPG_Raw, PPG_Clean, PPG_Quality, PPG_Peaks, PPG_Rate
    session_metrics: DataFrame with session-level metrics (optional)
    processing_metadata: Dictionary with moment-specific processing info (optional)
    
Returns:
    Dictionary with lists of created file paths

##### `save_rr_intervals(self, subject_id: str, session_id: str, moment: str, rr_intervals_df: pandas.core.frame.DataFrame) -> Tuple[pathlib.Path, pathlib.Path]`

Save RR intervals data in BIDS-compliant format.

Creates TSV file with RR intervals time-series and JSON sidecar with metadata.
Files follow BIDS naming: `*_task-{moment}_desc-rrintervals_physio.{tsv,json}`

Args:
    subject_id: Subject identifier WITH prefix (e.g., 'sub-f01p01')
    session_id: Session identifier WITH prefix (e.g., 'ses-01')
    moment: Moment/task name (e.g., 'restingstate', 'therapy')
    rr_intervals_df: DataFrame from BVPMetricsExtractor.extract_rr_intervals()
                    Expected columns: time_peak_start, time_peak_end, rr_interval_ms, is_valid
    
Returns:
    Tuple of (tsv_path, json_path) for created files

Notes:
    - TSV saved with 3 decimal precision for timestamps
    - JSON sidecar includes column descriptions, units, valid range, statistics
    - Output directory: derivatives/therasync-bvp/{subject}/{session}/physio/
    - Statistics include total intervals, valid count, valid percentage

Example:
    >>> rr_df = extractor.extract_rr_intervals(peaks, 64.0, 'therapy')
    >>> tsv_path, json_path = writer.save_rr_intervals('sub-f01p01', 'ses-01', 'therapy', rr_df)
    >>> print(f"Saved {len(rr_df)} intervals to {tsv_path}")


---

## EDA Preprocessing

#### Class: `EDALoader`

**Module**: `src.physio.preprocessing.eda_loader`

Load and validate EDA data files from BIDS-formatted Empatica recordings.

This class handles loading EDA (skin conductance) data with associated metadata,
validates data integrity, and segments data according to configured moments
(e.g., restingstate, therapy).

EDA data from Empatica E4:
- Sampling rate: 4 Hz
- Unit: microsiemens (μS)
- Measures skin conductance response (SCR)

**Constructor**:

```python
EDALoader(self, config_path: Union[str, pathlib.Path, NoneType] = None)
```

**Methods**:

##### `find_eda_files(self, subject: str, session: str) -> List[Tuple[pathlib.Path, pathlib.Path]]`

Find all EDA TSV/JSON file pairs for a subject/session.

Args:
    subject: Subject ID (e.g., 'sub-f01p01')
    session: Session ID (e.g., 'ses-01')

Returns:
    List of tuples (tsv_path, json_path) for each EDA recording

Raises:
    FileNotFoundError: If physio directory doesn't exist

Example:
    >>> loader = EDALoader()
    >>> files = loader.find_eda_files('sub-f01p01', 'ses-01')
    >>> print(f"Found {len(files)} EDA recordings")

##### `get_data_info(self, data: pandas.core.frame.DataFrame, metadata: dict) -> dict`

Get summary information about loaded EDA data.

Args:
    data: Loaded EDA DataFrame
    metadata: Associated metadata

Returns:
    Dictionary with data summary information

Example:
    >>> info = loader.get_data_info(data, metadata)
    >>> print(f"Samples: {info['num_samples']}, Duration: {info['duration_seconds']:.1f}s")

##### `get_moment_duration(self, data: pandas.core.frame.DataFrame) -> float`

Calculate duration of EDA recording in seconds.

Args:
    data: DataFrame with time column

Returns:
    Duration in seconds

Example:
    >>> duration = loader.get_moment_duration(data)
    >>> print(f"Recording duration: {duration:.1f} seconds")

##### `load_subject_session(self, subject: str, session: str, moment: Optional[str] = None) -> Tuple[pandas.core.frame.DataFrame, dict]`

Load EDA data for a specific subject/session, optionally filtered by moment.

Args:
    subject: Subject ID (e.g., 'sub-f01p01')
    session: Session ID (e.g., 'ses-01')
    moment: Optional moment/task name (e.g., 'restingstate', 'therapy').
           If None, loads and concatenates all moments.

Returns:
    Tuple of:
        - DataFrame with columns ['time', 'eda']
        - Dictionary with combined metadata from JSON sidecars

Raises:
    FileNotFoundError: If no EDA files found for subject/session
    ValueError: If data validation fails

Example:
    >>> loader = EDALoader()
    >>> data, metadata = loader.load_subject_session('sub-f01p01', 'ses-01', moment='restingstate')
    >>> print(f"Loaded {len(data)} samples at {metadata['SamplingFrequency']} Hz")

#### Class: `EDACleaner`

**Module**: `src.physio.preprocessing.eda_cleaner`

Clean and process EDA signals using NeuroKit2.

This class handles:
- Signal cleaning and filtering
- Tonic-phasic decomposition
- SCR (Skin Conductance Response) peak detection
- Quality assessment

The tonic component represents the baseline skin conductance level,
while the phasic component represents rapid SCRs (sympathetic arousal responses).

**Constructor**:

```python
EDACleaner(self, config_path: Union[str, pathlib.Path, NoneType] = None)
```

**Methods**:

##### `clean_signal(self, eda_data: pandas.core.frame.DataFrame, moment: Optional[str] = None) -> pandas.core.frame.DataFrame`

Process raw EDA signal and decompose into tonic and phasic components.

This method:
1. Cleans the raw EDA signal
2. Decomposes into tonic (slow-changing baseline) and phasic (fast responses) components
3. Detects SCR peaks in the phasic component
4. Computes quality metrics

Args:
    eda_data: DataFrame with columns ['time', 'eda']
    moment: Optional moment name for logging

Returns:
    DataFrame with processed EDA signals including:
        - EDA_Raw: Original signal
        - EDA_Clean: Cleaned signal
        - EDA_Tonic: Slow-varying baseline (tonic component)
        - EDA_Phasic: Fast-varying responses (phasic component)
        - SCR_Peaks: Binary indicators of SCR peaks (1 = peak, 0 = no peak)
        - SCR_Amplitude: Amplitude of detected SCRs (0 if no peak)
        - SCR_RiseTime: Rise time of SCRs (0 if no peak)
        - SCR_RecoveryTime: Recovery time of SCRs (0 if no peak)

Raises:
    ValueError: If input data is invalid

Example:
    >>> cleaner = EDACleaner()
    >>> processed = cleaner.clean_signal(raw_data, moment='restingstate')
    >>> print(f"Detected {processed['SCR_Peaks'].sum()} SCRs")

##### `compute_scr_features(self, processed_signals: pandas.core.frame.DataFrame) -> pandas.core.frame.DataFrame`

Compute detailed features for each detected SCR.

Args:
    processed_signals: Output from clean_signal()

Returns:
    DataFrame with one row per SCR, containing:
        - scr_index: Sample index of SCR peak
        - scr_time: Time of SCR peak (seconds)
        - scr_amplitude: SCR amplitude (μS)
        - scr_rise_time: Rise time (seconds)
        - scr_recovery_time: Recovery time (seconds)

Example:
    >>> scr_features = cleaner.compute_scr_features(processed_signals)
    >>> print(f"Found {len(scr_features)} SCRs")
    >>> print(scr_features[['scr_time', 'scr_amplitude']].head())

##### `get_phasic_component(self, processed_signals: pandas.core.frame.DataFrame) -> pandas.core.series.Series`

Extract phasic (response) component from processed signals.

The phasic component represents rapid skin conductance responses (SCRs)
associated with sympathetic nervous system activity and emotional arousal.

Args:
    processed_signals: Output from clean_signal()

Returns:
    Series containing phasic EDA values

Example:
    >>> phasic = cleaner.get_phasic_component(processed_signals)
    >>> print(f"Mean phasic activity: {phasic.mean():.4f} μS")

##### `get_scr_metadata(self, processed_signals: pandas.core.frame.DataFrame, moment: Optional[str] = None) -> dict`

Compute comprehensive SCR statistics and quality metrics.

Args:
    processed_signals: Output from clean_signal()
    moment: Optional moment name to include in metadata

Returns:
    Dictionary with SCR statistics:
        - num_scrs: Total number of detected SCRs
        - scr_indices: List of sample indices where SCRs occur
        - scr_rate: SCRs per minute
        - mean_scr_amplitude: Average SCR amplitude (μS)
        - max_scr_amplitude: Maximum SCR amplitude (μS)
        - mean_rise_time: Average rise time (seconds)
        - mean_recovery_time: Average recovery time (seconds)
        - tonic_mean: Mean tonic level (μS)
        - tonic_std: Tonic level standard deviation (μS)
        - phasic_mean: Mean phasic activity (μS)
        - phasic_std: Phasic activity standard deviation (μS)
        - processing_method: Method used for processing
        - sampling_rate: Sampling rate (Hz)
        - duration_seconds: Signal duration

Example:
    >>> metadata = cleaner.get_scr_metadata(processed_signals, moment='therapy')
    >>> print(f"SCR rate: {metadata['scr_rate']:.2f} per minute")

##### `get_scr_peaks(self, processed_signals: pandas.core.frame.DataFrame) -> numpy.ndarray`

Extract SCR peak indices from processed signals.

Args:
    processed_signals: Output from clean_signal()

Returns:
    Array of sample indices where SCR peaks occur

Example:
    >>> peaks = cleaner.get_scr_peaks(processed_signals)
    >>> print(f"SCR peaks at indices: {peaks}")

##### `get_tonic_component(self, processed_signals: pandas.core.frame.DataFrame) -> pandas.core.series.Series`

Extract tonic (baseline) component from processed signals.

The tonic component represents the slow-varying baseline skin conductance level.
It reflects overall arousal state and changes gradually over time.

Args:
    processed_signals: Output from clean_signal()

Returns:
    Series containing tonic EDA values

Example:
    >>> tonic = cleaner.get_tonic_component(processed_signals)
    >>> print(f"Mean tonic level: {tonic.mean():.3f} μS")

#### Class: `EDAMetricsExtractor`

**Module**: `src.physio.preprocessing.eda_metrics`

Extract EDA metrics from processed signals.

This class computes comprehensive metrics from EDA data:
- SCR features: count, amplitude (mean/max), rise time, recovery time
- Tonic component: mean, standard deviation
- Phasic component: mean, standard deviation, rate

Metrics can be extracted for full recordings or individual moments.

**Constructor**:

```python
EDAMetricsExtractor(self, config_path: Union[str, pathlib.Path, NoneType] = None)
```

**Methods**:

##### `extract_eda_metrics(self, processed_signals: pandas.core.frame.DataFrame, moment: Optional[str] = None) -> pandas.core.frame.DataFrame`

Extract EDA metrics from processed signals.

Args:
    processed_signals: Output from EDACleaner.clean_signal()
        Must contain columns: EDA_Tonic, EDA_Phasic, SCR_Peaks, SCR_Amplitude,
        SCR_RiseTime, SCR_RecoveryTime
    moment: Optional moment name to include in output

Returns:
    DataFrame with one row containing all EDA metrics

Example:
    >>> extractor = EDAMetricsExtractor()
    >>> metrics = extractor.extract_eda_metrics(processed_signals, moment='restingstate')
    >>> print(metrics[['SCR_Peaks_N', 'SCR_Peaks_Amplitude_Mean', 'EDA_Tonic_Mean']])

##### `extract_multiple_moments(self, moment_data: Dict[str, pandas.core.frame.DataFrame]) -> pandas.core.frame.DataFrame`

Extract EDA metrics for multiple moments at once.

Args:
    moment_data: Dictionary mapping moment names to processed signals
        Example: {'restingstate': processed_rest, 'therapy': processed_therapy}

Returns:
    DataFrame with one row per moment containing all metrics

Example:
    >>> moments = {
    ...     'restingstate': processed_rest,
    ...     'therapy': processed_therapy
    ... }
    >>> all_metrics = extractor.extract_multiple_moments(moments)
    >>> print(all_metrics[['moment', 'SCR_Peaks_N', 'EDA_Tonic_Mean']])

##### `get_metric_descriptions(self) -> Dict[str, dict]`

Get detailed descriptions of all EDA metrics.

Returns:
    Dictionary with metric names as keys and description dicts as values.
    Each description contains: name, unit, domain, description, interpretation

Example:
    >>> extractor = EDAMetricsExtractor()
    >>> descriptions = extractor.get_metric_descriptions()
    >>> print(descriptions['SCR_Peaks_N'])

##### `get_selected_metrics(self) -> List[str]`

Get list of metrics configured to be extracted.

Returns:
    List of metric names to extract

#### Class: `EDABIDSWriter`

**Module**: `src.physio.preprocessing.eda_bids_writer`

Save processed EDA data and metrics in BIDS-compliant format.

This class handles saving processed signals (tonic, phasic, SCR events), 
extracted metrics, and metadata following BIDS derivatives specifications 
for physiological data.

Inherits from PhysioBIDSWriter to ensure consistent API across modalities.

**Constructor**:

```python
EDABIDSWriter(self, config_path: str)
```

**Methods**:

##### `create_group_summary(self, subjects_data: Dict[str, Dict[str, pandas.core.frame.DataFrame]], output_filename: str = 'group_eda_metrics.tsv') -> str`

Create group-level summary of EDA metrics across subjects.

Args:
    subjects_data: Nested dict {subject_id: {session_id: metrics_df}}
    output_filename: Name of output file
    
Returns:
    Path to created group summary file

##### `save_processed_data(self, subject_id: str, session_id: str, processed_results: Dict[str, pandas.core.frame.DataFrame], session_metrics: Optional[pandas.core.frame.DataFrame] = None, processing_metadata: Optional[Dict] = None) -> Dict[str, List[pathlib.Path]]`

Save processed EDA data and metrics in BIDS format.

Args:
    subject_id: Subject identifier (with or without 'sub-' prefix)
    session_id: Session identifier (with or without 'ses-' prefix)
    processed_results: Dict of processed DataFrames from EDACleaner 
                     (keys: moment names, values: processed signals with EDA_Quality)
    session_metrics: DataFrame with extracted metrics from EDAMetricsExtractor
    processing_metadata: Additional metadata about processing
    
Returns:
    Dictionary with lists of created file paths (Path objects)


---

## HR Preprocessing

#### Class: `HRLoader`

**Module**: `src.physio.preprocessing.hr_loader`

Load and validate HR data files from BIDS-formatted Empatica recordings.

This class handles loading HR (heart rate) data with associated metadata,
validates data integrity, and segments data according to configured moments
(e.g., restingstate, therapy).

HR data from Empatica E4:
- Sampling rate: 1 Hz
- Unit: beats per minute (BPM)
- Measures instantaneous heart rate

**Constructor**:

```python
HRLoader(self, config: Optional[src.core.config_loader.ConfigLoader] = None)
```

**Methods**:

##### `find_hr_files(self, subject: str, session: str) -> List[Tuple[pathlib.Path, pathlib.Path]]`

Find all HR TSV and JSON file pairs for a subject/session.

Args:
    subject: Subject ID (e.g., 'sub-f01p01')
    session: Session ID (e.g., 'ses-01')

Returns:
    List of tuples, each containing (tsv_path, json_path)

##### `get_available_moments(self, subject: str, session: str) -> List[str]`

Get list of available moments/tasks for a subject/session.

Args:
    subject: Subject ID
    session: Session ID

Returns:
    List of moment names (e.g., ['restingstate', 'therapy'])

##### `load_single_moment(self, subject: str, session: str, moment: str) -> Tuple[pandas.core.frame.DataFrame, dict]`

Load HR data for a single moment/task.

Args:
    subject: Subject ID
    session: Session ID  
    moment: Moment/task name

Returns:
    Tuple of (DataFrame, metadata) for the specified moment

Raises:
    FileNotFoundError: If moment not found

##### `load_subject_session(self, subject: str, session: str, moment: Optional[str] = None) -> Tuple[pandas.core.frame.DataFrame, dict]`

Load HR data for a specific subject/session, optionally filtered by moment.

Args:
    subject: Subject ID (e.g., 'sub-f01p01')
    session: Session ID (e.g., 'ses-01')
    moment: Optional moment/task name (e.g., 'restingstate', 'therapy').
           If None, loads and concatenates all moments.

Returns:
    Tuple of:
        - DataFrame with columns ['time', 'hr']
        - Dictionary with combined metadata from JSON sidecars

Raises:
    FileNotFoundError: If no HR files found for subject/session
    ValueError: If data validation fails

Example:
    >>> loader = HRLoader()
    >>> data, metadata = loader.load_subject_session('sub-f01p01', 'ses-01', moment='restingstate')
    >>> print(f"Loaded {len(data)} samples at {metadata['SamplingFrequency']} Hz")

#### Class: `HRCleaner`

**Module**: `src.physio.preprocessing.hr_cleaner`

Clean and preprocess HR data with conservative approach.

This class handles:
- Physiological outlier removal (< 40 or > 180 BPM)
- Short gap interpolation (< 5 seconds)
- Quality assessment and scoring
- Data validation

HR data characteristics:
- Sampling rate: 1 Hz
- Unit: BPM (beats per minute)
- Expected range: 40-180 BPM for most populations

**Constructor**:

```python
HRCleaner(self, config: Optional[src.core.config_loader.ConfigLoader] = None)
```

**Methods**:

##### `clean_signal(self, data: pandas.core.frame.DataFrame, moment: str = 'unknown') -> Tuple[pandas.core.frame.DataFrame, Dict]`

Clean HR signal with conservative approach.

Args:
    data: DataFrame with columns ['time', 'hr']
    moment: Moment/task name for logging

Returns:
    Tuple of:
        - Cleaned DataFrame with additional columns:
          ['time', 'hr', 'hr_clean', 'hr_outliers', 'hr_interpolated', 'hr_quality']
        - Processing metadata dictionary

Example:
    >>> cleaner = HRCleaner()
    >>> cleaned_data, metadata = cleaner.clean_signal(raw_data, moment='restingstate')
    >>> print(f"Quality score: {metadata['quality_score']:.3f}")

##### `validate_cleaning_quality(self, metadata: Dict) -> Tuple[bool, str]`

Validate the quality of cleaning results.

Args:
    metadata: Cleaning metadata from clean_signal()

Returns:
    Tuple of (is_valid, message)

#### Class: `HRMetricsExtractor`

**Module**: `src.physio.preprocessing.hr_metrics_extractor`

Extract comprehensive HR metrics from cleaned data.

This class extracts 25 HR metrics across 5 categories:
1. Descriptive Statistics (7): Mean, SD, Min, Max, Range, Median, IQR
2. Trend Analysis (5): Slope, Initial_HR, Final_HR, HR_Change, Peak_Time
3. Stability Metrics (4): HR_Stability, RMSSD_Simple, CV, MAD
4. Response Patterns (6): Elevated_Percent, Recovery_Rate, Acceleration, etc.
5. Contextual Metrics (3): Duration, Valid_Samples, Quality_Score

Note: This is distinct from HRV metrics (already extracted from BVP pipeline).
HR metrics focus on beat-to-beat heart rate patterns, not inter-beat intervals.

**Constructor**:

```python
HRMetricsExtractor(self, config: Optional[src.core.config_loader.ConfigLoader] = None)
```

**Methods**:

##### `extract_metrics(self, data: pandas.core.frame.DataFrame, moment: str = 'unknown') -> Dict[str, Any]`

Extract comprehensive HR metrics from cleaned data.

Args:
    data: DataFrame with columns ['time', 'hr_clean', 'hr_quality']
    moment: Moment/task name for context

Returns:
    Dictionary with 25 HR metrics organized by category

Example:
    >>> extractor = HRMetricsExtractor()
    >>> metrics = extractor.extract_metrics(cleaned_data, moment='therapy')
    >>> print(f"Mean HR: {metrics['descriptive']['hr_mean']:.1f} BPM")

##### `get_metrics_description(self) -> Dict[str, Dict[str, str]]`

Get detailed descriptions of all HR metrics.

Returns:
    Dictionary with metric descriptions by category

#### Class: `HRBIDSWriter`

**Module**: `src.physio.preprocessing.hr_bids_writer`

Write HR processing results in BIDS-compliant format.

This class creates 7 file types per moment following the BIDS specification:
1. _desc-processed_recording-hr.tsv: Processed HR signals (uncompressed)
2. _desc-processed_recording-hr.json: Signal metadata and processing parameters
3. _events.tsv: HR-related events (elevated periods, peaks, etc.)
4. _events.json: Events metadata
5. _desc-hr-metrics.tsv: Extracted HR metrics
6. _desc-hr-metrics.json: Metrics metadata and descriptions  
7. _desc-hr-summary.json: Processing summary and quality assessment

Output structure:
derivatives/preprocessing/
├── sub-{subject}/
│   ├── ses-{session}/
│   │   ├── hr/
│   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_desc-processed_recording-hr.tsv
│   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_desc-processed_recording-hr.json
│   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_events.tsv
│   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_events.json
│   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_desc-hr-metrics.tsv
│   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_desc-hr-metrics.json
│   │   │   └── sub-{subject}_ses-{session}_task-{moment}_desc-hr-summary.json

Changes from original:
- Inherits from PhysioBIDSWriter base class
- Files are now PER MOMENT (restingstate, therapy) instead of combined
- Columns renamed: hr → HR_Clean, quality → HR_Quality, + HR_Raw added
- Files are UNCOMPRESSED (.tsv instead of .tsv.gz)
- Unified API with save_processed_data() method

**Constructor**:

```python
HRBIDSWriter(self, config_path: Union[str, pathlib.Path, NoneType] = None)
```

**Methods**:

##### `save_processed_data(self, subject_id: str, session_id: str, processed_results: Dict[str, pandas.core.frame.DataFrame], session_metrics: Optional[pandas.core.frame.DataFrame] = None, processing_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, List[pathlib.Path]]`

Write complete HR processing results in BIDS format.

Args:
    subject_id: Subject identifier WITH prefix (e.g., 'sub-f01p01')
    session_id: Session identifier WITH prefix (e.g., 'ses-01')
    processed_results: Dictionary mapping moment names to processed DataFrames
                     Expected columns: time, HR_Raw, HR_Clean, HR_Quality, 
                                      HR_Outliers, HR_Interpolated
    session_metrics: DataFrame with session-level metrics (optional)
    processing_metadata: Additional processing metadata (optional)

Returns:
    Dictionary mapping file types to lists of paths (one per moment)

Example:
    >>> writer = HRBIDSWriter()
    >>> processed_results = {
    ...     'restingstate': df_resting,
    ...     'therapy': df_therapy
    ... }
    >>> file_paths = writer.save_processed_data(
    ...     'sub-f01p01', 'ses-01', processed_results, metrics, metadata
    ... )
    >>> print(f"Physio files: {file_paths['physio']}")


---

## Epoching

#### Class: `EpochAssigner`

**Module**: `src.physio.epoching.epoch_assigner`

Assigns epoch IDs to physiological time series data using multiple methods.

This class segments continuous physiological signals into discrete time windows (epochs)
for analysis. Supports three epoching methods: fixed windows with overlap, n-split
(equal division), and sliding windows. Epoch IDs are stored as JSON-formatted lists
(e.g., `"[0]"`, `"[0, 1, 2]"`) to handle overlapping epochs.

**Constructor**:

```python
EpochAssigner(self, config: Dict[str, Any])
```

**Parameters**:
- `config`: Configuration dictionary containing epoching settings

**Methods**:

##### `assign_fixed_epochs(self, time: np.ndarray, duration: float, overlap: float, min_ratio: float = 0.5) -> List[str]`

Assign samples to fixed-duration epochs with overlap.

Creates non-overlapping windows with a specified overlap. Samples can belong to
1-2 epochs depending on their position relative to window boundaries.

Args:
    time: Array of timestamps in seconds.
    duration: Epoch duration in seconds (e.g., 30).
    overlap: Overlap between epochs in seconds (e.g., 5).
    min_ratio: Minimum ratio of sample presence in epoch (default: 0.5).

Returns:
    List of JSON-formatted epoch ID lists (e.g., `["[0]", "[0, 1]", "[1]"]`).

Example:
    - duration=30s, overlap=5s → step=25s
    - Sample at t=27s belongs to epochs [0, 1]

##### `assign_nsplit_epochs(self, time: np.ndarray, n_epochs: int) -> np.ndarray`

Divide signal into n equal epochs (no overlap).

Each sample belongs to exactly one epoch. Useful for equal-duration analysis windows.

Args:
    time: Array of timestamps in seconds.
    n_epochs: Number of epochs to create (e.g., 120).

Returns:
    Array of epoch IDs (integers 0 to n_epochs-1).

##### `assign_all_epochs(self, df: pd.DataFrame, task: str, time_column: str = 'time') -> pd.DataFrame`

Assign all configured epoch columns based on task type.

Special case: restingstate task always assigns epoch ID `[0]` to all samples.
For therapy tasks, applies all enabled epoching methods from configuration.

Args:
    df: DataFrame with time series data.
    task: Task name ('restingstate' or 'therapy').
    time_column: Name of time column (default: 'time').

Returns:
    DataFrame with added epoch columns:
        - `epoch_fixed_duration{X}s_overlap{Y}s` (JSON list format)
        - `epoch_nsplit{N}` (JSON list format)
        - `epoch_sliding_duration{X}s_step{Y}s` (JSON list format)

---

#### Class: `EpochBIDSWriter`

**Module**: `src.physio.epoching.epoch_bids_writer`

BIDS-compliant file I/O for epoched physiological data.

Reads preprocessed physiological signals, assigns epoch IDs, and writes
BIDS-formatted output files with epoch columns. Handles file pattern matching,
task detection, and JSON sidecar generation.

**Constructor**:

```python
EpochBIDSWriter(self, config: Dict[str, Any])
```

**Parameters**:
- `config`: Configuration dictionary containing paths and epoching settings

**Methods**:

##### `detect_task(self, filename: str) -> str`

Detect task type from BIDS filename.

Args:
    filename: BIDS-formatted filename (e.g., `sub-f01p01_ses-01_task-therapy_physio.tsv`).

Returns:
    Task name ('therapy', 'restingstate', or 'unknown').

##### `should_epoch_file(self, filename: str) -> bool`

Check if file should be epoched based on include/exclude patterns.

Applies glob pattern matching from configuration to determine which files
to process. Typically includes RR intervals and processed_recording files,
excludes events and metrics files.

Args:
    filename: Filename to check.

Returns:
    True if file matches include patterns and not exclude patterns.

##### `process_file(self, input_path: Path, subject: str, session: str, modality: str) -> Optional[Path]`

Process a single file: load, assign epochs, write output.

Args:
    input_path: Path to input TSV file.
    subject: Subject ID (e.g., 'f01p01').
    session: Session ID (e.g., '01').
    modality: Modality type ('bvp', 'eda', 'hr').

Returns:
    Path to output file if successful, None if failed.

##### `process_session(self, subject: str, session: str) -> Tuple[int, int, int]`

Process all eligible files for one session.

Args:
    subject: Subject ID.
    session: Session ID.

Returns:
    Tuple of (files_processed, files_skipped, files_failed).

**Epoch Column Format**:

All epoch columns use JSON list format for consistency:
- Single epoch: `"[0]"`
- Multiple epochs: `"[0, 1, 2, 3, 4]"`

Parsing example:
```python
import ast
epoch_ids = ast.literal_eval(df['epoch_sliding_duration30s_step5s'].iloc[0])
# Result: [0, 1, 2, 3, 4] (Python list)
```

**Column Naming Convention**:
- Fixed windows: `epoch_fixed_duration{duration}s_overlap{overlap}s`
- N-split: `epoch_nsplit{n_epochs}`
- Sliding windows: `epoch_sliding_duration{duration}s_step{step}s`

---

## Version History

### v1.1.0 (2025-11-11)

**Epoching Module Release**:
- ✅ **EpochAssigner**: Three epoching methods (fixed, nsplit, sliding)
- ✅ **EpochBIDSWriter**: BIDS-compliant file I/O with pattern matching
- ✅ **Multi-epoch Support**: Samples can belong to multiple overlapping epochs
- ✅ **JSON Format**: Epoch IDs stored as `"[0, 1, 2]"` for type consistency
- ✅ **Restingstate Rule**: All restingstate samples assigned to epoch `[0]`
- ✅ **Batch Processing**: 401 files processed across 51 sessions (100% success)
- ✅ **Testing**: 12 new tests, 55 total tests passing (100%)
- ✅ **Configuration**: step=5s for sliding windows (3.9% epoch gaps)

### v1.0.0 (2025-11-11)

**Production Release - Phase 2 Harmonization Complete**:
- ✅ **Code Harmonization**: All BIDS writers use identical patterns
- ✅ **HR Format Update**: Changed to per-moment uncompressed files (14 files/session)
- ✅ **Visualization Fix**: Integrated HR data into visualization pipeline
- ✅ **Production Validation**: 
  - 34/34 unit tests passing (100%)
  - 49/51 preprocessing sessions (96% success)
  - 306/306 visualizations generated (100% success)
- ✅ **Performance**: ~3 min preprocessing + ~3 min visualization per full dataset
- ✅ **Quality Tracking**: 114 quality flags across all modalities
- ✅ **Documentation**: Complete update to v1.0.0 production status

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
- ✅ CLI scripts with comprehensive testing

---

## Support & Contribution

**Authors**: Lena Adel, Remy Ramadour  
**Repository**: https://github.com/Ramdam17/TherasyncAnalysis  
**License**: MIT

For issues, questions, or contributions, please open an issue on GitHub.

---

**Generated from source code**: November 11, 2025  
**Accuracy**: All signatures, parameters, and docstrings extracted directly from implementation
