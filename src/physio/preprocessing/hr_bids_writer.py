"""
HR BIDS Writer for TherasyncPipeline.

This module writes HR processing results to BIDS-compliant output format,
creating standardized files for processed signals, metrics, and metadata.

Authors: Lena Adel, Remy Ramadour
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import pandas as pd
import numpy as np
import json

from src.core.config_loader import ConfigLoader
from src.physio.preprocessing.base_bids_writer import PhysioBIDSWriter


logger = logging.getLogger(__name__)


class HRBIDSWriter(PhysioBIDSWriter):
    """
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
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the HR BIDS writer.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        super().__init__(config_path)
        logger.info(f"HR BIDS Writer initialized (modality: hr, output: {self.derivatives_base}/{self.preprocessing_dir}/)")
    
    def _get_modality_name(self) -> str:
        """Get the modality identifier for HR."""
        return 'hr'
    
    def save_processed_data(
        self,
        subject_id: str,
        session_id: str,
        processed_results: Dict[str, pd.DataFrame],
        session_metrics: Optional[pd.DataFrame] = None,
        processing_metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, List[Path]]:
        """
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
        """
        # Ensure IDs have prefixes
        subject_id = self._ensure_prefix(subject_id, 'sub')
        session_id = self._ensure_prefix(session_id, 'ses')
        
        logger.info(f"Writing HR results for {subject_id} {session_id} ({len(processed_results)} moments)")
        
        # Get subject/session directory
        output_dir = self._get_subject_session_dir(subject_id, session_id)
        
        # Initialize file paths dictionary
        all_file_paths: Dict[str, List[Path]] = {
            'physio': [],
            'physio_json': [],
            'events': [],
            'events_json': [],
            'metrics': [],
            'metrics_json': [],
            'summary': []
        }
        
        try:
            # Process each moment separately
            for moment, moment_data in processed_results.items():
                logger.debug(f"Processing moment: {moment}")
                
                # Generate BIDS filename prefix for this moment
                prefix = f"{subject_id}_{session_id}_task-{moment}"
                
                # Extract moment-specific metadata
                moment_metadata = processing_metadata.get(moment, {}) if processing_metadata else {}
                
                # Write physio signal files
                physio_file = self._write_physio_file(output_dir, prefix, moment_data)
                all_file_paths['physio'].append(physio_file)
                
                physio_json = self._write_physio_metadata(output_dir, prefix, moment_data, moment_metadata)
                all_file_paths['physio_json'].append(physio_json)
                
                # Write events files
                events_file = self._write_events_file(output_dir, prefix, moment_data)
                all_file_paths['events'].append(events_file)
                
                events_json = self._write_events_metadata(output_dir, prefix)
                all_file_paths['events_json'].append(events_json)
                
                # Extract moment-specific metrics if session_metrics provided
                if session_metrics is not None and moment in session_metrics.index:
                    moment_metrics = session_metrics.loc[moment].to_dict()
                else:
                    # Fallback: extract basic metrics from data
                    moment_metrics = self._extract_basic_metrics(moment_data, moment)
                
                # Write metrics files
                metrics_file = self._write_metrics_file(output_dir, prefix, moment_metrics)
                all_file_paths['metrics'].append(metrics_file)
                
                metrics_json = self._write_metrics_metadata(output_dir, prefix, moment_metrics)
                all_file_paths['metrics_json'].append(metrics_json)
                
                # Write summary file
                summary_file = self._write_summary_file(
                    output_dir, prefix, moment_metrics, moment_metadata,
                    {
                        'physio': physio_file,
                        'physio_json': physio_json,
                        'events': events_file,
                        'events_json': events_json,
                        'metrics': metrics_file,
                        'metrics_json': metrics_json
                    }
                )
                all_file_paths['summary'].append(summary_file)
            
            total_files = sum(len(paths) for paths in all_file_paths.values())
            logger.info(f"HR results written successfully ({total_files} files across {len(processed_results)} moments)")
            return all_file_paths
            
        except Exception as e:
            logger.error(f"Failed to write HR results: {str(e)}")
            raise
    
    def _extract_basic_metrics(self, data: pd.DataFrame, moment: str) -> Dict[str, Any]:
        """
        Extract basic HR metrics from processed data.
        
        Args:
            data: Processed HR DataFrame
            moment: Moment identifier
        
        Returns:
            Dictionary of basic metrics
        """
        hr_clean = data['HR_Clean'].dropna()
        
        if len(hr_clean) == 0:
            return {
                'moment': moment,
                'hr_mean': np.nan,
                'hr_std': np.nan,
                'hr_min': np.nan,
                'hr_max': np.nan,
                'hr_range': np.nan
            }
        
        return {
            'moment': moment,
            'hr_mean': float(hr_clean.mean()),
            'hr_std': float(hr_clean.std()),
            'hr_min': float(hr_clean.min()),
            'hr_max': float(hr_clean.max()),
            'hr_range': float(hr_clean.max() - hr_clean.min())
        }
    
    def _write_physio_file(
        self,
        output_dir: Path,
        prefix: str,
        data: pd.DataFrame
    ) -> Path:
        """
        Write processed HR signals to UNCOMPRESSED TSV file.
        
        Args:
            output_dir: Output directory
            prefix: BIDS filename prefix
            data: Cleaned HR data with columns: time, HR_Raw, HR_Clean, HR_Quality, 
                  HR_Outliers, HR_Interpolated
        
        Returns:
            Path to written file
        """
        # New filename convention: _desc-processed_recording-hr.tsv (UNCOMPRESSED)
        file_path = output_dir / f"{prefix}_desc-processed_recording-hr.tsv"
        
        # Select columns in standardized order
        # Expected input columns: time, HR_Raw, HR_Clean, HR_Quality, HR_Outliers, HR_Interpolated
        output_columns = ['time', 'HR_Raw', 'HR_Clean', 'HR_Quality', 'HR_Outliers', 'HR_Interpolated']
        
        # Check if all expected columns exist
        missing_cols = [col for col in output_columns if col not in data.columns]
        if missing_cols:
            logger.warning(f"Missing expected columns: {missing_cols}. Using available columns.")
            output_columns = [col for col in output_columns if col in data.columns]
        
        output_data = data[output_columns].copy()
        
        # Convert boolean/flag columns to int
        if 'HR_Outliers' in output_data.columns:
            output_data['HR_Outliers'] = output_data['HR_Outliers'].astype(int)
        if 'HR_Interpolated' in output_data.columns:
            output_data['HR_Interpolated'] = output_data['HR_Interpolated'].astype(int)
        
        # Write UNCOMPRESSED TSV (no .gz)
        output_data.to_csv(file_path, sep='\t', index=False)
        
        logger.debug(f"Physio data written: {file_path} ({len(output_data)} samples, {len(output_columns)} columns)")
        return file_path
    
    def _write_physio_metadata(
        self,
        output_dir: Path,
        prefix: str,
        data: pd.DataFrame,
        cleaning_metadata: Dict[str, Any]
    ) -> Path:
        """
        Write physio signal metadata file.
        
        Args:
            output_dir: Output directory
            prefix: BIDS filename prefix
            data: Cleaned HR data
            cleaning_metadata: Cleaning process metadata
        
        Returns:
            Path to written file
        """
        # Match new filename convention
        file_path = output_dir / f"{prefix}_desc-processed_recording-hr.json"
        
        # Calculate signal characteristics
        sampling_rate = 1.0  # HR is typically 1 Hz
        duration = data['time'].iloc[-1] - data['time'].iloc[0] if len(data) > 0 else 0.0
        
        # Updated column names and descriptions
        metadata = {
            "TaskName": cleaning_metadata.get('moment', 'unknown'),
            "SamplingFrequency": sampling_rate,
            "StartTime": 0.0,
            "Columns": [
                "time",
                "HR_Raw",
                "HR_Clean", 
                "HR_Quality",
                "HR_Outliers",
                "HR_Interpolated"
            ],
            "Units": [
                "s",
                "BPM",
                "BPM",
                "a.u.",
                "n/a",
                "n/a"
            ],
            "Descriptions": [
                "Time in seconds from start of recording",
                "Raw heart rate in beats per minute (before cleaning)",
                "Cleaned heart rate in beats per minute (after outlier removal and interpolation)",
                "Quality score (0-1, 1=highest quality)",
                "Outlier flag (1=outlier removed, 0=valid)",
                "Interpolation flag (1=interpolated, 0=original)"
            ],
            "ProcessingMetadata": {
                "Pipeline": "TherasyncPipeline",
                "Version": "1.0.0",
                "ProcessingDate": datetime.now().isoformat(),
                "Duration": float(duration),
                "ValidSamples": int(cleaning_metadata.get('valid_samples', len(data))),
                "TotalSamples": int(cleaning_metadata.get('total_samples', len(data))),
                "QualityScore": float(cleaning_metadata.get('quality_score', 0)),
                "OutlierThreshold": cleaning_metadata.get('processing_parameters', {}).get('outlier_threshold_bpm', [40, 180]),
                "InterpolationMaxGap": cleaning_metadata.get('processing_parameters', {}).get('interpolation_max_gap_seconds', 5)
            }
        }
        
        # Use base class JSON serialization
        self._save_json_sidecar(file_path, metadata)
        
        logger.debug(f"Physio metadata written: {file_path}")
        return file_path
    
    def _write_events_file(
        self,
        output_dir: Path,
        prefix: str,
        data: pd.DataFrame
    ) -> Path:
        """
        Write HR events file (peaks, elevated periods, etc.).
        
        Args:
            output_dir: Output directory
            prefix: BIDS filename prefix
            data: Cleaned HR data with HR_Clean column
        
        Returns:
            Path to written file
        """
        file_path = output_dir / f"{prefix}_events.tsv"
        
        events = []
        
        # Use new column name HR_Clean
        if 'HR_Clean' in data.columns:
            hr_values = data['HR_Clean'].dropna()
            time_values = data.loc[hr_values.index, 'time']
            
            if len(hr_values) > 0:
                # Find HR peaks (local maxima)
                peaks = self._find_hr_peaks(hr_values.values, time_values.values)
                events.extend(peaks)
                
                # Find elevated periods (above baseline + 20%)
                baseline = np.mean(hr_values.iloc[:min(60, len(hr_values))])  # First minute as baseline
                elevated_periods = self._find_elevated_periods(
                    hr_values.values, time_values.values, baseline * 1.2
                )
                events.extend(elevated_periods)
        
        # Create events DataFrame
        if events:
            events_df = pd.DataFrame(events)
        else:
            # Empty events file with proper structure
            events_df = pd.DataFrame(columns=['onset', 'duration', 'trial_type', 'value'])
        
        # Write events TSV
        events_df.to_csv(file_path, sep='\t', index=False)
        
        logger.debug(f"Events written: {file_path} ({len(events_df)} events)")
        return file_path
        
        # Create events DataFrame
        if events:
            events_df = pd.DataFrame(events)
        else:
            # Empty events file with proper structure
            events_df = pd.DataFrame(columns=['onset', 'duration', 'trial_type', 'value'])
        
        # Write events TSV
        events_df.to_csv(file_path, sep='\t', index=False)
        
        logger.debug(f"Events written: {file_path} ({len(events_df)} events)")
        return file_path
    
    def _write_events_metadata(self, output_dir: Path, prefix: str) -> Path:
        """
        Write events metadata file.
        
        Args:
            output_dir: Output directory
            prefix: BIDS filename prefix
        
        Returns:
            Path to written file
        """
        file_path = output_dir / f"{prefix}_events.json"
        
        metadata = {
            "onset": {
                "Description": "Event onset time in seconds from start of recording",
                "Units": "s"
            },
            "duration": {
                "Description": "Event duration in seconds",
                "Units": "s"
            },
            "trial_type": {
                "Description": "Type of HR event",
                "Levels": {
                    "hr_peak": "Local maximum in HR signal",
                    "hr_elevated": "Period of elevated HR above baseline threshold"
                }
            },
            "value": {
                "Description": "HR value at event onset",
                "Units": "BPM"
            }
        }
        
        # Use base class JSON serialization
        self._save_json_sidecar(file_path, metadata)
        
        logger.debug(f"Events metadata written: {file_path}")
        return file_path
    
    def _write_metrics_file(
        self,
        output_dir: Path,
        prefix: str,
        metrics: Dict[str, Any]
    ) -> Path:
        """
        Write HR metrics to TSV file.
        
        Args:
            output_dir: Output directory
            prefix: BIDS filename prefix
            metrics: Extracted HR metrics (can be flat dict or nested with categories)
        
        Returns:
            Path to written file
        """
        # New filename convention
        file_path = output_dir / f"{prefix}_desc-hr-metrics.tsv"
        
        # Handle both flat and nested dictionaries
        flattened_metrics = {}
        
        # If metrics is already a flat dictionary (from _extract_basic_metrics)
        if all(not isinstance(v, dict) for v in metrics.values()):
            flattened_metrics = metrics.copy()
        else:
            # Flatten nested metrics dictionary (from HRMetricsExtractor)
            for category, category_metrics in metrics.items():
                if category in ['moment', 'summary']:
                    # Keep these at top level
                    if category == 'moment':
                        flattened_metrics['moment'] = category_metrics
                    continue
                if isinstance(category_metrics, dict):
                    for metric_name, value in category_metrics.items():
                        flattened_metrics[metric_name] = value
            
            # Add summary information if present
            if 'summary' in metrics:
                flattened_metrics['total_metrics'] = metrics['summary'].get('total_metrics_extracted', 0)
                flattened_metrics['quality_assessment'] = metrics['summary'].get('overall_quality_assessment', 'unknown')
        
        # Create single-row DataFrame
        metrics_df = pd.DataFrame([flattened_metrics])
        
        # Write metrics TSV
        metrics_df.to_csv(file_path, sep='\t', index=False)
        
        logger.debug(f"HR metrics written: {file_path} ({len(flattened_metrics)} metrics)")
        return file_path
    
    def _write_metrics_metadata(
        self,
        output_dir: Path,
        prefix: str,
        metrics: Dict[str, Any]
    ) -> Path:
        """
        Write metrics metadata file with descriptions.
        
        Args:
            output_dir: Output directory
            prefix: BIDS filename prefix
            metrics: Extracted HR metrics
        
        Returns:
            Path to written file
        """
        # New filename convention
        file_path = output_dir / f"{prefix}_desc-hr-metrics.json"
        
        # Try to import HRMetricsExtractor to get full descriptions
        try:
            from src.physio.preprocessing.hr_metrics_extractor import HRMetricsExtractor
            extractor = HRMetricsExtractor()
            descriptions = extractor.get_metrics_description()
        except ImportError:
            logger.warning("Could not import HRMetricsExtractor, using basic descriptions")
            descriptions = {}
        
        metadata = {
            "Description": "Heart Rate (HR) metrics extracted from cleaned HR signals",
            "MetricsCategories": {
                "descriptive": "Basic statistical measures of HR distribution",
                "trend": "Temporal trends and changes in HR over time", 
                "stability": "Measures of HR variability and stability",
                "response": "Physiological response patterns and dynamics",
                "contextual": "Recording context and data quality metrics"
            },
            "MetricsDescriptions": descriptions,
            "ProcessingInfo": {
                "Pipeline": "TherasyncPipeline",
                "Version": "1.0.0",
                "ProcessingDate": datetime.now().isoformat(),
                "TotalMetrics": metrics.get('summary', {}).get('total_metrics_extracted', len([k for k in metrics if k not in ['moment', 'summary']])),
                "QualityAssessment": metrics.get('summary', {}).get('overall_quality_assessment', 'unknown')
            },
            "Units": {
                "hr_*": "BPM (beats per minute) unless otherwise specified",
                "*_time": "seconds",
                "*_duration": "seconds", 
                "*_percent": "percentage",
                "*_quality": "quality score (0-1)",
                "*_samples": "number of samples"
            }
        }
        
        # Use base class JSON serialization
        self._save_json_sidecar(file_path, metadata)
        
        logger.debug(f"Metrics metadata written: {file_path}")
        return file_path
    
    def _write_summary_file(
        self,
        output_dir: Path,
        prefix: str,
        metrics: Dict[str, Any],
        cleaning_metadata: Dict[str, Any],
        file_paths: Dict[str, Path]
    ) -> Path:
        """
        Write processing summary file.
        
        Args:
            output_dir: Output directory
            prefix: BIDS filename prefix
            metrics: Extracted HR metrics
            cleaning_metadata: Cleaning process metadata
            file_paths: Dictionary of written file paths for this moment
        
        Returns:
            Path to written file
        """
        # New filename convention
        file_path = output_dir / f"{prefix}_desc-hr-summary.json"
        
        # Parse subject/session/task from prefix
        parts = prefix.split('_')
        subject = parts[0].replace('sub-', '') if len(parts) > 0 else 'unknown'
        session = parts[1].replace('ses-', '') if len(parts) > 1 else 'unknown'
        task = parts[2].replace('task-', '') if len(parts) > 2 else 'unknown'
        
        summary = {
            "ProcessingInfo": {
                "Pipeline": "TherasyncPipeline",
                "Version": "1.0.0",
                "ProcessingDate": datetime.now().isoformat(),
                "Subject": subject,
                "Session": session,
                "Task": task
            },
            "DataQuality": {
                "TotalSamples": int(cleaning_metadata.get('total_samples', 0)),
                "ValidSamples": int(cleaning_metadata.get('valid_samples', 0)),
                "DataCompleteness": float(cleaning_metadata.get('data_completeness', 0)),
                "QualityScore": float(cleaning_metadata.get('quality_score', 0)),
                "OutlierPercentage": float(cleaning_metadata.get('outlier_percentage', 0)),
                "InterpolatedPercentage": float(cleaning_metadata.get('interpolated_percentage', 0))
            },
            "MetricsSummary": {
                "TotalMetricsExtracted": metrics.get('summary', {}).get('total_metrics_extracted', len([k for k in metrics if k not in ['moment', 'summary']])),
                "QualityAssessment": metrics.get('summary', {}).get('overall_quality_assessment', 'unknown'),
                "DescriptiveMetrics": metrics.get('summary', {}).get('descriptive_count', 0),
                "TrendMetrics": metrics.get('summary', {}).get('trend_count', 0),
                "StabilityMetrics": metrics.get('summary', {}).get('stability_count', 0),
                "ResponseMetrics": metrics.get('summary', {}).get('response_count', 0),
                "ContextualMetrics": metrics.get('summary', {}).get('contextual_count', 0)
            },
            "KeyResults": {
                "MeanHR": metrics.get('descriptive', {}).get('hr_mean') if 'descriptive' in metrics else metrics.get('hr_mean'),
                "HRRange": metrics.get('descriptive', {}).get('hr_range') if 'descriptive' in metrics else metrics.get('hr_range'),
                "HRStability": metrics.get('stability', {}).get('hr_stability') if 'stability' in metrics else None,
                "Duration": metrics.get('contextual', {}).get('hr_duration') if 'contextual' in metrics else None
            },
            "OutputFiles": {
                "ProcessedSignals": str(file_paths.get('physio', '')),
                "Events": str(file_paths.get('events', '')),
                "Metrics": str(file_paths.get('metrics', '')),
                "Summary": str(file_path)
            }
        }
        
        # Use base class JSON serialization
        self._save_json_sidecar(file_path, summary)
        
        logger.debug(f"Processing summary written: {file_path}")
        return file_path
    
    def _find_hr_peaks(self, hr_values: np.ndarray, time_values: np.ndarray) -> List[Dict]:
        """
        Find HR peaks (local maxima) in the signal.
        
        Args:
            hr_values: Array of HR values
            time_values: Array of time values
        
        Returns:
            List of peak events
        """
        events = []
        
        if len(hr_values) < 3:
            return events
        
        # Simple peak detection (local maxima)
        for i in range(1, len(hr_values) - 1):
            if hr_values[i] > hr_values[i-1] and hr_values[i] > hr_values[i+1]:
                # Additional criteria: peak must be significantly above neighbors
                if hr_values[i] > max(hr_values[i-1], hr_values[i+1]) + 2:  # 2 BPM threshold
                    events.append({
                        'onset': float(time_values[i]),
                        'duration': 0.0,
                        'trial_type': 'hr_peak',
                        'value': float(hr_values[i])
                    })
        
        return events
    
    def _find_elevated_periods(
        self,
        hr_values: np.ndarray,
        time_values: np.ndarray,
        threshold: float
    ) -> List[Dict]:
        """
        Find periods of elevated HR above threshold.
        
        Args:
            hr_values: Array of HR values
            time_values: Array of time values
            threshold: HR threshold for elevated periods
        
        Returns:
            List of elevated period events
        """
        events = []
        
        if len(hr_values) == 0:
            return events
        
        # Find elevated samples
        elevated_mask = hr_values > threshold
        
        # Find consecutive elevated periods
        elevated_diff = np.diff(np.concatenate(([False], elevated_mask, [False])).astype(int))
        starts = np.where(elevated_diff == 1)[0]
        ends = np.where(elevated_diff == -1)[0]
        
        for start_idx, end_idx in zip(starts, ends):
            if end_idx > start_idx:  # Valid period
                onset_time = time_values[start_idx]
                end_time = time_values[min(end_idx, len(time_values) - 1)]
                duration = end_time - onset_time
                
                # Only include periods longer than 5 seconds
                if duration >= 5.0:
                    events.append({
                        'onset': float(onset_time),
                        'duration': float(duration),
                        'trial_type': 'hr_elevated',
                        'value': float(hr_values[start_idx])
                    })
        
        return events