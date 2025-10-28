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
from typing import Dict, Any, Optional, List

import pandas as pd
import numpy as np
import json

from src.core.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class HRBIDSWriter:
    """
    Write HR processing results in BIDS-compliant format.
    
    This class creates 7 file types following the BIDS specification:
    1. _physio.tsv.gz: Processed HR signals (compressed)
    2. _physio.json: Signal metadata and processing parameters
    3. _events.tsv: HR-related events (elevated periods, peaks, etc.)
    4. _events.json: Events metadata
    5. _hr-metrics.tsv: Extracted HR metrics
    6. _hr-metrics.json: Metrics metadata and descriptions  
    7. _hr-summary.json: Processing summary and quality assessment
    
    Output structure:
    derivatives/preprocessing/
    ├── sub-{subject}/
    │   ├── ses-{session}/
    │   │   ├── hr/
    │   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_physio.tsv.gz
    │   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_physio.json
    │   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_events.tsv
    │   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_events.json
    │   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_hr-metrics.tsv
    │   │   │   ├── sub-{subject}_ses-{session}_task-{moment}_hr-metrics.json
    │   │   │   └── sub-{subject}_ses-{session}_task-{moment}_hr-summary.json
    """
    
    def __init__(self, config: Optional[ConfigLoader] = None):
        """
        Initialize the HR BIDS writer.
        
        Args:
            config: ConfigLoader instance. If None, creates new instance.
        """
        self.config = config if config is not None else ConfigLoader()
        
        # Get output configuration
        derivatives_dir = Path(self.config.get('paths.derivatives', 'data/derivatives'))
        preprocessing_dir = self.config.get('output.preprocessing_dir', 'preprocessing')
        modality_subdir = self.config.get('output.modality_subdirs.hr', 'hr')
        
        # Store base directories
        self.derivatives_base = derivatives_dir
        self.preprocessing_dir = preprocessing_dir
        self.modality_subdir = modality_subdir
        
        logger.info(f"HR BIDS Writer initialized (output: {derivatives_dir}/{preprocessing_dir}/sub-{{subject}}/ses-{{session}}/{modality_subdir}/)")
    
    def write_hr_results(
        self,
        subject: str,
        session: str,
        moment: str,
        cleaned_data: pd.DataFrame,
        metrics: Dict[str, Any],
        cleaning_metadata: Dict[str, Any]
    ) -> Dict[str, Path]:
        """
        Write complete HR processing results in BIDS format.
        
        Args:
            subject: Subject identifier (e.g., 'f01p01')
            session: Session identifier (e.g., '01')
            moment: Moment/task identifier (e.g., 'restingstate', 'therapy')
            cleaned_data: DataFrame with processed HR data
            metrics: Extracted HR metrics dictionary
            cleaning_metadata: Cleaning process metadata
        
        Returns:
            Dictionary mapping file types to their paths
        
        Example:
            >>> writer = HRBIDSWriter()
            >>> file_paths = writer.write_hr_results(
            ...     'f01p01', '01', 'therapy', cleaned_data, metrics, metadata
            ... )
            >>> print(f"Physio file: {file_paths['physio']}")
        """
        logger.info(f"Writing HR results for {subject} ses-{session} task-{moment}")
        
        # Create subject/session directory structure
        physio_dir = self._create_subject_directory(subject, session)
        
        # Generate BIDS filename prefix
        prefix = f"sub-{subject}_ses-{session}_task-{moment}"
        
        # Write each file type
        file_paths = {}
        
        try:
            # 1. Processed signals
            file_paths['physio'] = self._write_physio_file(physio_dir, prefix, cleaned_data)
            file_paths['physio_json'] = self._write_physio_metadata(
                physio_dir, prefix, cleaned_data, cleaning_metadata
            )
            
            # 2. Events
            file_paths['events'] = self._write_events_file(physio_dir, prefix, cleaned_data)
            file_paths['events_json'] = self._write_events_metadata(physio_dir, prefix)
            
            # 3. Metrics
            file_paths['metrics'] = self._write_metrics_file(physio_dir, prefix, metrics)
            file_paths['metrics_json'] = self._write_metrics_metadata(physio_dir, prefix, metrics)
            
            # 4. Summary
            file_paths['summary'] = self._write_summary_file(
                physio_dir, prefix, metrics, cleaning_metadata, file_paths
            )
            
            logger.info(f"HR results written successfully ({len(file_paths)} files)")
            return file_paths
            
        except Exception as e:
            logger.error(f"Failed to write HR results: {str(e)}")
            raise
    
    def _create_subject_directory(self, subject: str, session: str) -> Path:
        """
        Create BIDS-compliant directory structure for subject/session.
        
        Args:
            subject: Subject identifier
            session: Session identifier
        
        Returns:
            Path to hr subdirectory
        """
        # New structure: derivatives/preprocessing/sub-xxx/ses-yyy/hr/
        hr_dir = (self.derivatives_base / self.preprocessing_dir / 
                  f"sub-{subject}" / f"ses-{session}" / self.modality_subdir)
        hr_dir.mkdir(parents=True, exist_ok=True)
        return hr_dir
    
    def _write_physio_file(
        self,
        output_dir: Path,
        prefix: str,
        data: pd.DataFrame
    ) -> Path:
        """
        Write processed HR signals to compressed TSV file.
        
        Args:
            output_dir: Output directory
            prefix: BIDS filename prefix
            data: Cleaned HR data
        
        Returns:
            Path to written file
        """
        file_path = output_dir / f"{prefix}_physio.tsv.gz"
        
        # Select and rename columns for output
        output_data = data[['time', 'hr_clean', 'hr_quality']].copy()
        output_data.columns = ['time', 'hr', 'quality']
        
        # Add processing flags as separate columns
        output_data['outlier'] = data['hr_outliers'].astype(int)
        output_data['interpolated'] = data['hr_interpolated'].astype(int)
        
        # Write compressed TSV
        output_data.to_csv(file_path, sep='\t', index=False, compression='gzip')
        
        logger.debug(f"Physio data written: {file_path} ({len(output_data)} samples)")
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
        file_path = output_dir / f"{prefix}_physio.json"
        
        # Calculate signal characteristics
        sampling_rate = 1.0  # HR is typically 1 Hz
        duration = data['time'].iloc[-1] - data['time'].iloc[0]
        
        metadata = {
            "TaskName": cleaning_metadata.get('moment', 'unknown'),
            "SamplingFrequency": sampling_rate,
            "StartTime": 0.0,
            "Columns": [
                "time",
                "hr", 
                "quality",
                "outlier",
                "interpolated"
            ],
            "Units": [
                "s",
                "BPM",
                "a.u.",
                "n/a",
                "n/a"
            ],
            "Descriptions": [
                "Time in seconds from start of recording",
                "Heart rate in beats per minute (cleaned)",
                "Quality score (0-1, 1=highest quality)",
                "Outlier flag (1=outlier removed, 0=valid)",
                "Interpolation flag (1=interpolated, 0=original)"
            ],
            "ProcessingMetadata": {
                "Pipeline": "TherasyncPipeline",
                "Version": "1.0.0",
                "ProcessingDate": datetime.now().isoformat(),
                "Duration": float(duration),
                "ValidSamples": int(cleaning_metadata.get('valid_samples', 0)),
                "TotalSamples": int(cleaning_metadata.get('total_samples', 0)),
                "QualityScore": float(cleaning_metadata.get('quality_score', 0)),
                "OutlierThreshold": cleaning_metadata.get('processing_parameters', {}).get('outlier_threshold_bpm', [40, 180]),
                "InterpolationMaxGap": cleaning_metadata.get('processing_parameters', {}).get('interpolation_max_gap_seconds', 5)
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
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
            data: Cleaned HR data
        
        Returns:
            Path to written file
        """
        file_path = output_dir / f"{prefix}_events.tsv"
        
        events = []
        hr_values = data['hr_clean'].dropna()
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
        
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
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
            metrics: Extracted HR metrics
        
        Returns:
            Path to written file
        """
        file_path = output_dir / f"{prefix}_hr-metrics.tsv"
        
        # Flatten metrics dictionary
        flattened_metrics = {}
        for category, category_metrics in metrics.items():
            if category in ['moment', 'summary']:
                continue
            if isinstance(category_metrics, dict):
                for metric_name, value in category_metrics.items():
                    flattened_metrics[metric_name] = value
        
        # Add summary information
        flattened_metrics['moment'] = metrics.get('moment', 'unknown')
        flattened_metrics['total_metrics'] = metrics.get('summary', {}).get('total_metrics_extracted', 0)
        flattened_metrics['quality_assessment'] = metrics.get('summary', {}).get('overall_quality_assessment', 'unknown')
        
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
        file_path = output_dir / f"{prefix}_hr-metrics.json"
        
        # Import HRMetricsExtractor to get descriptions
        from src.physio.hr_metrics_extractor import HRMetricsExtractor
        extractor = HRMetricsExtractor()
        descriptions = extractor.get_metrics_description()
        
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
                "TotalMetrics": metrics.get('summary', {}).get('total_metrics_extracted', 0),
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
        
        with open(file_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
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
            file_paths: Dictionary of written file paths
        
        Returns:
            Path to written file
        """
        file_path = output_dir / f"{prefix}_hr-summary.json"
        
        summary = {
            "ProcessingInfo": {
                "Pipeline": "TherasyncPipeline",
                "Version": "1.0.0",
                "ProcessingDate": datetime.now().isoformat(),
                "Subject": prefix.split('_')[0].replace('sub-', ''),
                "Session": prefix.split('_')[1].replace('ses-', ''),
                "Task": prefix.split('_')[2].replace('task-', '')
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
                "TotalMetricsExtracted": metrics.get('summary', {}).get('total_metrics_extracted', 0),
                "QualityAssessment": metrics.get('summary', {}).get('overall_quality_assessment', 'unknown'),
                "DescriptiveMetrics": metrics.get('summary', {}).get('descriptive_count', 0),
                "TrendMetrics": metrics.get('summary', {}).get('trend_count', 0),
                "StabilityMetrics": metrics.get('summary', {}).get('stability_count', 0),
                "ResponseMetrics": metrics.get('summary', {}).get('response_count', 0),
                "ContextualMetrics": metrics.get('summary', {}).get('contextual_count', 0)
            },
            "KeyResults": {
                "MeanHR": metrics.get('descriptive', {}).get('hr_mean'),
                "HRRange": metrics.get('descriptive', {}).get('hr_range'),
                "HRStability": metrics.get('stability', {}).get('hr_stability'),
                "Duration": metrics.get('contextual', {}).get('hr_duration')
            },
            "OutputFiles": {
                "ProcessedSignals": str(file_paths.get('physio', '')),
                "Events": str(file_paths.get('events', '')),
                "Metrics": str(file_paths.get('metrics', '')),
                "Summary": str(file_path)
            }
        }
        
        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
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