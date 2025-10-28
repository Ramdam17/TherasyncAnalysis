"""
BVP BIDS Writer for TherasyncPipeline.

This module provides functionality to save processed BVP data and extracted metrics
in BIDS-compliant format under data/derivatives/.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime

import pandas as pd
import numpy as np

from src.core.config_loader import ConfigLoader
from src.core.bids_utils import BIDSUtils


logger = logging.getLogger(__name__)


class BVPBIDSWriter:
    """
    Save processed BVP data and metrics in BIDS-compliant format.
    
    This class handles saving processed signals, extracted metrics, and metadata
    following BIDS derivatives specifications for physiological data.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the BVP BIDS writer with configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = ConfigLoader(config_path)
        self.bids_utils = BIDSUtils()
        
        # Get paths and BIDS configuration
        self.derivatives_path = Path(self.config.get('paths.derivatives'))
        self.bids_config = self.config.get('bids', {})
        
        # Create derivatives directory structure
        self.pipeline_name = "therasync-bvp"
        self.pipeline_version = "1.0.0"
        self.pipeline_dir = self.derivatives_path / self.pipeline_name
        
        # Ensure derivatives directory exists
        self.pipeline_dir.mkdir(parents=True, exist_ok=True)
        
        # Create dataset description for derivatives
        self._create_dataset_description()
        
        logger.info(f"BVP BIDS Writer initialized: {self.pipeline_dir}")
    
    def save_processed_data(
        self,
        subject_id: str,
        session_id: str,
        processed_results: Dict[str, Tuple[pd.DataFrame, Dict]],
        session_metrics: Dict[str, Dict[str, float]],
        processing_metadata: Optional[Dict] = None
    ) -> Dict[str, List[str]]:
        """
        Save processed BVP data and metrics in BIDS format.
        
        Args:
            subject_id: Subject identifier (e.g., 'sub-f01p01')
            session_id: Session identifier (e.g., 'ses-01')
            processed_results: Output from BVPCleaner.process_moment_signals()
            session_metrics: Output from BVPMetricsExtractor.extract_session_metrics()
            processing_metadata: Additional metadata about processing
            
        Returns:
            Dictionary with lists of created file paths
        """
        created_files = {
            'processed_signals': [],
            'metrics': [],
            'metadata': [],
            'summary': []
        }
        
        # Create subject/session directory
        subject_dir = self.pipeline_dir / subject_id / session_id / 'physio'
        subject_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving processed BVP data for {subject_id}/{session_id}")
        
        # Save processed signals for each moment
        for moment, (processed_signals, processing_info) in processed_results.items():
            # Save processed signals
            signal_files = self._save_processed_signals(
                subject_dir, subject_id, session_id, moment, 
                processed_signals, processing_info
            )
            created_files['processed_signals'].extend(signal_files)
            
            # Save moment-specific metadata
            metadata_file = self._save_moment_metadata(
                subject_dir, subject_id, session_id, moment, processing_info
            )
            if metadata_file:
                created_files['metadata'].append(metadata_file)
        
        # Save extracted metrics
        metrics_files = self._save_session_metrics(
            subject_dir, subject_id, session_id, session_metrics
        )
        created_files['metrics'].extend(metrics_files)
        
        # Save processing summary
        summary_file = self._save_processing_summary(
            subject_dir, subject_id, session_id, 
            processed_results, session_metrics, processing_metadata
        )
        if summary_file:
            created_files['summary'].append(summary_file)
        
        total_files = sum(len(files) for files in created_files.values())
        logger.info(f"Created {total_files} BIDS-compliant files for {subject_id}/{session_id}")
        
        return created_files
    
    def _save_processed_signals(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        moment: str,
        processed_signals: pd.DataFrame,
        processing_info: Dict
    ) -> List[str]:
        """
        Save processed BVP signals in BIDS format.
        
        Args:
            subject_dir: Subject directory path
            subject_id: Subject identifier
            session_id: Session identifier
            moment: Moment/task name
            processed_signals: Processed signals DataFrame
            processing_info: Processing information
            
        Returns:
            List of created file paths
        """
        created_files = []
        
        # BIDS filename pattern for processed physio data
        base_filename = f"{subject_id}_{session_id}_task-{moment}_desc-processed_recording-bvp"
        
        # Save processed signals as TSV
        signals_tsv = subject_dir / f"{base_filename}.tsv"
        
        # Prepare signals data for saving
        output_data = processed_signals.copy()
        
        # Add time column if not present
        if 'time' not in output_data.columns:
            sampling_rate = processing_info.get('sampling_rate', 64)
            time_values = np.arange(len(output_data)) / sampling_rate
            output_data.insert(0, 'time', time_values)
        
        # Save TSV file
        output_data.to_csv(signals_tsv, sep='\t', index=False, na_rep='n/a')
        created_files.append(str(signals_tsv))
        
        # Create JSON sidecar for processed signals
        signals_json = subject_dir / f"{base_filename}.json"
        signals_metadata = self._create_processed_signals_metadata(
            processing_info, processed_signals
        )
        
        with open(signals_json, 'w') as f:
            json.dump(signals_metadata, f, indent=2, default=self._json_serializer)
        created_files.append(str(signals_json))
        
        logger.debug(f"Saved processed signals: {signals_tsv}")
        
        return created_files
    
    def _save_session_metrics(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        session_metrics: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """
        Save extracted BVP metrics in BIDS format.
        
        Args:
            subject_dir: Subject directory path
            subject_id: Subject identifier
            session_id: Session identifier
            session_metrics: Extracted metrics dictionary
            
        Returns:
            List of created file paths
        """
        created_files = []
        
        if not session_metrics:
            logger.warning("No session metrics to save")
            return created_files
        
        # BIDS filename for metrics
        base_filename = f"{subject_id}_{session_id}_desc-bvpmetrics_physio"
        
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame.from_dict(session_metrics, orient='index')
        metrics_df.index.name = 'moment'
        metrics_df = metrics_df.reset_index()
        
        # Save metrics as TSV
        metrics_tsv = subject_dir / f"{base_filename}.tsv"
        metrics_df.to_csv(metrics_tsv, sep='\t', index=False, na_rep='n/a')
        created_files.append(str(metrics_tsv))
        
        # Create JSON sidecar for metrics
        metrics_json = subject_dir / f"{base_filename}.json"
        metrics_metadata = self._create_metrics_metadata(session_metrics)
        
        with open(metrics_json, 'w') as f:
            json.dump(metrics_metadata, f, indent=2, default=self._json_serializer)
        created_files.append(str(metrics_json))
        
        logger.debug(f"Saved BVP metrics: {metrics_tsv}")
        
        return created_files
    
    def _save_moment_metadata(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        moment: str,
        processing_info: Dict
    ) -> Optional[str]:
        """
        Save moment-specific processing metadata.
        
        Args:
            subject_dir: Subject directory path
            subject_id: Subject identifier
            session_id: Session identifier
            moment: Moment/task name
            processing_info: Processing information
            
        Returns:
            Path to created metadata file, or None if failed
        """
        try:
            # BIDS filename for moment metadata
            metadata_filename = f"{subject_id}_{session_id}_task-{moment}_desc-processing_recording-bvp.json"
            metadata_file = subject_dir / metadata_filename
            
            # Create comprehensive metadata
            metadata = {
                "TaskName": moment,
                "ProcessingMethod": processing_info.get('processing_method', 'elgendi'),
                "QualityMethod": processing_info.get('quality_method', 'templatematch'),
                "SamplingFrequency": processing_info.get('sampling_rate', 64),
                "NumberOfPeaks": len(processing_info.get('PPG_Peaks', [])),
                "ProcessingTimestamp": datetime.now().isoformat(),
                "ProcessingPipeline": "therasync-bvp",
                "ProcessingVersion": self.pipeline_version,
                "ProcessingInfo": processing_info
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2, default=self._json_serializer)
            
            logger.debug(f"Saved moment metadata: {metadata_file}")
            return str(metadata_file)
            
        except Exception as e:
            logger.error(f"Failed to save moment metadata for {moment}: {e}")
            return None
    
    def _save_processing_summary(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        processed_results: Dict,
        session_metrics: Dict,
        processing_metadata: Optional[Dict] = None
    ) -> Optional[str]:
        """
        Save overall processing summary.
        
        Args:
            subject_dir: Subject directory path
            subject_id: Subject identifier
            session_id: Session identifier
            processed_results: Processed results dictionary
            session_metrics: Session metrics dictionary
            processing_metadata: Additional processing metadata
            
        Returns:
            Path to created summary file, or None if failed
        """
        try:
            # BIDS filename for summary
            summary_filename = f"{subject_id}_{session_id}_desc-summary_recording-bvp.json"
            summary_file = subject_dir / summary_filename
            
            # Create processing summary
            summary = {
                "SubjectID": subject_id,
                "SessionID": session_id,
                "ProcessingDate": datetime.now().isoformat(),
                "ProcessingPipeline": "therasync-bvp",
                "ProcessingVersion": self.pipeline_version,
                "MomentsProcessed": list(processed_results.keys()),
                "MetricsExtracted": len(next(iter(session_metrics.values()), {})),
                "TotalSignalDuration": sum(
                    len(signals) / info.get('sampling_rate', 64)
                    for signals, info in processed_results.values()
                ),
                "QualityAssessment": self._assess_overall_quality(processed_results, session_metrics)
            }
            
            # Add custom metadata if provided
            if processing_metadata:
                summary["AdditionalMetadata"] = processing_metadata
            
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=self._json_serializer)
            
            logger.debug(f"Saved processing summary: {summary_file}")
            return str(summary_file)
            
        except Exception as e:
            logger.error(f"Failed to save processing summary: {e}")
            return None
    
    def _create_processed_signals_metadata(
        self,
        processing_info: Dict,
        processed_signals: pd.DataFrame
    ) -> Dict:
        """Create metadata for processed signals."""
        metadata = {
            "Description": "Processed BVP signals from TherasyncPipeline",
            "SamplingFrequency": processing_info.get('sampling_rate', 64),
            "StartTime": 0,
            "ProcessingMethod": processing_info.get('processing_method', 'elgendi'),
            "QualityMethod": processing_info.get('quality_method', 'templatematch'),
            "Columns": list(processed_signals.columns),
            "Units": {
                "time": "s",
                "PPG_Clean": "AU",
                "PPG_Rate": "BPM"
            },
            "ProcessingPipeline": "therasync-bvp",
            "ProcessingVersion": self.pipeline_version
        }
        
        # Add quality information if available
        if 'PPG_Quality' in processed_signals.columns:
            metadata["Units"]["PPG_Quality"] = "score"
        
        return metadata
    
    def _create_metrics_metadata(self, session_metrics: Dict) -> Dict:
        """Create metadata for extracted metrics."""
        # Get all unique metric names
        all_metrics = set()
        for moment_metrics in session_metrics.values():
            all_metrics.update(moment_metrics.keys())
        
        metadata = {
            "Description": "BVP-derived HRV and cardiovascular metrics from TherasyncPipeline",
            "ProcessingPipeline": "therasync-bvp",
            "ProcessingVersion": self.pipeline_version,
            "MetricsExtracted": sorted(list(all_metrics)),
            "Columns": {
                "moment": "Task/moment identifier",
            },
            "Units": {
                "moment": "categorical",
                # Time-domain metrics (ms)
                "HRV_MeanNN": "ms",
                "HRV_SDNN": "ms", 
                "HRV_RMSSD": "ms",
                "HRV_CVNN": "ratio",
                "HRV_pNN50": "percentage",
                # Frequency-domain metrics
                "HRV_LF": "ms²",
                "HRV_HF": "ms²",
                "HRV_TP": "ms²",
                "HRV_LFHF": "ratio",
                # Non-linear metrics
                "HRV_SD1": "ms",
                "HRV_SD2": "ms",
                "HRV_SampEn": "dimensionless",
                # Quality metrics
                "BVP_NumPeaks": "count",
                "BVP_Duration": "s",
                "BVP_PeakRate": "BPM",
                "BVP_MeanQuality": "score",
                "BVP_QualityStd": "score",
                "BVP_MeanAmplitude": "AU",
                "BVP_StdAmplitude": "AU",
                "BVP_RangeAmplitude": "AU"
            }
        }
        
        return metadata
    
    def _assess_overall_quality(
        self,
        processed_results: Dict,
        session_metrics: Dict
    ) -> Dict:
        """Assess overall quality of processing."""
        quality_assessment = {
            "moments_processed": len(processed_results),
            "moments_with_metrics": len([m for m in session_metrics.values() if not all(np.isnan(list(m.values())))]),
            "total_peaks_detected": sum(
                len(info.get('PPG_Peaks', []))
                for _, info in processed_results.values()
            ),
            "mean_peak_rate": np.mean([
                metrics.get('BVP_PeakRate', np.nan)
                for metrics in session_metrics.values()
                if not np.isnan(metrics.get('BVP_PeakRate', np.nan))
            ]) if session_metrics else np.nan
        }
        
        # Convert NaN to None for JSON serialization
        for key, value in quality_assessment.items():
            if isinstance(value, float) and np.isnan(value):
                quality_assessment[key] = None
        
        return quality_assessment
    
    def _create_dataset_description(self) -> None:
        """Create BIDS dataset_description.json for derivatives."""
        dataset_desc_file = self.pipeline_dir / "dataset_description.json"
        
        if dataset_desc_file.exists():
            return  # Already exists
        
        dataset_description = {
            "Name": "TherasyncPipeline BVP Processing",
            "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{
                "Name": "TherasyncPipeline",
                "Version": self.pipeline_version,
                "Description": "BVP processing pipeline for family therapy physiological data",
                "CodeURL": "https://github.com/ppsp-team/TherasyncPipeline"
            }],
            "SourceDatasets": [{
                "Description": "Therasync family therapy physiological data"
            }],
            "HowToAcknowledge": "Please cite the TherasyncPipeline paper when using this processed data."
        }
        
        with open(dataset_desc_file, 'w') as f:
            json.dump(dataset_description, f, indent=2)
        
        logger.info(f"Created dataset description: {dataset_desc_file}")
    
    def _json_serializer(self, obj):
        """JSON serializer for numpy types."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def create_group_summary(
        self,
        subjects_data: Dict[str, Dict[str, Dict[str, float]]],
        output_filename: str = "group_bvp_metrics.tsv"
    ) -> str:
        """
        Create group-level summary of BVP metrics across subjects.
        
        Args:
            subjects_data: Nested dict {subject_id: {session_id: session_metrics}}
            output_filename: Name of output file
            
        Returns:
            Path to created group summary file
        """
        group_data = []
        
        for subject_id, sessions in subjects_data.items():
            for session_id, session_metrics in sessions.items():
                for moment, metrics in session_metrics.items():
                    row = {
                        'subject_id': subject_id,
                        'session_id': session_id,
                        'moment': moment
                    }
                    # Add metrics to row
                    if isinstance(metrics, dict):
                        row.update(metrics)
                    group_data.append(row)
        
        # Create DataFrame and save
        group_df = pd.DataFrame(group_data)
        group_file = self.pipeline_dir / output_filename
        group_df.to_csv(group_file, sep='\t', index=False, na_rep='n/a')
        
        # Create accompanying JSON
        group_json = self.pipeline_dir / f"{output_filename.replace('.tsv', '.json')}"
        group_metadata = {
            "Description": "Group-level BVP metrics summary",
            "ProcessingPipeline": "therasync-bvp",
            "ProcessingVersion": self.pipeline_version,
            "NumberOfSubjects": len(subjects_data),
            "TotalSessions": sum(len(sessions) for sessions in subjects_data.values()),
            "CreationDate": datetime.now().isoformat()
        }
        
        with open(group_json, 'w') as f:
            json.dump(group_metadata, f, indent=2)
        
        logger.info(f"Created group summary: {group_file}")
        return str(group_file)