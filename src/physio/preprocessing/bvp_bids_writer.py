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
from src.physio.preprocessing.base_bids_writer import PhysioBIDSWriter


logger = logging.getLogger(__name__)


class BVPBIDSWriter(PhysioBIDSWriter):
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
        super().__init__(config_path)
        logger.info(f"BVP BIDS Writer initialized (modality: bvp, output: {self.derivatives_base}/{self.preprocessing_dir}/)")
    
    def _get_modality_name(self) -> str:
        """Get the modality identifier for BVP."""
        return 'bvp'
    
    def save_processed_data(
        self,
        subject_id: str,
        session_id: str,
        processed_results: Dict[str, pd.DataFrame],
        session_metrics: Optional[pd.DataFrame] = None,
        processing_metadata: Optional[Dict] = None
    ) -> Dict[str, List[Path]]:
        """
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
        """
        # Ensure IDs have prefixes
        subject_id = self._ensure_prefix(subject_id, 'sub')
        session_id = self._ensure_prefix(session_id, 'ses')
        
        created_files: Dict[str, List[Path]] = {
            'processed_signals': [],
            'metrics': [],
            'metadata': [],
            'summary': []
        }
        
        # Get subject/session directory
        subject_dir = self._get_subject_session_dir(subject_id, session_id)
        
        logger.info(f"Saving processed BVP data for {subject_id}/{session_id} ({len(processed_results)} moments)")
        
        # Save processed signals for each moment
        for moment, signals_data in processed_results.items():
            # Handle both formats: DataFrame or Tuple[DataFrame, Dict]
            if isinstance(signals_data, tuple):
                processed_signals, processing_info = signals_data
            else:
                processed_signals = signals_data
                # Get moment-specific processing info from metadata
                processing_info = {}
                if processing_metadata and moment in processing_metadata:
                    processing_info = processing_metadata[moment]
            
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
        if session_metrics is not None:
            # Handle both Dict and DataFrame formats
            if isinstance(session_metrics, pd.DataFrame):
                # Convert DataFrame to dict format
                metrics_dict = session_metrics.to_dict(orient='index')
            else:
                # Already a dict
                metrics_dict = session_metrics
            
            metrics_files = self._save_session_metrics(
                subject_dir, subject_id, session_id, metrics_dict
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
    ) -> List[Path]:
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
        created_files.append(signals_tsv)
        
        # Create JSON sidecar for processed signals
        signals_json = subject_dir / f"{base_filename}.json"
        signals_metadata = self._create_processed_signals_metadata(
            processing_info, processed_signals
        )
        
        # Use base class JSON serialization
        self._save_json_sidecar(signals_json, signals_metadata)
        created_files.append(signals_json)
        
        logger.debug(f"Saved processed signals: {signals_tsv}")
        
        return created_files
    
    def _save_session_metrics(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        session_metrics: Dict[str, Dict[str, float]]
    ) -> List[Path]:
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
        base_filename = f"{subject_id}_{session_id}_desc-bvp-metrics_physio"
        
        # Convert metrics to DataFrame
        metrics_df = pd.DataFrame.from_dict(session_metrics, orient='index')
        metrics_df.index.name = 'moment'
        metrics_df = metrics_df.reset_index()
        
        # Save metrics as TSV
        metrics_tsv = subject_dir / f"{base_filename}.tsv"
        metrics_df.to_csv(metrics_tsv, sep='\t', index=False, na_rep='n/a')
        created_files.append(metrics_tsv)
        
        # Create JSON sidecar for metrics
        metrics_json = subject_dir / f"{base_filename}.json"
        metrics_metadata = self._create_metrics_metadata(session_metrics)
        
        # Use base class JSON serialization
        self._save_json_sidecar(metrics_json, metrics_metadata)
        created_files.append(metrics_json)
        
        logger.debug(f"Saved BVP metrics: {metrics_tsv}")
        
        return created_files
    
    def _save_moment_metadata(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        moment: str,
        processing_info: Dict
    ) -> Optional[Path]:
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
                "ProcessingPipeline": self.pipeline_name,
                "ProcessingVersion": self.pipeline_version,
                "ProcessingInfo": processing_info
            }
            
            # Use base class JSON serialization
            self._save_json_sidecar(metadata_file, metadata)
            
            logger.debug(f"Saved moment metadata: {metadata_file}")
            return metadata_file
            
        except Exception as e:
            logger.error(f"Failed to save moment metadata for {moment}: {e}")
            return None
    
    def _save_processing_summary(
        self,
        subject_dir: Path,
        subject_id: str,
        session_id: str,
        processed_results: Dict,
        session_metrics: Any,  # Can be Dict or DataFrame or None
        processing_metadata: Optional[Dict] = None
    ) -> Optional[Path]:
        """
        Save overall processing summary.
        
        Args:
            subject_dir: Subject directory path
            subject_id: Subject identifier
            session_id: Session identifier
            processed_results: Processed results dictionary
            session_metrics: Session metrics (Dict, DataFrame, or None)
            processing_metadata: Additional processing metadata
            
        Returns:
            Path to created summary file, or None if failed
        """
        try:
            # BIDS filename for summary
            summary_filename = f"{subject_id}_{session_id}_desc-bvp-summary_recording-bvp.json"
            summary_file = subject_dir / summary_filename
            
            # Calculate total signal duration
            total_duration = 0.0
            for moment, signals_data in processed_results.items():
                # Handle both DataFrame and Tuple[DataFrame, Dict] formats
                if isinstance(signals_data, tuple):
                    signals, info = signals_data
                    sampling_rate = info.get('sampling_rate', 64)
                else:
                    signals = signals_data
                    # Get sampling rate from metadata if available
                    sampling_rate = 64
                    if processing_metadata and moment in processing_metadata:
                        sampling_rate = processing_metadata[moment].get('sampling_rate', 64)
                
                total_duration += len(signals) / sampling_rate
            
            # Get number of metrics
            num_metrics = 0
            if session_metrics is not None:
                if isinstance(session_metrics, pd.DataFrame):
                    num_metrics = len(session_metrics.columns)
                elif isinstance(session_metrics, dict):
                    num_metrics = len(next(iter(session_metrics.values()), {}))
            
            # Create processing summary
            summary = {
                "SubjectID": subject_id,
                "SessionID": session_id,
                "ProcessingDate": datetime.now().isoformat(),
                "ProcessingPipeline": self.pipeline_name,
                "ProcessingVersion": self.pipeline_version,
                "MomentsProcessed": list(processed_results.keys()),
                "MetricsExtracted": num_metrics,
                "TotalSignalDuration": total_duration,
                "QualityAssessment": "good"  # Simplified for now
            }
            
            # Add custom metadata if provided
            if processing_metadata:
                summary["AdditionalMetadata"] = processing_metadata
            
            # Use base class JSON serialization
            self._save_json_sidecar(summary_file, summary)
            
            logger.debug(f"Saved processing summary: {summary_file}")
            return summary_file
            
        except Exception as e:
            logger.error(f"Failed to save processing summary: {e}")
            import traceback
            logger.debug(traceback.format_exc())
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
        """Create metadata for metrics file."""
        return {
            "Description": "Extracted BVP metrics from processed signals",
            "ProcessingPipeline": self.pipeline_name,
            "ProcessingVersion": self.pipeline_version,
            "Metrics": {
                "BVP_NumPeaks": "Number of detected PPG peaks",
                "BVP_PeakRate": "Average peak rate (peaks per minute)",
                "HRV_RMSSD": "Root mean square of successive differences",
                "HRV_SDNN": "Standard deviation of NN intervals"
            }
        }
    
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
        
        # Save to preprocessing directory
        preprocessing_base = self.derivatives_base / self.preprocessing_dir
        group_file = preprocessing_base / output_filename
        group_df.to_csv(group_file, sep='\t', index=False, na_rep='n/a')
        
        # Create accompanying JSON
        group_json = preprocessing_base / f"{output_filename.replace('.tsv', '.json')}"
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