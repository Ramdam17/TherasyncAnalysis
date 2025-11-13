"""
BIDS writer for epoched physiological signals.

This module handles reading preprocessed BIDS files, adding epoch columns,
and writing to the epoched derivatives directory.

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import fnmatch
import logging
import json
import shutil
import pandas as pd
from pathlib import Path
from typing import Optional, Union, List, Dict

from src.core.config_loader import ConfigLoader
from src.physio.epoching.epoch_assigner import EpochAssigner

logger = logging.getLogger(__name__)


class EpochBIDSWriter:
    """
    Reads BIDS preprocessed files, adds epoch columns, writes to epoched directory.
    
    Input:  derivatives/preprocessing/sub-xxx/ses-yyy/{modality}/file.tsv
    Output: derivatives/epoched/sub-xxx/ses-yyy/{modality}/file.tsv (+ epoch columns)
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize Epoch BIDS Writer.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = ConfigLoader(config_path)
        self.epoching_config = self.config.get("epoching", {})
        self.paths = self.config.get("paths", {})
        
        self.preprocessing_dir = Path(self.paths.get("derivatives", "data/derivatives")) / "preprocessing"
        self.epoched_dir = Path(self.paths.get("derivatives", "data/derivatives")) / "epoched"
        
        self.assigner = EpochAssigner(config_path)
        
        logger.info("Epoch BIDS Writer initialized")
        logger.info(f"Input: {self.preprocessing_dir}")
        logger.info(f"Output: {self.epoched_dir}")
    
    def detect_task_from_filename(self, filename: str) -> str:
        """
        Extract task name from BIDS filename.
        
        Args:
            filename: BIDS filename (e.g., 'sub-g01p01_ses-01_task-therapy_...')
        
        Returns:
            Task name ('restingstate' or 'therapy')
        
        Raises:
            ValueError: If task cannot be detected
        """
        if "_task-restingstate_" in filename:
            return "restingstate"
        elif "_task-therapy_" in filename:
            return "therapy"
        else:
            raise ValueError(f"Cannot detect task from filename: {filename}")
    
    def should_epoch_file(self, filename: str) -> bool:
        """
        Check if a file should be epoched based on include/exclude patterns.
        
        Args:
            filename: Name of the file to check
        
        Returns:
            True if file should be epoched, False otherwise
        """
        include_patterns = self.epoching_config.get("include", [])
        exclude_patterns = self.epoching_config.get("exclude", [])
        
        # Check exclude patterns first (using fnmatch for proper glob matching)
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(filename, pattern):
                logger.debug(f"File excluded by pattern '{pattern}': {filename}")
                return False
        
        # Check include patterns (using fnmatch for proper glob matching)
        for pattern in include_patterns:
            if fnmatch.fnmatch(filename, pattern):
                logger.debug(f"File included by pattern '{pattern}': {filename}")
                return True
        
        logger.debug(f"File does not match any include pattern: {filename}")
        return False
    
    def process_file(
        self,
        input_path: Path,
        subject: str,
        session: str,
        modality: str
    ) -> Optional[Path]:
        """
        Process a single file: read, add epoch columns, write to epoched directory.
        
        Args:
            input_path: Path to input TSV file
            subject: Subject ID (e.g., 'g01p01')
            session: Session ID (e.g., '01')
            modality: Modality ('bvp', 'eda', or 'hr')
        
        Returns:
            Path to output file if successful, None otherwise
        """
        filename = input_path.name
        
        # Check if file should be epoched
        if not self.should_epoch_file(filename):
            logger.debug(f"Skipping file: {filename}")
            return None
        
        logger.info(f"Processing: {filename}")
        
        # Detect task
        try:
            task = self.detect_task_from_filename(filename)
        except ValueError as e:
            logger.error(f"Skipping file: {e}")
            return None
        
        # Read TSV file
        try:
            df = pd.read_csv(input_path, sep='\t')
            logger.debug(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")
            return None
        
        # Detect time column
        time_col = None
        for possible_col in ['time', 'time_peak_start']:
            if possible_col in df.columns:
                time_col = possible_col
                break
        
        if time_col is None:
            logger.error(f"No time column found in {filename}")
            return None
        
        # Add epoch columns
        try:
            df = self.assigner.assign_all_epochs(df, task=task, time_column=time_col)
        except Exception as e:
            logger.error(f"Failed to assign epochs to {filename}: {e}")
            return None
        
        # Create output directory
        output_dir = self.epoched_dir / f"sub-{subject}" / f"ses-{session}" / modality
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Write epoched file
        output_path = output_dir / filename
        try:
            df.to_csv(output_path, sep='\t', index=False)
            logger.info(f"Saved epoched file: {output_path}")
        except Exception as e:
            logger.error(f"Failed to write {output_path}: {e}")
            return None
        
        # Copy JSON sidecar if exists
        json_path = input_path.with_suffix('.json')
        if json_path.exists():
            output_json_path = output_path.with_suffix('.json')
            try:
                # Read, add epoch info, write
                with open(json_path, 'r') as f:
                    metadata = json.load(f)
                
                metadata['Epoching'] = {
                    'EpochingApplied': True,
                    'Task': task,
                    'Methods': self.epoching_config.get('methods', {})
                }
                
                with open(output_json_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.debug(f"Copied and updated JSON sidecar: {output_json_path}")
            except Exception as e:
                logger.warning(f"Failed to copy JSON sidecar: {e}")
        
        return output_path
    
    def process_session(
        self,
        subject: str,
        session: str,
        modalities: Optional[List[str]] = None
    ) -> Dict[str, int]:
        """
        Process all epochable files for a session.
        
        Args:
            subject: Subject ID (e.g., 'g01p01')
            session: Session ID (e.g., '01')
            modalities: List of modalities to process (default: ['bvp', 'eda', 'hr'])
        
        Returns:
            Dictionary with counts: {'processed': N, 'skipped': M, 'failed': K}
        """
        if modalities is None:
            modalities = ['bvp', 'eda', 'hr']
        
        stats = {'processed': 0, 'skipped': 0, 'failed': 0}
        
        session_dir = self.preprocessing_dir / f"sub-{subject}" / f"ses-{session}"
        
        if not session_dir.exists():
            logger.warning(f"Session directory not found: {session_dir}")
            return stats
        
        for modality in modalities:
            modality_dir = session_dir / modality
            
            if not modality_dir.exists():
                logger.debug(f"Modality directory not found: {modality_dir}")
                continue
            
            # Find all TSV files
            tsv_files = list(modality_dir.glob("*.tsv"))
            logger.info(f"Found {len(tsv_files)} TSV files in {modality_dir}")
            
            for tsv_file in tsv_files:
                result = self.process_file(tsv_file, subject, session, modality)
                
                if result is not None:
                    stats['processed'] += 1
                elif self.should_epoch_file(tsv_file.name):
                    stats['failed'] += 1
                else:
                    stats['skipped'] += 1
        
        logger.info(f"Session {subject}/ses-{session}: {stats['processed']} processed, "
                   f"{stats['skipped']} skipped, {stats['failed']} failed")
        
        return stats
