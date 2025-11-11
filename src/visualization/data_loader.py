"""
Data Loader for Visualization Module.

This module loads preprocessed physiological data from the BIDS derivatives
structure for visualization purposes.

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import json
import gzip
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import pandas as pd
import numpy as np
import logging

from src.core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class VisualizationDataLoader:
    """
    Loads preprocessed BVP, EDA, and HR data for visualization.
    
    This class handles loading from the BIDS derivatives structure:
    data/derivatives/preprocessing/sub-{subject}/ses-{session}/{modality}/
    
    Examples:
        >>> loader = VisualizationDataLoader()
        >>> data = loader.load_subject_session('f01p01', '01')
        >>> bvp_signals = data['bvp']['signals']
        >>> eda_metrics = data['eda']['metrics']
    """
    
    def __init__(self, derivatives_path: Optional[Path] = None, config_path: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            derivatives_path: Path to derivatives directory
                Default: Loaded from config YAML
            config_path: Path to configuration YAML file
                Default: config/config.yaml
        """
        # Load configuration
        self.config_loader = ConfigLoader(config_path)
        self.config = self.config_loader.config
        
        # Use derivatives_path from config if not provided
        if derivatives_path is None:
            base_path = Path(self.config['paths']['derivatives'])
            preprocessing_dir = self.config['output']['preprocessing_dir']
            derivatives_path = base_path / preprocessing_dir
        
        self.derivatives_path = Path(derivatives_path)
        
        if not self.derivatives_path.exists():
            raise FileNotFoundError(
                f"Derivatives directory not found: {self.derivatives_path}"
            )
    
    def load_subject_session(
        self, 
        subject: str, 
        session: str,
        modalities: Optional[List[str]] = None
    ) -> Dict:
        """
        Load all data for a subject/session.
        
        Args:
            subject: Subject ID (e.g., 'f01p01')
            session: Session ID (e.g., '01')
            modalities: List of modalities to load ['bvp', 'eda', 'hr']
                Default: Load all available
        
        Returns:
            Dictionary with structure:
            {
                'bvp': {'signals': {...}, 'metrics': {...}, 'metadata': {...}},
                'eda': {'signals': {...}, 'metrics': {...}, 'events': {...}, 'metadata': {...}},
                'hr': {'signals': {...}, 'metrics': {...}, 'metadata': {...}},
                'subject': 'f01p01',
                'session': '01'
            }
        """
        if modalities is None:
            modalities = ['bvp', 'eda', 'hr']
        
        # Build paths
        subject_id = f"sub-{subject}" if not subject.startswith('sub-') else subject
        session_id = f"ses-{session}" if not session.startswith('ses-') else session
        
        subject_session_path = self.derivatives_path / subject_id / session_id
        
        if not subject_session_path.exists():
            raise FileNotFoundError(
                f"Subject/session directory not found: {subject_session_path}"
            )
        
        logger.info(f"Loading data for {subject_id}/{session_id}")
        
        data = {
            'subject': subject,
            'session': session,
            'subject_id': subject_id,
            'session_id': session_id
        }
        
        # Load each modality
        for modality in modalities:
            modality_path = subject_session_path / modality
            
            if not modality_path.exists():
                logger.warning(f"Modality {modality} not found for {subject_id}/{session_id}")
                continue
            
            logger.info(f"Loading {modality} data...")
            
            if modality == 'bvp':
                data['bvp'] = self._load_bvp_data(modality_path, subject_id, session_id)
            elif modality == 'eda':
                data['eda'] = self._load_eda_data(modality_path, subject_id, session_id)
            elif modality == 'hr':
                data['hr'] = self._load_hr_data(modality_path, subject_id, session_id)
        
        return data
    
    def _load_bvp_data(self, modality_path: Path, subject_id: str, session_id: str) -> Dict:
        """Load BVP processed signals and metrics."""
        bvp_data = {
            'signals': {},
            'metrics': None,
            'metadata': {}
        }
        
        # Load processed signals for each moment
        for moment in ['restingstate', 'therapy']:
            signal_file = modality_path / f"{subject_id}_{session_id}_task-{moment}_desc-processed_recording-bvp.tsv"
            metadata_file = modality_path / f"{subject_id}_{session_id}_task-{moment}_desc-processed_recording-bvp.json"
            
            if signal_file.exists():
                # Load TSV (not compressed)
                bvp_data['signals'][moment] = pd.read_csv(signal_file, sep='\t')
                logger.info(f"  Loaded BVP signals for {moment}: {len(bvp_data['signals'][moment])} samples")
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    bvp_data['metadata'][moment] = json.load(f)
        
        # Load metrics
        metrics_file = modality_path / f"{subject_id}_{session_id}_desc-bvp-metrics_physio.tsv"
        if metrics_file.exists():
            bvp_data['metrics'] = pd.read_csv(metrics_file, sep='\t')
            logger.info(f"  Loaded BVP metrics: {len(bvp_data['metrics'])} rows")
        
        return bvp_data
    
    def _load_eda_data(self, modality_path: Path, subject_id: str, session_id: str) -> Dict:
        """Load EDA processed signals, SCR events, and metrics."""
        eda_data = {
            'signals': {},
            'events': {},
            'metrics': None,
            'metadata': {}
        }
        
        # Load processed signals and events for each moment
        for moment in ['restingstate', 'therapy']:
            # Signals
            signal_file = modality_path / f"{subject_id}_{session_id}_task-{moment}_desc-processed_recording-eda.tsv"
            if signal_file.exists():
                eda_data['signals'][moment] = pd.read_csv(signal_file, sep='\t')
                logger.info(f"  Loaded EDA signals for {moment}: {len(eda_data['signals'][moment])} samples")
            
            # SCR Events
            events_file = modality_path / f"{subject_id}_{session_id}_task-{moment}_desc-scr_events.tsv"
            if events_file.exists():
                eda_data['events'][moment] = pd.read_csv(events_file, sep='\t')
                logger.info(f"  Loaded SCR events for {moment}: {len(eda_data['events'][moment])} events")
            
            # Metadata
            metadata_file = modality_path / f"{subject_id}_{session_id}_task-{moment}_desc-processed_recording-eda.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    eda_data['metadata'][moment] = json.load(f)
        
        # Load metrics
        metrics_file = modality_path / f"{subject_id}_{session_id}_desc-eda-metrics_physio.tsv"
        if metrics_file.exists():
            eda_data['metrics'] = pd.read_csv(metrics_file, sep='\t')
            logger.info(f"  Loaded EDA metrics: {len(eda_data['metrics'])} rows")
        
        return eda_data
    
    def _load_hr_data(self, modality_path: Path, subject_id: str, session_id: str) -> Dict:
        """Load HR processed signals and metrics."""
        hr_data = {
            'signals': {},
            'metrics': None,
            'metadata': {}
        }
        
        # HR now uses separate per-moment files (new format)
        moments = ['restingstate', 'therapy']
        
        for moment in moments:
            signal_file = modality_path / f"{subject_id}_{session_id}_task-{moment}_desc-processed_recording-hr.tsv"
            metadata_file = modality_path / f"{subject_id}_{session_id}_task-{moment}_desc-processed_recording-hr.json"
            
            if signal_file.exists():
                hr_data['signals'][moment] = pd.read_csv(signal_file, sep='\t')
                logger.info(f"  Loaded HR signals for {moment}: {len(hr_data['signals'][moment])} samples")
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    hr_data['metadata'][moment] = json.load(f)
        
        # Load combined metrics (aggregated across moments)
        metrics_file = modality_path / f"{subject_id}_{session_id}_desc-hr-summary.json"
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                summary_data = json.load(f)
                # Convert summary to DataFrame for compatibility
                hr_data['metrics'] = pd.DataFrame([summary_data])
                logger.info(f"  Loaded HR summary metrics")
        
        return hr_data
    
    def list_available_subjects(self) -> List[Tuple[str, str]]:
        """
        List all available subject/session combinations.
        
        Returns:
            List of (subject, session) tuples
        """
        subjects_sessions = []
        
        for subject_dir in sorted(self.derivatives_path.glob('sub-*')):
            subject = subject_dir.name.replace('sub-', '')
            
            for session_dir in sorted(subject_dir.glob('ses-*')):
                session = session_dir.name.replace('ses-', '')
                
                # Check if at least one modality exists
                has_data = any([
                    (session_dir / 'bvp').exists(),
                    (session_dir / 'eda').exists(),
                    (session_dir / 'hr').exists()
                ])
                
                if has_data:
                    subjects_sessions.append((subject, session))
        
        logger.info(f"Found {len(subjects_sessions)} subject/session combinations")
        return subjects_sessions
    
    def get_available_modalities(self, subject: str, session: str) -> List[str]:
        """
        Get list of available modalities for a subject/session.
        
        Args:
            subject: Subject ID
            session: Session ID
        
        Returns:
            List of available modalities ['bvp', 'eda', 'hr']
        """
        subject_id = f"sub-{subject}" if not subject.startswith('sub-') else subject
        session_id = f"ses-{session}" if not session.startswith('ses-') else session
        
        subject_session_path = self.derivatives_path / subject_id / session_id
        
        modalities = []
        for modality in ['bvp', 'eda', 'hr']:
            if (subject_session_path / modality).exists():
                modalities.append(modality)
        
        return modalities
