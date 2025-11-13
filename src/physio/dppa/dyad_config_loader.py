"""
Dyad Configuration Loader for DPPA Analysis.

This module loads and parses the dyad configuration file (dppa_dyads.yaml)
which defines how participants are paired for inter-session and intra-family
comparisons.

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import logging
import yaml
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from itertools import combinations

logger = logging.getLogger(__name__)


class DyadConfigLoader:
    """
    Load and parse dyad configuration from YAML file.
    
    This class provides methods to retrieve dyad pairs for both inter-session
    (all pairs across all sessions) and intra-family (same session pairs)
    comparisons.
    
    Attributes:
        config_path: Path to dyad configuration YAML file
        config: Parsed configuration dictionary
    
    Example:
        >>> loader = DyadConfigLoader()
        >>> pairs = loader.get_inter_session_pairs(task='therapy')
        >>> for (s1, ses1), (s2, ses2) in pairs:
        ...     print(f"Compare {s1}/{ses1} with {s2}/{ses2}")
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize DyadConfigLoader.
        
        Args:
            config_path: Path to dyad config file. If None, uses default.
        """
        if config_path is None:
            config_path = Path("config/dppa_dyads.yaml")
        else:
            config_path = Path(config_path)
        
        self.config_path = config_path
        
        # Load configuration
        if not config_path.exists():
            raise FileNotFoundError(f"Dyad config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Validate configuration structure
        self._validate_config()
        
        logger.info(f"Dyad Config Loader initialized from {config_path}")
    
    def _validate_config(self):
        """Validate configuration structure and required keys."""
        # Check for main sections
        if 'inter_session' not in self.config:
            raise ValueError("Missing 'inter_session' section in config")
        if 'intra_family' not in self.config:
            raise ValueError("Missing 'intra_family' section in config")
        
        # Validate inter_session
        inter = self.config['inter_session']
        if 'method' not in inter:
            raise ValueError("Missing 'method' in inter_session config")
        if 'tasks' not in inter or not isinstance(inter['tasks'], list):
            raise ValueError("Missing or invalid 'tasks' in inter_session config")
        
        # Validate intra_family
        intra = self.config['intra_family']
        if 'method' not in intra:
            raise ValueError("Missing 'method' in intra_family config")
        if 'tasks' not in intra or not isinstance(intra['tasks'], list):
            raise ValueError("Missing or invalid 'tasks' in intra_family config")
        if 'families' not in intra or not isinstance(intra['families'], dict):
            raise ValueError("Missing or invalid 'families' in intra_family config")
        
        logger.debug("Configuration validation passed")
    
    def get_inter_session_pairs(
        self,
        task: Optional[str] = None
    ) -> List[Tuple[Tuple[str, str], Tuple[str, str]]]:
        """
        Get all inter-session dyad pairs.
        
        Generates all possible pairs across all sessions. For N sessions,
        this produces N×(N-1)/2 unique pairs.
        
        Args:
            task: Optional task filter (e.g., 'therapy'). If None, all tasks.
        
        Returns:
            List of pairs: [((subject1, session1), (subject2, session2)), ...]
        
        Example:
            >>> loader = DyadConfigLoader()
            >>> pairs = loader.get_inter_session_pairs(task='therapy')
            >>> len(pairs)  # ~1275 for 51 sessions
        """
        inter = self.config['inter_session']
        
        # Apply task filter
        if task:
            if task not in inter['tasks']:
                logger.warning(f"Task '{task}' not in inter_session config")
                return []
        
        # Get all sessions from intra_family configuration
        all_sessions = []
        for family, sessions in self.config['intra_family']['families'].items():
            for session, participants in sessions.items():
                for participant in participants:
                    all_sessions.append((participant, session))
        
        # Generate all unique pairs
        pairs = list(combinations(all_sessions, 2))
        
        logger.info(f"Generated {len(pairs)} inter-session pairs")
        return pairs
    
    def get_intra_family_pairs(
        self,
        family: Optional[str] = None,
        session: Optional[str] = None,
        task: Optional[str] = None
    ) -> List[Tuple[Tuple[str, str, str], Tuple[str, str, str]]]:
        """
        Get intra-family dyad pairs (within same session).
        
        Args:
            family: Optional family filter (e.g., 'g01')
            session: Optional session filter (e.g., 'ses-01')
            task: Optional task filter (e.g., 'therapy')
        
        Returns:
            List of pairs: [((family, subject1, session), (family, subject2, session)), ...]
        
        Example:
            >>> loader = DyadConfigLoader()
            >>> pairs = loader.get_intra_family_pairs(family='g01', session='ses-01')
            >>> # Returns 15 pairs (6 participants choose 2)
        """
        intra = self.config['intra_family']
        
        # Apply task filter
        if task:
            if task not in intra['tasks']:
                logger.warning(f"Task '{task}' not in intra_family config")
                return []
        
        families_dict = intra['families']
        
        # Apply family filter
        if family:
            if family not in families_dict:
                logger.warning(f"Family '{family}' not in config")
                return []
            families_dict = {family: families_dict[family]}
        
        # Generate pairs
        all_pairs = []
        
        for fam_id, sessions_dict in families_dict.items():
            # Apply session filter
            if session:
                # Normalize session format
                ses_key = session if session.startswith('ses-') else f'ses-{session}'
                if ses_key not in sessions_dict:
                    continue
                sessions_dict = {ses_key: sessions_dict[ses_key]}
            
            for ses_id, participants in sessions_dict.items():
                if len(participants) < 2:
                    continue
                
                # Generate all pairs within this family/session
                for p1, p2 in combinations(participants, 2):
                    all_pairs.append((
                        (fam_id, p1, ses_id),
                        (fam_id, p2, ses_id)
                    ))
        
        logger.info(f"Generated {len(all_pairs)} intra-family pairs")
        return all_pairs
    
    def get_inter_session_method(self) -> str:
        """Get the epoching method for inter-session comparison."""
        return self.config['inter_session']['method']
    
    def get_intra_family_method(self) -> str:
        """Get the epoching method for intra-family comparison."""
        return self.config['intra_family']['method']
    
    def get_inter_session_tasks(self) -> List[str]:
        """Get list of tasks for inter-session comparison."""
        return self.config['inter_session']['tasks']
    
    def get_intra_family_tasks(self) -> List[str]:
        """Get list of tasks for intra-family comparison."""
        return self.config['intra_family']['tasks']
    
    def get_families(self) -> List[str]:
        """Get list of all family IDs."""
        return list(self.config['intra_family']['families'].keys())
    
    def get_family_sessions(self, family: str) -> List[str]:
        """
        Get list of sessions for a specific family.
        
        Args:
            family: Family ID (e.g., 'g01')
        
        Returns:
            List of session IDs (e.g., ['ses-01', 'ses-02'])
        """
        families = self.config['intra_family']['families']
        if family not in families:
            logger.warning(f"Family '{family}' not found in config")
            return []
        return list(families[family].keys())
