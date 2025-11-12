"""
DPPA Writer for exporting ICD results.

This module handles writing Inter-Centroid Distance results to BIDS-compliant
CSV files with two different formats:
- Inter-session: Rectangular CSV (120 epochs × N dyads as columns)
- Intra-family: Long format CSV with dyad_id column (variable rows)

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import logging
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime

from src.core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class DPPAWriter:
    """
    Write ICD results to BIDS-compliant CSV files.
    
    This class provides methods to export Inter-Centroid Distance results
    in two formats:
    1. Inter-session: Wide format (epochs as rows, dyads as columns)
    2. Intra-family: Long format (dyad_id column, variable rows per dyad)
    
    Attributes:
        output_dir: Path to DPPA output directory
    
    Example:
        >>> writer = DPPAWriter()
        >>> writer.write_inter_session(icd_data, task='therapy', method='nsplit120')
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize DPPAWriter.
        
        Args:
            config_path: Path to configuration file. If None, uses default.
        """
        self.config = ConfigLoader(config_path)
        
        # Get output directory path
        paths = self.config.get("paths", {})
        self.output_dir = Path(paths.get("dppa", "data/derivatives/dppa"))
        
        logger.info("DPPA Writer initialized")
    
    def write_inter_session(
        self,
        icd_results: Dict[Tuple[str, str, str, str], pd.DataFrame],
        task: str,
        method: str,
        output_name: Optional[str] = None
    ) -> Path:
        """
        Write inter-session ICD results to rectangular CSV.
        
        Format: 120 rows (epochs) × N columns (dyad pairs)
        Column names: {subject1}_{session1}_vs_{subject2}_{session2}
        
        Args:
            icd_results: Dict mapping (subj1, ses1, subj2, ses2) -> ICD DataFrame
            task: Task name (e.g., 'therapy')
            method: Epoching method (e.g., 'nsplit120')
            output_name: Optional custom filename (without extension)
        
        Returns:
            Path to created CSV file
        
        Example:
            >>> icd_data = {
            ...     ('f01p01', 'ses-01', 'f01p02', 'ses-01'): icd_df1,
            ...     ('f01p01', 'ses-01', 'f01p03', 'ses-01'): icd_df2,
            ... }
            >>> path = writer.write_inter_session(icd_data, 'therapy', 'nsplit120')
        """
        if not icd_results:
            logger.warning("No ICD results to write")
            return None
        
        # Create output directory
        inter_dir = self.output_dir / "inter_session"
        inter_dir.mkdir(parents=True, exist_ok=True)
        
        # Build wide DataFrame
        wide_data = {}
        
        for (subj1, ses1, subj2, ses2), icd_df in icd_results.items():
            # Create column name
            col_name = f"{subj1}_{ses1}_vs_{subj2}_{ses2}"
            
            # Extract ICD values (ensure sorted by epoch_id)
            icd_df_sorted = icd_df.sort_values('epoch_id')
            wide_data[col_name] = icd_df_sorted['icd'].values
        
        # Create DataFrame
        df_wide = pd.DataFrame(wide_data)
        df_wide.insert(0, 'epoch_id', range(len(df_wide)))
        
        # Generate filename
        if output_name is None:
            output_name = f"inter_session_icd_task-{task}_method-{method}"
        
        csv_file = inter_dir / f"{output_name}.csv"
        
        # Save CSV
        df_wide.to_csv(csv_file, index=False)
        
        # Create JSON sidecar
        json_file = csv_file.with_suffix('.json')
        metadata = {
            "Description": "Inter-session Inter-Centroid Distances across all dyad pairs",
            "TaskName": task,
            "EpochingMethod": method,
            "Format": "Rectangular (epochs × dyads)",
            "Columns": {
                "epoch_id": "Epoch identifier (0-119 for nsplit120)",
                "dyad_columns": "ICD values in ms for each dyad pair (NaN if invalid)"
            },
            "Formula": "ICD = sqrt((centroid_x1 - centroid_x2)^2 + (centroid_y1 - centroid_y2)^2)",
            "CreationDate": datetime.now().isoformat(),
            "NumberOfDyads": len(icd_results),
            "NumberOfEpochs": len(df_wide),
            "ValidICDs": int(df_wide.iloc[:, 1:].notna().sum().sum()),
            "TotalICDs": int((len(df_wide) - 1) * len(icd_results))  # -1 for epoch_id column
        }
        
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Wrote inter-session ICD: {csv_file.name} ({len(icd_results)} dyads)")
        return csv_file
    
    def write_intra_family(
        self,
        icd_results: Dict[Tuple[str, str, str, str, str], pd.DataFrame],
        task: str,
        method: str,
        output_name: Optional[str] = None
    ) -> Path:
        """
        Write intra-family ICD results to long-format CSV.
        
        Format: Variable rows with dyad_id column
        Columns: family, dyad_id, subject1, subject2, session, epoch_id, icd
        
        Args:
            icd_results: Dict mapping (family, subj1, subj2, session, task) -> ICD DataFrame
            task: Task name (e.g., 'therapy')
            method: Epoching method (e.g., 'sliding_duration30s_step5s')
            output_name: Optional custom filename (without extension)
        
        Returns:
            Path to created CSV file
        
        Example:
            >>> icd_data = {
            ...     ('f01', 'f01p01', 'f01p02', 'ses-01', 'therapy'): icd_df1,
            ...     ('f01', 'f01p01', 'f01p03', 'ses-01', 'therapy'): icd_df2,
            ... }
            >>> path = writer.write_intra_family(icd_data, 'therapy', 'sliding_duration30s_step5s')
        """
        if not icd_results:
            logger.warning("No ICD results to write")
            return None
        
        # Create output directory
        intra_dir = self.output_dir / "intra_family"
        intra_dir.mkdir(parents=True, exist_ok=True)
        
        # Build long-format DataFrame
        long_data = []
        
        for (family, subj1, subj2, session, _), icd_df in icd_results.items():
            # Create dyad_id
            dyad_id = f"{subj1}_vs_{subj2}"
            
            # Add family/dyad info to each row
            df_copy = icd_df.copy()
            df_copy.insert(0, 'family', family)
            df_copy.insert(1, 'dyad_id', dyad_id)
            df_copy.insert(2, 'subject1', subj1)
            df_copy.insert(3, 'subject2', subj2)
            df_copy.insert(4, 'session', session)
            
            long_data.append(df_copy)
        
        # Concatenate all dyads
        df_long = pd.concat(long_data, ignore_index=True)
        
        # Select and order columns
        columns = ['family', 'dyad_id', 'subject1', 'subject2', 'session', 'epoch_id', 'icd']
        df_long = df_long[columns]
        
        # Sort by family, dyad, epoch
        df_long = df_long.sort_values(['family', 'dyad_id', 'epoch_id']).reset_index(drop=True)
        
        # Generate filename
        if output_name is None:
            output_name = f"intra_family_icd_task-{task}_method-{method}"
        
        csv_file = intra_dir / f"{output_name}.csv"
        
        # Save CSV
        df_long.to_csv(csv_file, index=False)
        
        # Create JSON sidecar
        json_file = csv_file.with_suffix('.json')
        metadata = {
            "Description": "Intra-family Inter-Centroid Distances within same-session dyads",
            "TaskName": task,
            "EpochingMethod": method,
            "Format": "Long format (dyad_id × epochs)",
            "Columns": {
                "family": "Family identifier (e.g., f01)",
                "dyad_id": "Dyad pair identifier (subject1_vs_subject2)",
                "subject1": "First participant ID",
                "subject2": "Second participant ID",
                "session": "Session identifier (e.g., ses-01)",
                "epoch_id": "Epoch identifier",
                "icd": "Inter-Centroid Distance in ms (NaN if invalid)"
            },
            "Formula": "ICD = sqrt((centroid_x1 - centroid_x2)^2 + (centroid_y1 - centroid_y2)^2)",
            "CreationDate": datetime.now().isoformat(),
            "NumberOfDyads": len(icd_results),
            "TotalRows": len(df_long),
            "ValidICDs": int(df_long['icd'].notna().sum()),
            "Families": sorted(df_long['family'].unique().tolist()),
            "Sessions": sorted(df_long['session'].unique().tolist())
        }
        
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Wrote intra-family ICD: {csv_file.name} ({len(icd_results)} dyads, {len(df_long)} rows)")
        return csv_file
