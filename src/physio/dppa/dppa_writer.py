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
            ...     ('g01p01', 'ses-01', 'g01p02', 'ses-01'): icd_df1,
            ...     ('g01p01', 'ses-01', 'g01p03', 'ses-01'): icd_df2,
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
        Write intra-family ICD results to rectangular CSV.
        
        Format: Epochs as rows, dyads as columns
        First column: epoch_id
        Remaining columns: dyad pairs (subject1_subject2_session format)
        
        Args:
            icd_results: Dict mapping (family, subj1, subj2, session, task) -> ICD DataFrame
            task: Task name (e.g., 'therapy')
            method: Epoching method (e.g., 'sliding_duration30s_step5s')
            output_name: Optional custom filename (without extension)
        
        Returns:
            Path to created CSV file
        
        Example:
            >>> icd_data = {
            ...     ('g01', 'g01p01', 'g01p02', 'ses-01', 'therapy'): icd_df1,
            ...     ('g01', 'g01p01', 'g01p03', 'ses-01', 'therapy'): icd_df2,
            ... }
            >>> path = writer.write_intra_family(icd_data, 'therapy', 'sliding_duration30s_step5s')
        """
        if not icd_results:
            logger.warning("No ICD results to write")
            return Path()  # Return empty Path
        
        # Create output directory
        intra_dir = self.output_dir / "intra_family"
        intra_dir.mkdir(parents=True, exist_ok=True)
        
        # Build wide-format DataFrame (epochs × dyads)
        wide_data = {}
        
        for (family, subj1, subj2, session, _), icd_df in icd_results.items():
            # Create column name: subject1_subject2_session
            col_name = f"{subj1}_vs_{subj2}_{session}"
            
            # Add ICD column
            wide_data[col_name] = icd_df.set_index('epoch_id')['icd']
        
        # Combine all dyads into rectangular DataFrame
        df_wide = pd.DataFrame(wide_data).reset_index()
        df_wide = df_wide.rename(columns={'index': 'epoch_id'})
        
        # Sort by epoch_id
        df_wide = df_wide.sort_values('epoch_id').reset_index(drop=True)
        
        # Generate filename
        if output_name is None:
            output_name = f"intra_family_icd_task-{task}_method-{method}"
        
        csv_file = intra_dir / f"{output_name}.csv"
        
        # Save CSV
        df_wide.to_csv(csv_file, index=False)
        
        # Create JSON sidecar
        json_file = csv_file.with_suffix('.json')
        metadata = {
            "Description": "Intra-family Inter-Centroid Distances within same-session dyads",
            "TaskName": task,
            "EpochingMethod": method,
            "Format": "Rectangular (epochs × dyads)",
            "Columns": {
                "epoch_id": "Epoch identifier (0-indexed)",
                "dyad_columns": "Each column = subject1_vs_subject2_session, values = ICD in ms (NaN if invalid)"
            },
            "Formula": "ICD = sqrt((centroid_x1 - centroid_x2)^2 + (centroid_y1 - centroid_y2)^2)",
            "CreationDate": datetime.now().isoformat(),
            "NumberOfDyads": len(icd_results),
            "NumberOfEpochs": len(df_wide),
            "ValidICDs": int((~df_wide.iloc[:, 1:].isna()).sum().sum()),
            "TotalICDs": int((len(df_wide)) * len(icd_results))
        }
        
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Wrote intra-family ICD: {csv_file.name} ({len(icd_results)} dyads, {len(df_wide)} epochs)")
        return csv_file
