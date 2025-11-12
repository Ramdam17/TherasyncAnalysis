"""
Module for loading Inter-Centroid Distance (ICD) data for dyadic analysis.

This module provides functionality to load ICD time series for specific dyads,
supporting both resting state and therapy tasks.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

from src.core.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class DyadICDLoader:
    """
    Load Inter-Centroid Distance (ICD) data for dyadic analysis.

    This class handles loading ICD CSV files for specific dyad pairs,
    supporting both inter-session and intra-family dyad types.

    Attributes:
        config: Configuration object containing paths and settings.
        derivatives_path: Path to derivatives directory.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize DyadICDLoader.

        Args:
            config_path: Optional path to configuration file.
                        If None, uses default config.yaml.
        """
        self.config = ConfigLoader(config_path)
        self.derivatives_path = Path(self.config.get("paths.derivatives", "data/derivatives"))
        logger.info("DyadICDLoader initialized")

    def parse_dyad_info(self, dyad_pair: str) -> Dict[str, str]:
        """
        Parse dyad pair string to extract subject and session information.

        Args:
            dyad_pair: Dyad identifier in format:
                      "sub1_ses-XX_vs_sub2_ses-YY"
                      Example: "f01p01_ses-01_vs_f01p02_ses-01"

        Returns:
            Dictionary with keys: 'sub1', 'ses1', 'sub2', 'ses2'

        Raises:
            ValueError: If dyad_pair format is invalid.

        Example:
            >>> loader = DyadICDLoader()
            >>> info = loader.parse_dyad_info("f01p01_ses-01_vs_f01p02_ses-01")
            >>> print(info)
            {'sub1': 'f01p01', 'ses1': '01', 'sub2': 'f01p02', 'ses2': '01'}
        """
        if not dyad_pair or "_vs_" not in dyad_pair:
            raise ValueError(
                f"Invalid dyad pair format: '{dyad_pair}'. "
                "Expected format: 'sub1_ses-XX_vs_sub2_ses-YY'"
            )

        try:
            # Split on "_vs_" to get both sides
            parts = dyad_pair.split("_vs_")
            if len(parts) != 2:
                raise ValueError("Expected exactly one '_vs_' separator")

            # Parse first subject
            left_parts = parts[0].split("_ses-")
            if len(left_parts) != 2:
                raise ValueError("Missing session in first subject")
            sub1, ses1 = left_parts[0], left_parts[1]

            # Parse second subject
            right_parts = parts[1].split("_ses-")
            if len(right_parts) != 2:
                raise ValueError("Missing session in second subject")
            sub2, ses2 = right_parts[0], right_parts[1]

            return {
                "sub1": sub1,
                "ses1": ses1,
                "sub2": sub2,
                "ses2": ses2,
            }

        except (IndexError, ValueError) as e:
            raise ValueError(
                f"Invalid dyad pair format: '{dyad_pair}'. "
                f"Expected format: 'sub1_ses-XX_vs_sub2_ses-YY'. Error: {e}"
            )

    def load_icd(
        self, dyad_pair: str, task: str, method: str
    ) -> pd.DataFrame:
        """
        Load ICD time series for a specific dyad and task.

        Args:
            dyad_pair: Dyad identifier (e.g., "f01p01_ses-01_vs_f01p02_ses-01")
            task: Task name ('restingstate' or 'therapy')
            method: Epoching method (e.g., 'nsplit120', 'sliding_duration30s_step5s')

        Returns:
            DataFrame with columns: ['epoch_id', 'icd_value']
            Note: restingstate will have 1 row (epoch_id=0),
                  therapy will have multiple rows.

        Raises:
            FileNotFoundError: If ICD file does not exist.
            ValueError: If required columns are missing or invalid format.

        Example:
            >>> loader = DyadICDLoader()
            >>> df = loader.load_icd("f01p01_ses-01_vs_f01p02_ses-01", "therapy", "nsplit120")
            >>> print(df.head())
               epoch_id  icd_value
            0         0      50.23
            1         1      48.91
        """
        # Determine dyad type (inter_session or intra_family)
        dyad_info = self.parse_dyad_info(dyad_pair)
        
        # For now, assume inter_session (can be enhanced later)
        dyad_type = "inter_session"
        
        # Construct file path
        icd_file = (
            self.derivatives_path
            / "dppa"
            / dyad_type
            / f"{dyad_type}_icd_task-{task}_method-{method}.csv"
        )

        if not icd_file.exists():
            raise FileNotFoundError(
                f"ICD file not found: {icd_file}\n"
                f"Dyad: {dyad_pair}, Task: {task}, Method: {method}"
            )

        logger.info(f"Loading ICD data from: {icd_file}")

        # Load CSV
        df = pd.read_csv(icd_file)

        # Validate columns
        if "epoch_id" not in df.columns:
            raise ValueError(f"Missing 'epoch_id' column in {icd_file}")

        if dyad_pair not in df.columns:
            raise ValueError(
                f"Dyad pair '{dyad_pair}' not found in {icd_file}. "
                f"Available columns: {list(df.columns)}"
            )

        # Extract relevant columns and rename
        result = df[["epoch_id", dyad_pair]].copy()
        result.rename(columns={dyad_pair: "icd_value"}, inplace=True)

        logger.info(
            f"Loaded {len(result)} epochs for dyad {dyad_pair} "
            f"(task={task}, method={method})"
        )

        return result

    def load_both_tasks(self, dyad_pair: str, method: str) -> Dict[str, pd.DataFrame]:
        """
        Load ICD data for both restingstate and therapy tasks.

        Args:
            dyad_pair: Dyad identifier (e.g., "f01p01_ses-01_vs_f01p02_ses-01")
            method: Epoching method (e.g., 'nsplit120')

        Returns:
            Dictionary with keys 'restingstate' and 'therapy',
            each containing a DataFrame with ICD time series.

        Raises:
            FileNotFoundError: If either task file is missing.

        Example:
            >>> loader = DyadICDLoader()
            >>> data = loader.load_both_tasks("f01p01_ses-01_vs_f01p02_ses-01", "nsplit120")
            >>> print(f"Resting: {len(data['restingstate'])} epochs")
            >>> print(f"Therapy: {len(data['therapy'])} epochs")
            Resting: 1 epochs
            Therapy: 120 epochs
        """
        logger.info(f"Loading both tasks for dyad: {dyad_pair}, method: {method}")

        result = {}
        for task in ["restingstate", "therapy"]:
            result[task] = self.load_icd(dyad_pair, task, method)

        logger.info(
            f"Successfully loaded both tasks: "
            f"restingstate={len(result['restingstate'])} epochs, "
            f"therapy={len(result['therapy'])} epochs"
        )

        return result
