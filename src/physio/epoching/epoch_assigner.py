"""
Epoch assignment module for physiological signals.

This module provides functions to assign epoch IDs to time series data using
different epoching methods: fixed windows, N-split, and sliding windows.

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union

from src.core.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


class EpochAssigner:
    """
    Assigns epoch IDs to time series data using configurable methods.
    
    Supports three epoching methods:
    1. Fixed windows: Fixed duration with configurable overlap
    2. N-split: Divide signal into N equal epochs
    3. Sliding window: Fixed duration with small step (high overlap)
    
    Special case: task-restingstate always gets epoch_id=0 for all methods.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize Epoch Assigner.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        self.config = ConfigLoader(config_path)
        self.epoching_config = self.config.get("epoching", {})
        
        logger.info("Epoch Assigner initialized")
        logger.debug(f"Epoching enabled: {self.epoching_config.get('enabled', False)}")
    
    def assign_fixed_epochs(
        self,
        time: np.ndarray,
        duration: float,
        overlap: float,
        min_duration_ratio: float = 0.0
    ) -> np.ndarray:
        """
        Assign epoch IDs using fixed window with overlap.
        
        Args:
            time: Time array (in seconds)
            duration: Epoch duration in seconds
            overlap: Overlap between epochs in seconds
            min_duration_ratio: Minimum ratio of epoch duration to keep
                                (0.0 = keep all, 1.0 = only complete epochs)
        
        Returns:
            Array of epoch IDs (same length as time), -1 for rejected samples
        
        Example:
            duration=30s, overlap=5s → step=25s
            Epochs: [0-30], [25-55], [50-80], [75-105]
        """
        step = duration - overlap
        total_duration = time[-1] - time[0]
        
        # Initialize all to -1 (not in any epoch)
        epoch_ids = np.full(len(time), -1, dtype=int)
        
        epoch_id = 0
        epoch_start = time[0]
        
        while epoch_start < time[-1]:
            epoch_end = epoch_start + duration
            
            # Find samples in this epoch
            mask = (time >= epoch_start) & (time < epoch_end)
            
            # Check if epoch meets minimum duration requirement
            actual_duration = min(epoch_end, time[-1]) - epoch_start
            duration_ratio = actual_duration / duration
            
            if duration_ratio >= min_duration_ratio:
                epoch_ids[mask] = epoch_id
                logger.debug(f"Epoch {epoch_id}: {epoch_start:.1f}s - {min(epoch_end, time[-1]):.1f}s " 
                           f"(duration ratio: {duration_ratio:.2f})")
            else:
                logger.debug(f"Rejected partial epoch {epoch_id}: duration ratio {duration_ratio:.2f} "
                           f"< threshold {min_duration_ratio}")
            
            epoch_id += 1
            epoch_start += step
        
        n_epochs = len(np.unique(epoch_ids[epoch_ids >= 0]))
        n_rejected = np.sum(epoch_ids == -1)
        logger.info(f"Fixed epochs assigned: {n_epochs} epochs, {n_rejected} samples rejected")
        
        return epoch_ids
    
    def assign_nsplit_epochs(
        self,
        time: np.ndarray,
        n_epochs: int
    ) -> np.ndarray:
        """
        Assign epoch IDs by splitting signal into N equal epochs.
        
        Args:
            time: Time array (in seconds)
            n_epochs: Number of epochs to create
        
        Returns:
            Array of epoch IDs (same length as time)
        
        Example:
            Signal of 120s, n_epochs=10 → 10 epochs of 12s each
        """
        total_duration = time[-1] - time[0]
        epoch_duration = total_duration / n_epochs
        
        # Calculate epoch ID for each sample
        normalized_time = time - time[0]
        epoch_ids = np.floor(normalized_time / epoch_duration).astype(int)
        
        # Ensure last sample is in last epoch (handle floating point rounding)
        epoch_ids = np.clip(epoch_ids, 0, n_epochs - 1)
        
        logger.info(f"N-split epochs assigned: {n_epochs} epochs of {epoch_duration:.1f}s each")
        
        return epoch_ids
    
    def assign_all_epochs(
        self,
        df: pd.DataFrame,
        task: str,
        time_column: str = 'time'
    ) -> pd.DataFrame:
        """
        Assign all epoch columns based on configuration and task type.
        
        Args:
            df: DataFrame with time series data
            task: Task name ('restingstate' or 'therapy')
            time_column: Name of time column (default: 'time')
        
        Returns:
            DataFrame with added epoch columns:
                - epoch_fixed_duration{X}s_overlap{Y}s
                - epoch_nsplit{N}
                - epoch_sliding_duration{X}s_step{Y}s
        
        Note:
            For task='restingstate', all epoch columns are set to 0.
            For task='therapy', epochs are assigned based on methods config.
        """
        if time_column not in df.columns:
            raise ValueError(f"Time column '{time_column}' not found in DataFrame")
        
        time = df[time_column].values
        
        # Special case: restingstate always epoch 0
        if task == "restingstate":
            logger.info(f"Task is restingstate: assigning epoch_id=0 to all {len(df)} samples")
            
            # Build column names from config (same as therapy)
            methods = self.epoching_config.get("methods", {})
            
            if methods.get("fixed", {}).get("enabled", False):
                fixed = methods["fixed"]
                duration = fixed.get("duration", 30)
                overlap = fixed.get("overlap", 5)
                col_name = f"epoch_fixed_duration{duration}s_overlap{overlap}s"
                df[col_name] = "[0]"  # JSON list format for consistency
            
            if methods.get("nsplit", {}).get("enabled", False):
                nsplit = methods["nsplit"]
                n_epochs = nsplit.get("n_epochs", 120)
                col_name = f"epoch_nsplit{n_epochs}"
                df[col_name] = "[0]"  # JSON list format for consistency
            
            if methods.get("sliding", {}).get("enabled", False):
                sliding = methods["sliding"]
                duration = sliding.get("duration", 30)
                step = sliding.get("step", 1)
                col_name = f"epoch_sliding_duration{duration}s_step{step}s"
                df[col_name] = "[0]"  # JSON list format for consistency
            
            return df
        
        # Normal epoching for therapy
        logger.info(f"Task is {task}: assigning epochs based on methods configuration")
        
        methods = self.epoching_config.get("methods", {})
        
        # Method 1: Fixed window (with overlap - samples can be in multiple epochs)
        if methods.get("fixed", {}).get("enabled", False):
            fixed = methods["fixed"]
            duration = fixed.get("duration", 30)
            overlap = fixed.get("overlap", 5)
            min_ratio = fixed.get("min_duration_ratio", 0.0)
            step = duration - overlap
            
            col_name = f"epoch_fixed_duration{duration}s_overlap{overlap}s"
            
            # Each sample can belong to multiple epochs (due to overlap)
            epoch_lists = []
            for t in time:
                # Find all epochs that contain this time point
                # Epoch i spans [i*step, i*step + duration[
                epochs = []
                epoch_start = 0
                epoch_id = 0
                while epoch_start <= t:
                    epoch_end = epoch_start + duration
                    if epoch_start <= t < epoch_end:
                        epochs.append(epoch_id)
                    epoch_start += step
                    epoch_id += 1
                
                # Convert list to JSON format string
                epoch_lists.append(str(epochs) if epochs else '[]')
            
            df[col_name] = epoch_lists
        
        # Method 2: N-split (single epoch per sample, stored as JSON list for consistency)
        if methods.get("nsplit", {}).get("enabled", False):
            nsplit = methods["nsplit"]
            n_epochs = nsplit.get("n_epochs", 120)
            
            col_name = f"epoch_nsplit{n_epochs}"
            # Convert to JSON list format: [0], [1], [2], etc.
            epoch_ids = self.assign_nsplit_epochs(time, n_epochs)
            df[col_name] = ['[' + str(eid) + ']' for eid in epoch_ids]
        
        # Method 3: Sliding window (samples can be in multiple epochs)
        if methods.get("sliding", {}).get("enabled", False):
            sliding = methods["sliding"]
            duration = sliding.get("duration", 30)
            step = sliding.get("step", 1)
            min_ratio = sliding.get("min_duration_ratio", 0.0)
            
            col_name = f"epoch_sliding_duration{duration}s_step{step}s"
            
            # Each sample can belong to multiple epochs (due to sliding windows)
            epoch_lists = []
            for t in time:
                # Find all epochs that contain this time point
                # Epoch at index i starts at i*step and spans [i*step, i*step + duration[
                epochs = []
                
                # Find first epoch that could contain t
                # Epoch i contains t if: i*step <= t < i*step + duration
                # i.e., t - duration < i*step <= t
                # i.e., (t - duration)/step < i <= t/step
                first_epoch = max(0, int(np.floor((t - duration + step) / step)))
                last_epoch = int(np.floor(t / step))
                
                for epoch_id in range(first_epoch, last_epoch + 1):
                    epoch_start = epoch_id * step
                    epoch_end = epoch_start + duration
                    if epoch_start <= t < epoch_end:
                        epochs.append(epoch_id)
                
                # Convert list to JSON format string
                epoch_lists.append(str(epochs) if epochs else '[]')
            
            df[col_name] = epoch_lists
        
        logger.info(f"Added {len([c for c in df.columns if c.startswith('epoch_')])} epoch columns")
        
        return df
