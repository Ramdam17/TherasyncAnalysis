"""
BVP Metrics Extractor for TherasyncPipeline.

This module provides functionality to extract Heart Rate Variability (HRV) and other
cardiovascular metrics from processed BVP data using NeuroKit2.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any

import pandas as pd
import numpy as np
import neurokit2 as nk

from src.core.config_loader import ConfigLoader


logger = logging.getLogger(__name__)


class BVPMetricsExtractor:
    """
    Extract HRV and cardiovascular metrics from processed BVP data.
    
    This class implements the essential HRV metrics extraction using NeuroKit2,
    supporting both session-level analysis and future epoched analysis capabilities.
    """
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """
        Initialize the BVP metrics extractor with configuration.
        
        Args:
            config_path: Path to configuration file. If None, uses default config.
        """
        self.config = ConfigLoader(config_path)
        
        # Get BVP metrics configuration
        self.bvp_config = self.config.get('physio.bvp', {})
        self.metrics_config = self.bvp_config.get('metrics', {})
        
        # Get selected metrics
        self.selected_metrics = self.metrics_config.get('selected_metrics', {})
        self.time_domain_metrics = self.selected_metrics.get('time_domain', [])
        self.frequency_domain_metrics = self.selected_metrics.get('frequency_domain', [])
        self.nonlinear_metrics = self.selected_metrics.get('nonlinear', [])
        
        # Epoched analysis configuration (for future implementation)
        self.epoched_config = self.metrics_config.get('epoched_analysis', {})
        self.epoched_enabled = self.epoched_config.get('enabled', False)
        
        logger.info(
            f"BVP Metrics Extractor initialized: "
            f"{len(self.time_domain_metrics)} time-domain, "
            f"{len(self.frequency_domain_metrics)} frequency-domain, "
            f"{len(self.nonlinear_metrics)} nonlinear metrics"
        )
    
    def extract_session_metrics(
        self, 
        processed_results: Dict[str, Tuple[pd.DataFrame, Dict]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Extract HRV metrics for entire sessions/moments.
        
        Args:
            processed_results: Output from BVPCleaner.process_moment_signals()
                              Format: {moment: (processed_signals, processing_info)}
            
        Returns:
            Dictionary with extracted metrics for each moment.
            Format: {moment: {metric_name: value}}
        """
        session_metrics = {}
        
        for moment, (processed_signals, processing_info) in processed_results.items():
            try:
                # Extract peaks from processing info
                peaks = processing_info.get('PPG_Peaks', [])
                sampling_rate = processing_info.get('sampling_rate', 64)
                
                # Validate peaks for HRV analysis
                if not self._validate_peaks_for_hrv(peaks, moment):
                    logger.warning(f"Skipping HRV analysis for {moment}: insufficient peaks")
                    session_metrics[moment] = self._get_empty_metrics_dict()
                    continue
                
                # Extract HRV metrics
                moment_metrics = self._extract_hrv_metrics(
                    peaks, sampling_rate, moment
                )
                
                # Add basic signal quality metrics
                quality_metrics = self._extract_signal_quality_metrics(
                    processed_signals, processing_info, moment
                )
                moment_metrics.update(quality_metrics)
                
                session_metrics[moment] = moment_metrics
                
                logger.info(
                    f"Extracted {len(moment_metrics)} metrics for {moment}: "
                    f"HRV_MeanNN={moment_metrics.get('HRV_MeanNN', 'N/A'):.1f}ms"
                )
                
            except Exception as e:
                logger.error(f"Failed to extract metrics for {moment}: {e}")
                session_metrics[moment] = self._get_empty_metrics_dict()
                continue
        
        return session_metrics
    
    def _extract_hrv_metrics(
        self, 
        peaks: Union[List, np.ndarray], 
        sampling_rate: int, 
        moment: str
    ) -> Dict[str, float]:
        """
        Extract HRV metrics from peaks using NeuroKit2.
        
        Args:
            peaks: Array of peak indices
            sampling_rate: Sampling rate in Hz
            moment: Moment name for logging
            
        Returns:
            Dictionary of extracted HRV metrics
        """
        metrics = {}
        peaks_array = np.array(peaks)
        
        try:
            # Extract time-domain metrics
            if self.time_domain_metrics:
                time_metrics = nk.hrv_time(peaks_array, sampling_rate=sampling_rate)
                for metric in self.time_domain_metrics:
                    if metric in time_metrics.columns:
                        metrics[metric] = float(time_metrics[metric].iloc[0])
                    else:
                        logger.warning(f"Time-domain metric {metric} not found for {moment}")
                        metrics[metric] = np.nan
            
            # Extract frequency-domain metrics
            if self.frequency_domain_metrics:
                try:
                    freq_metrics = nk.hrv_frequency(peaks_array, sampling_rate=sampling_rate)
                    for metric in self.frequency_domain_metrics:
                        if metric in freq_metrics.columns:
                            metrics[metric] = float(freq_metrics[metric].iloc[0])
                        else:
                            logger.warning(f"Frequency-domain metric {metric} not found for {moment}")
                            metrics[metric] = np.nan
                except Exception as e:
                    logger.warning(f"Frequency-domain analysis failed for {moment}: {e}")
                    for metric in self.frequency_domain_metrics:
                        metrics[metric] = np.nan
            
            # Extract nonlinear metrics
            if self.nonlinear_metrics:
                try:
                    nonlinear_metrics = nk.hrv_nonlinear(peaks_array, sampling_rate=sampling_rate)
                    for metric in self.nonlinear_metrics:
                        if metric in nonlinear_metrics.columns:
                            metrics[metric] = float(nonlinear_metrics[metric].iloc[0])
                        else:
                            logger.warning(f"Nonlinear metric {metric} not found for {moment}")
                            metrics[metric] = np.nan
                except Exception as e:
                    logger.warning(f"Nonlinear analysis failed for {moment}: {e}")
                    for metric in self.nonlinear_metrics:
                        metrics[metric] = np.nan
            
            return metrics
            
        except Exception as e:
            logger.error(f"HRV extraction failed for {moment}: {e}")
            return self._get_empty_hrv_metrics_dict()
    
    def _extract_signal_quality_metrics(
        self, 
        processed_signals: pd.DataFrame, 
        processing_info: Dict, 
        moment: str
    ) -> Dict[str, float]:
        """
        Extract signal quality and basic metrics.
        
        Args:
            processed_signals: Processed signals DataFrame
            processing_info: Processing information dictionary
            moment: Moment name for logging
            
        Returns:
            Dictionary of quality metrics
        """
        quality_metrics = {}
        
        try:
            # Number of detected peaks
            peaks = processing_info.get('PPG_Peaks', [])
            quality_metrics['BVP_NumPeaks'] = len(peaks)
            
            # Signal duration
            sampling_rate = processing_info.get('sampling_rate', 64)
            duration = len(processed_signals) / sampling_rate
            quality_metrics['BVP_Duration'] = duration
            
            # Peak rate (peaks per minute)
            if duration > 0:
                quality_metrics['BVP_PeakRate'] = (len(peaks) / duration) * 60
            else:
                quality_metrics['BVP_PeakRate'] = np.nan
            
            # Mean signal quality if available
            if 'PPG_Quality' in processed_signals.columns:
                quality_scores = processed_signals['PPG_Quality'].dropna()
                if not quality_scores.empty:
                    quality_metrics['BVP_MeanQuality'] = float(quality_scores.mean())
                    quality_metrics['BVP_QualityStd'] = float(quality_scores.std())
                else:
                    quality_metrics['BVP_MeanQuality'] = np.nan
                    quality_metrics['BVP_QualityStd'] = np.nan
            
            # Signal amplitude metrics from cleaned signal
            if 'PPG_Clean' in processed_signals.columns:
                clean_signal = processed_signals['PPG_Clean'].dropna()
                if not clean_signal.empty:
                    quality_metrics['BVP_MeanAmplitude'] = float(clean_signal.mean())
                    quality_metrics['BVP_StdAmplitude'] = float(clean_signal.std())
                    quality_metrics['BVP_RangeAmplitude'] = float(clean_signal.max() - clean_signal.min())
                else:
                    quality_metrics['BVP_MeanAmplitude'] = np.nan
                    quality_metrics['BVP_StdAmplitude'] = np.nan
                    quality_metrics['BVP_RangeAmplitude'] = np.nan
            
        except Exception as e:
            logger.warning(f"Signal quality extraction failed for {moment}: {e}")
        
        return quality_metrics
    
    def _validate_peaks_for_hrv(
        self, 
        peaks: Union[List, np.ndarray], 
        moment: str
    ) -> bool:
        """
        Validate that peaks are sufficient for HRV analysis.
        
        Args:
            peaks: Array of peak indices
            moment: Moment name for logging
            
        Returns:
            True if peaks are sufficient for HRV analysis
        """
        if len(peaks) < 10:
            logger.warning(f"Insufficient peaks for HRV analysis in {moment}: {len(peaks)} < 10")
            return False
        
        # Check for reasonable peak intervals (avoid artifacts)
        peaks_array = np.array(peaks)
        if len(peaks_array) > 1:
            intervals = np.diff(peaks_array)
            # Check for very short intervals (< 200ms at 64Hz = 12.8 samples)
            min_interval = 0.2 * 64  # 200ms in samples
            if np.any(intervals < min_interval):
                short_intervals = np.sum(intervals < min_interval)
                logger.warning(
                    f"Found {short_intervals} very short intervals in {moment}, "
                    f"may indicate artifacts"
                )
        
        return True
    
    def _get_empty_metrics_dict(self) -> Dict[str, float]:
        """Get dictionary with all configured metrics set to NaN."""
        metrics = {}
        
        # Add all configured metrics with NaN values
        for metric in self.time_domain_metrics:
            metrics[metric] = np.nan
        for metric in self.frequency_domain_metrics:
            metrics[metric] = np.nan
        for metric in self.nonlinear_metrics:
            metrics[metric] = np.nan
        
        # Add quality metrics
        quality_metrics = [
            'BVP_NumPeaks', 'BVP_Duration', 'BVP_PeakRate',
            'BVP_MeanQuality', 'BVP_QualityStd',
            'BVP_MeanAmplitude', 'BVP_StdAmplitude', 'BVP_RangeAmplitude'
        ]
        for metric in quality_metrics:
            metrics[metric] = np.nan
            
        return metrics
    
    def _get_empty_hrv_metrics_dict(self) -> Dict[str, float]:
        """Get dictionary with HRV metrics set to NaN."""
        metrics = {}
        
        for metric in self.time_domain_metrics:
            metrics[metric] = np.nan
        for metric in self.frequency_domain_metrics:
            metrics[metric] = np.nan
        for metric in self.nonlinear_metrics:
            metrics[metric] = np.nan
            
        return metrics
    
    def get_metrics_summary(self, session_metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Convert session metrics to a summary DataFrame.
        
        Args:
            session_metrics: Output from extract_session_metrics()
            
        Returns:
            DataFrame with moments as rows and metrics as columns
        """
        if not session_metrics:
            logger.warning("No session metrics to summarize")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(session_metrics, orient='index')
        
        # Add moment names as a column for convenience
        df.index.name = 'moment'
        df = df.reset_index()
        
        logger.info(f"Created metrics summary: {len(df)} moments × {len(df.columns)-1} metrics")
        
        return df
    
    def compare_moments(
        self, 
        session_metrics: Dict[str, Dict[str, float]],
        baseline_moment: str = "restingstate",
        comparison_moment: str = "therapy"
    ) -> Dict[str, float]:
        """
        Compare metrics between two moments (e.g., resting vs therapy).
        
        Args:
            session_metrics: Output from extract_session_metrics()
            baseline_moment: Name of baseline moment
            comparison_moment: Name of comparison moment
            
        Returns:
            Dictionary of differences (comparison - baseline)
        """
        if baseline_moment not in session_metrics:
            logger.error(f"Baseline moment '{baseline_moment}' not found in session metrics")
            return {}
        
        if comparison_moment not in session_metrics:
            logger.error(f"Comparison moment '{comparison_moment}' not found in session metrics")
            return {}
        
        baseline_metrics = session_metrics[baseline_moment]
        comparison_metrics = session_metrics[comparison_moment]
        
        differences = {}
        
        for metric in baseline_metrics:
            baseline_val = baseline_metrics.get(metric, np.nan)
            comparison_val = comparison_metrics.get(metric, np.nan)
            
            # Ensure we have valid float values
            if baseline_val is not None and comparison_val is not None:
                baseline_float = float(baseline_val)
                comparison_float = float(comparison_val)
                
                if not (np.isnan(baseline_float) or np.isnan(comparison_float)):
                    differences[f"{metric}_diff"] = comparison_float - baseline_float
                    if baseline_float != 0:
                        differences[f"{metric}_pct_change"] = ((comparison_float - baseline_float) / baseline_float) * 100
                    else:
                        differences[f"{metric}_pct_change"] = np.nan
                else:
                    differences[f"{metric}_diff"] = np.nan
                    differences[f"{metric}_pct_change"] = np.nan
            else:
                differences[f"{metric}_diff"] = np.nan
                differences[f"{metric}_pct_change"] = np.nan
        
        logger.info(
            f"Computed {len(differences)} comparison metrics: "
            f"{comparison_moment} vs {baseline_moment}"
        )
        
        return differences
    
    def get_configured_metrics_list(self) -> List[str]:
        """
        Get list of all configured metrics that will be extracted.
        
        Returns:
            List of metric names
        """
        all_metrics = []
        all_metrics.extend(self.time_domain_metrics)
        all_metrics.extend(self.frequency_domain_metrics)
        all_metrics.extend(self.nonlinear_metrics)
        
        # Add quality metrics
        quality_metrics = [
            'BVP_NumPeaks', 'BVP_Duration', 'BVP_PeakRate',
            'BVP_MeanQuality', 'BVP_QualityStd',
            'BVP_MeanAmplitude', 'BVP_StdAmplitude', 'BVP_RangeAmplitude'
        ]
        all_metrics.extend(quality_metrics)
        
        return all_metrics
    
    # TODO: Future implementation for epoched analysis
    def extract_epoched_metrics(
        self, 
        processed_signals: pd.DataFrame, 
        processing_info: Dict,
        moment: str
    ) -> pd.DataFrame:
        """
        Extract HRV metrics from sliding windows (future implementation).
        
        This method will implement the 30-second sliding window approach
        with 1-second steps for dynamic HRV analysis.
        
        Args:
            processed_signals: Processed signals DataFrame
            processing_info: Processing information dictionary
            moment: Moment name
            
        Returns:
            DataFrame with time-series of HRV metrics
        """
        if not self.epoched_enabled:
            logger.info("Epoched analysis not enabled in configuration")
            return pd.DataFrame()
        
        # TODO: Implement epoched analysis
        logger.info("Epoched HRV analysis not yet implemented")
        return pd.DataFrame()