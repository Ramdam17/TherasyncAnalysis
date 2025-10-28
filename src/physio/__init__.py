"""
Physiological data processing modules for Therasync Pipeline.

This package contains modules for processing BVP, EDA, HR and other
physiological signals from Empatica devices.

Authors: Lena Adel, Remy Ramadour
"""

from src.physio.bvp_loader import BVPLoader
from src.physio.bvp_cleaner import BVPCleaner
from src.physio.bvp_metrics import BVPMetricsExtractor
from src.physio.bvp_bids_writer import BVPBIDSWriter

from src.physio.eda_loader import EDALoader
from src.physio.eda_cleaner import EDACleaner
from src.physio.eda_metrics import EDAMetricsExtractor
from src.physio.eda_bids_writer import EDABIDSWriter

__all__ = [
    # BVP Pipeline
    'BVPLoader',
    'BVPCleaner',
    'BVPMetricsExtractor',
    'BVPBIDSWriter',
    # EDA Pipeline
    'EDALoader',
    'EDACleaner',
    'EDAMetricsExtractor',
    'EDABIDSWriter',
]