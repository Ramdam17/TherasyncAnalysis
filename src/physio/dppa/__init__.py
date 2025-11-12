"""
DPPA (Dyadic Poincaré Plot Analysis) Module.

This module provides tools for analyzing physiological synchrony between dyads
using Inter-Centroid Distances (ICD) computed from Poincaré plot centroids.

Modules:
- poincare_calculator: Compute Poincaré centroids per participant/session/epoch
- centroid_loader: Load pre-computed centroid files
- icd_calculator: Calculate Inter-Centroid Distances between dyads
- dyad_config_loader: Load dyad configuration mappings
- dppa_writer: Export ICD results to BIDS-compliant CSV

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

from .poincare_calculator import PoincareCalculator
from .centroid_loader import CentroidLoader
from .dyad_config_loader import DyadConfigLoader
from .icd_calculator import ICDCalculator
from .dppa_writer import DPPAWriter

__all__ = ['PoincareCalculator', 'CentroidLoader', 'DyadConfigLoader', 'ICDCalculator', 'DPPAWriter']
