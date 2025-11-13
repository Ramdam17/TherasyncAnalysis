"""
Visualization Configuration Module.

Defines plotting styles, colors, and parameters for consistent visualizations.

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

from typing import Dict
import matplotlib.pyplot as plt

# Color Schemes
COLORS = {
    # Moment colors (legacy support for named moments)
    'restingstate': '#3498db',  # Blue
    'therapy': '#e74c3c',  # Red
    
    # Signal colors
    'bvp': '#9b59b6',  # Purple
    'eda': '#2ecc71',  # Green
    'hr': '#e67e22',  # Orange
    
    # EDA components
    'tonic': '#3498db',  # Blue
    'phasic': '#2ecc71',  # Green
    'scr': '#e74c3c',  # Red
    
    # HRV components
    'lf': '#2ecc71',  # Green (sympathetic + parasympathetic)
    'hf': '#3498db',  # Blue (parasympathetic)
    
    # Quality indicators
    'good': '#2ecc71',  # Green
    'medium': '#f39c12',  # Yellow
    'poor': '#e74c3c',  # Red
    
    # Neutral colors
    'gray': '#95a5a6',
    'dark_gray': '#34495e',
    'light_gray': '#ecg0f1',
}

# Moment color palette (supports up to 8 distinct moments, with modulo fallback)
MOMENT_COLORS = [
    '#3498db',  # Blue
    '#e74c3c',  # Red
    '#2ecc71',  # Green
    '#f39c12',  # Orange
    '#9b59b6',  # Purple
    '#1abc9c',  # Teal
    '#e67e22',  # Dark orange
    '#34495e',  # Dark gray
]

# Map known moment names to indices for backward compatibility
MOMENT_NAME_TO_INDEX = {
    'restingstate': 0,
    'therapy': 1,
}

# Transparency levels
ALPHA = {
    'low': 0.2,
    'medium': 0.4,
    'high': 0.7,
    'fill': 0.3,
    'overlay': 0.5,
    'scatter': 0.6,
    'line': 0.8,
    'solid': 1.0
}

# Figure sizes (width, height in inches)
FIGSIZE = {
    'small': (8, 6),
    'medium': (12, 8),
    'large': (16, 10),
    'dashboard': (16, 12),
    'wide': (14, 6),
}

# Font sizes
FONTSIZE = {
    'title': 16,
    'subtitle': 14,
    'label': 12,
    'tick': 10,
    'legend': 10,
    'annotation': 9,
    'small': 8,
}

# Line widths
LINEWIDTH = {
    'thin': 0.5,
    'normal': 1.0,
    'signal': 1.2,
    'medium': 1.5,
    'thick': 2.0,
    'extra_thick': 3.0,
}

# Marker sizes
MARKERSIZE = {
    'tiny': 2,
    'small': 4,
    'medium': 6,
    'large': 8,
    'extra_large': 10,
}

# Plot Style Configuration
PLOT_STYLE = {
    'style': 'seaborn-v0_8-darkgrid',  # Matplotlib style
    'context': 'notebook',  # Seaborn context
    'palette': 'deep',  # Seaborn palette
    'grid': True,
    'grid_alpha': 0.3,
}

# DPI settings
DPI = {
    'screen': 100,
    'print': 300,
    'presentation': 150,
}

# Export formats
EXPORT_FORMATS = ['png', 'pdf', 'svg']

# Moment labels (for display)
MOMENT_LABELS = {
    'restingstate': 'Resting State',
    'therapy': 'Therapy Session',
}

# Metric labels and units
METRIC_LABELS = {
    # BVP/HRV metrics
    'HRV_MeanNN': 'Mean NN Interval (ms)',
    'HRV_SDNN': 'SDNN (ms)',
    'HRV_RMSSD': 'RMSSD (ms)',
    'HRV_pNN50': 'pNN50 (%)',
    'HRV_CVNN': 'CV of NN Intervals',
    'HRV_LF': 'LF Power (ms²)',
    'HRV_HF': 'HF Power (ms²)',
    'HRV_TP': 'Total Power (ms²)',
    'HRV_LFHF': 'LF/HF Ratio',
    'HRV_SD1': 'SD1 (ms)',
    'HRV_SD2': 'SD2 (ms)',
    'HRV_SampEn': 'Sample Entropy',
    
    # EDA metrics
    'SCR_Peaks_N': 'Number of SCRs',
    'SCR_Peaks_Rate': 'SCR Rate (per min)',
    'SCR_Peaks_Amplitude_Mean': 'Mean SCR Amplitude (µS)',
    'SCR_Peaks_Amplitude_Max': 'Max SCR Amplitude (µS)',
    'SCR_Peaks_Amplitude_SD': 'SCR Amplitude SD (µS)',
    'SCR_RiseTime_Mean': 'Mean Rise Time (s)',
    'SCR_RecoveryTime_Mean': 'Mean Recovery Time (s)',
    'EDA_Tonic_Mean': 'Mean Tonic EDA (µS)',
    'EDA_Tonic_SD': 'Tonic EDA SD (µS)',
    'EDA_Phasic_Mean': 'Mean Phasic EDA (µS)',
    'EDA_Phasic_SD': 'Phasic EDA SD (µS)',
    
    # HR metrics
    'HR_Mean': 'Mean HR (BPM)',
    'HR_SD': 'HR SD (BPM)',
    'HR_Min': 'Min HR (BPM)',
    'HR_Max': 'Max HR (BPM)',
    'HR_Range': 'HR Range (BPM)',
    'HR_Slope': 'HR Slope (BPM/min)',
}


def apply_plot_style():
    """Apply consistent plotting style to matplotlib."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    plt.rcParams['axes.edgecolor'] = COLORS['dark_gray']
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['grid.color'] = COLORS['gray']
    plt.rcParams['grid.alpha'] = PLOT_STYLE['grid_alpha']
    plt.rcParams['font.size'] = FONTSIZE['label']
    plt.rcParams['axes.titlesize'] = FONTSIZE['title']
    plt.rcParams['axes.labelsize'] = FONTSIZE['label']
    plt.rcParams['xtick.labelsize'] = FONTSIZE['tick']
    plt.rcParams['ytick.labelsize'] = FONTSIZE['tick']
    plt.rcParams['legend.fontsize'] = FONTSIZE['legend']


def get_moment_color(moment) -> str:
    """
    Get color for a specific moment.
    
    Supports both string names and integer indices.
    Uses modulo fallback for indices beyond the palette size.
    For unknown string names, generates a stable color based on hash.
    
    Args:
        moment: Either a moment name (str) or index (int)
    
    Returns:
        Hex color code
    
    Examples:
        get_moment_color('restingstate')  # '#3498db' (blue)
        get_moment_color('therapy')       # '#e74c3c' (red)
        get_moment_color(0)               # '#3498db' (blue)
        get_moment_color(5)               # '#1abc9c' (teal)
        get_moment_color(10)              # '#2ecc71' (green, wraps around)
        get_moment_color('baseline')      # Stable color based on hash
    """
    if isinstance(moment, int):
        # Direct index access with modulo fallback
        return MOMENT_COLORS[moment % len(MOMENT_COLORS)]
    elif isinstance(moment, str):
        # Try named moment first (backward compatibility)
        if moment in MOMENT_NAME_TO_INDEX:
            return MOMENT_COLORS[MOMENT_NAME_TO_INDEX[moment]]
        # For unknown names, use hash to get stable color index
        moment_hash = hash(moment)
        color_index = moment_hash % len(MOMENT_COLORS)
        return MOMENT_COLORS[color_index]
    else:
        return COLORS['gray']


def get_moment_label(moment: str, config: Dict = None) -> str:
    """
    Get display label for a moment.
    
    Priority order:
    1. Label from config.yaml override (if provided)
    2. Known moment labels from MOMENT_LABELS dict
    3. Auto-generated: capitalize and replace underscores
    
    Args:
        moment: Moment name (e.g., 'restingstate', 'baseline', 'intervention')
        config: Optional config dict with custom labels
    
    Returns:
        Formatted display label
    
    Examples:
        get_moment_label('restingstate')        # 'Resting State'
        get_moment_label('therapy')             # 'Therapy Session'
        get_moment_label('baseline')            # 'Baseline'
        get_moment_label('post_intervention')   # 'Post Intervention'
    """
    # Check config override
    if config and 'visualization' in config:
        moment_labels = config.get('visualization', {}).get('moment_labels', {})
        if moment in moment_labels:
            return moment_labels[moment]
    
    # Check known labels
    if moment in MOMENT_LABELS:
        return MOMENT_LABELS[moment]
    
    # Auto-generate: capitalize and replace underscores
    return moment.replace('_', ' ').title()


def get_moment_order(moment: str, moments_list: list) -> int:
    """
    Get the index/order of a moment in a list.
    
    This ensures consistent ordering across visualizations.
    
    Args:
        moment: Moment name
        moments_list: List of all available moments (sorted)
    
    Returns:
        Index of the moment in the list (0-based)
        Returns -1 if moment not found
    
    Examples:
        moments = ['baseline', 'restingstate', 'therapy']
        get_moment_order('restingstate', moments)  # 1
        get_moment_order('therapy', moments)       # 2
    """
    try:
        return moments_list.index(moment)
    except ValueError:
        return -1


def get_modality_color(modality: str) -> str:
    """Get color for a specific modality."""
    return COLORS.get(modality, COLORS['gray'])


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "5m 30s", "1h 15m")
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes:.0f}m {secs:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"


# Visualization output configuration
OUTPUT_CONFIG = {
    'base_path': 'data/derivatives/visualization/preprocessing',
    'figures_subdir': 'figures',
    'report_subdir': 'report',
    'summary_subdir': 'summary',
    'dpi': DPI['print'],
    'format': 'png',
    'bbox_inches': 'tight',
}
