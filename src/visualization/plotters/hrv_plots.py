"""
HRV Plots Module.

Implements HRV-specific visualizations:
- Visualization #2: Poincaré Plot
- Visualization #3: Autonomic Balance (Frequency Domain)

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from typing import Dict, Optional
from pathlib import Path

from ..config import (
    COLORS, ALPHA, FIGSIZE, FONTSIZE, LINEWIDTH, MARKERSIZE,
    apply_plot_style, get_moment_color, METRIC_LABELS
)


def plot_poincare_hrv(
    data: Dict,
    output_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Visualization #2: Poincaré Plot for HRV Analysis.
    
    Creates scatter plots of RR(n) vs RR(n+1) with SD1/SD2 ellipse.
    Each moment displayed in separate subplot for clarity.
    
    Args:
        data: Dictionary containing 'bvp' data with signals
        output_path: Where to save the figure
        show: Whether to display the figure
    
    Returns:
        Figure object
    """
    apply_plot_style()
    
    bvp_data = data.get('bvp', {})
    
    if not bvp_data or 'signals' not in bvp_data:
        fig, ax = plt.subplots(figsize=FIGSIZE['medium'])
        ax.text(0.5, 0.5, 'No BVP data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE['label'])
        return fig
    
    moments = ['restingstate', 'therapy']
    available_moments = [m for m in moments if m in bvp_data.get('signals', {})]
    
    if not available_moments:
        fig, ax = plt.subplots(figsize=FIGSIZE['medium'])
        ax.text(0.5, 0.5, 'No moment data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE['label'])
        return fig
    
    # Create subplots: one per moment (side by side)
    n_moments = len(available_moments)
    fig, axes = plt.subplots(1, n_moments, figsize=(8 * n_moments, 7))
    
    # Handle single moment case (axes is not a list)
    if n_moments == 1:
        axes = [axes]
    
    # Global limits for consistent scaling across subplots
    all_rr_n, all_rr_n1 = [], []
    moment_data = {}
    
    # First pass: collect data and calculate global limits
    for moment in available_moments:
        signals = bvp_data['signals'][moment]
        
        # Extract RR intervals from peaks
        if 'PPG_Peaks' not in signals.columns:
            continue
        
        peaks_idx = signals[signals['PPG_Peaks'] == 1].index
        if len(peaks_idx) < 2:
            continue
        
        # Calculate RR intervals (time between consecutive peaks)
        peak_times = signals.loc[peaks_idx, 'time'].values
        rr_intervals = np.diff(peak_times) * 1000  # Convert to ms
        
        if len(rr_intervals) < 2:
            continue
        
        # Create Poincaré plot data: RR(n) vs RR(n+1)
        rr_n = rr_intervals[:-1]
        rr_n1 = rr_intervals[1:]
        
        all_rr_n.extend(rr_n)
        all_rr_n1.extend(rr_n1)
        
        # Calculate SD1 and SD2
        diff = rr_n1 - rr_n
        sum_rr = rr_n1 + rr_n
        sd1 = np.std(diff) / np.sqrt(2)
        sd2 = np.std(sum_rr) / np.sqrt(2)
        
        moment_data[moment] = {
            'rr_n': rr_n,
            'rr_n1': rr_n1,
            'sd1': sd1,
            'sd2': sd2,
            'center_x': np.mean(rr_n),
            'center_y': np.mean(rr_n1)
        }
    
    # Set fixed physiologically relevant limits
    # Normal RR intervals: ~500-1500ms, extended to 2400ms for safety
    lims = [0, 2400]
    
    # Second pass: plot each moment in its own subplot
    for idx, moment in enumerate(available_moments):
        ax = axes[idx]
        color = get_moment_color(moment)
        
        if moment not in moment_data:
            ax.text(0.5, 0.5, f'Insufficient data for {moment}',
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=FONTSIZE['label'])
            continue
        
        data_m = moment_data[moment]
        
        # Scatter plot
        ax.scatter(data_m['rr_n'], data_m['rr_n1'], 
                  c=color, alpha=ALPHA['high'],
                  s=MARKERSIZE['medium'], 
                  edgecolors='white', linewidths=0.5)
        
        # Draw ellipse (2*SD to cover ~95% of points)
        ellipse = Ellipse((data_m['center_x'], data_m['center_y']), 
                         width=2*data_m['sd2'], height=2*data_m['sd1'],
                         angle=45, facecolor='none',
                         edgecolor=color, linewidth=LINEWIDTH['thick'],
                         linestyle='--', alpha=ALPHA['line'])
        ax.add_patch(ellipse)
        
        # Add identity line
        ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=LINEWIDTH['thin'], zorder=0)
        
        # Set equal aspect and limits
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect('equal', adjustable='box')
        
        # Labels
        ax.set_xlabel('RR(n) [ms]', fontsize=FONTSIZE['label'])
        if idx == 0:  # Only first subplot gets y-label
            ax.set_ylabel('RR(n+1) [ms]', fontsize=FONTSIZE['label'])
        
        # Title with SD1/SD2
        ax.set_title(f'{moment.capitalize()}\nSD1={data_m["sd1"]:.1f}ms, SD2={data_m["sd2"]:.1f}ms',
                    fontsize=FONTSIZE['subtitle'], color=color, fontweight='bold')
        
        # Grid
        ax.grid(True, alpha=ALPHA['fill'])
    
    # Overall title
    subject = data.get('subject', 'Unknown')
    session = data.get('session', 'Unknown')
    fig.suptitle(f'Poincaré Plot - HRV Analysis - Subject {subject}, Session {session}',
                fontsize=FONTSIZE['title'], fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig
    ax.plot(lims, lims, 'k--', alpha=0.3, linewidth=LINEWIDTH['thin'], 
           label='Identity Line', zorder=0)
    
    ax.set_xlabel('RR(n) - Current Interval (ms)', fontsize=FONTSIZE['label'])
    ax.set_ylabel('RR(n+1) - Next Interval (ms)', fontsize=FONTSIZE['label'])
    ax.set_title(f'Poincaré Plot - HRV Analysis\nSubject {data.get("subject", "Unknown")}, Session {data.get("session", "Unknown")}',
                fontsize=FONTSIZE['title'], fontweight='bold')
    ax.legend(loc='lower right', fontsize=FONTSIZE['legend'])
    ax.grid(True, alpha=ALPHA['fill'])
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_autonomic_balance(
    data: Dict,
    output_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Visualization #3: Autonomic Balance (Frequency Domain).
    
    Stacked bar chart showing LF and HF power with LF/HF ratio line.
    
    Args:
        data: Dictionary containing 'bvp' data with metrics
        output_path: Where to save the figure
        show: Whether to display the figure
    
    Returns:
        Figure object
    """
    apply_plot_style()
    
    fig, ax1 = plt.subplots(figsize=FIGSIZE['medium'])
    
    bvp_data = data.get('bvp', {})
    
    if not bvp_data or 'metrics' not in bvp_data or bvp_data['metrics'] is None:
        ax1.text(0.5, 0.5, 'No BVP metrics available', ha='center', va='center',
                transform=ax1.transAxes, fontsize=FONTSIZE['label'])
        return fig
    
    metrics = bvp_data['metrics']
    
    # Check required columns
    required_cols = ['moment', 'HRV_LF', 'HRV_HF', 'HRV_LFHF']
    if not all(col in metrics.columns for col in required_cols):
        ax1.text(0.5, 0.5, 'Missing required HRV metrics', ha='center', va='center',
                transform=ax1.transAxes, fontsize=FONTSIZE['label'])
        return fig
    
    # Extract metrics
    moments = metrics['moment'].values
    lf_power = metrics['HRV_LF'].values
    hf_power = metrics['HRV_HF'].values
    lfhf_ratio = metrics['HRV_LFHF'].values
    
    x = np.arange(len(moments))
    width = 0.6
    
    # Stacked bar chart
    bar1 = ax1.bar(x, lf_power, width, label='LF Power',
                   color=COLORS['lf'], alpha=ALPHA['solid'])
    bar2 = ax1.bar(x, hf_power, width, bottom=lf_power, label='HF Power',
                   color=COLORS['hf'], alpha=ALPHA['solid'])
    
    ax1.set_xlabel('Moment', fontsize=FONTSIZE['label'])
    ax1.set_ylabel('Power (ms²)', fontsize=FONTSIZE['label'])
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.capitalize() for m in moments], fontsize=FONTSIZE['tick'])
    ax1.legend(loc='upper left', fontsize=FONTSIZE['legend'])
    ax1.grid(True, alpha=ALPHA['fill'], axis='y')
    
    # Add LF/HF ratio on secondary axis
    ax2 = ax1.twinx()
    line = ax2.plot(x, lfhf_ratio, color=COLORS['scr'], marker='o',
                   linewidth=LINEWIDTH['thick'], markersize=MARKERSIZE['large'],
                   label='LF/HF Ratio', zorder=10)
    ax2.set_ylabel('LF/HF Ratio', fontsize=FONTSIZE['label'])
    ax2.legend(loc='upper right', fontsize=FONTSIZE['legend'])
    
    # Add reference zone for optimal LF/HF ratio (0.5-3.0)
    ax2.axhspan(0.5, 3.0, color=COLORS['good'], alpha=0.1, zorder=0)
    ax2.text(len(moments)-0.5, 1.75, 'Optimal\nRange',
            fontsize=FONTSIZE['annotation'], ha='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Title
    ax1.set_title(f'Autonomic Balance - Frequency Domain Analysis\nSubject {data.get("subject", "Unknown")}, Session {data.get("session", "Unknown")}',
                 fontsize=FONTSIZE['title'], fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(str(output_path), dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig
