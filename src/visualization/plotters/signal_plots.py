"""
Signal Plots Module.

Implements time-series and temporal visualizations:
- Visualization #1: Multi-Signal Dashboard
- Visualization #6: HR Dynamics Timeline
- Visualization #10: Events Timeline

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from typing import Dict, Optional, Tuple
from pathlib import Path

from ..config import (
    COLORS, ALPHA, FIGSIZE, FONTSIZE, LINEWIDTH, MARKERSIZE,
    apply_plot_style, get_moment_color, format_duration
)


def plot_multisignal_dashboard(
    data: Dict,
    output_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Visualization #1: Multi-Signal Dashboard.
    
    Creates a 4-panel synchronized time-series plot showing:
    - Panel 1: BVP signal + detected peaks
    - Panel 2: HR instantaneous (BPM)
    - Panel 3: EDA tonic (line) + phasic (filled area)
    - Panel 4: SCR events (stem plot)
    
    Args:
        data: Dictionary containing 'bvp', 'eda', 'hr' data
        output_path: Where to save the figure
        show: Whether to display the figure
    
    Returns:
        Figure object
    """
    apply_plot_style()
    
    # Create figure with 4 subplots
    fig = plt.figure(figsize=FIGSIZE['dashboard'])
    gs = GridSpec(4, 1, height_ratios=[1, 1, 1, 0.8], hspace=0.3)
    
    # Collect all moments for x-axis alignment
    moments = ['restingstate', 'therapy']
    
    # ========== Panel 1: BVP Signal ==========
    ax1 = fig.add_subplot(gs[0, 0])
    plot_bvp_signal(ax1, data.get('bvp', {}), moments)
    ax1.set_title('Blood Volume Pulse (BVP)', fontsize=FONTSIZE['title'], fontweight='bold')
    ax1.set_ylabel('BVP (a.u.)', fontsize=FONTSIZE['label'])
    ax1.legend(loc='upper right', fontsize=FONTSIZE['legend'])
    
    # ========== Panel 2: Heart Rate ==========
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    plot_hr_signal(ax2, data.get('hr', {}), moments)
    ax2.set_title('Instantaneous Heart Rate', fontsize=FONTSIZE['title'], fontweight='bold')
    ax2.set_ylabel('HR (BPM)', fontsize=FONTSIZE['label'])
    ax2.legend(loc='upper right', fontsize=FONTSIZE['legend'])
    
    # ========== Panel 3: EDA Tonic + Phasic ==========
    ax3 = fig.add_subplot(gs[2, 0], sharex=ax1)
    plot_eda_signal(ax3, data.get('eda', {}), moments)
    ax3.set_title('Electrodermal Activity (EDA)', fontsize=FONTSIZE['title'], fontweight='bold')
    ax3.set_ylabel('EDA (µS)', fontsize=FONTSIZE['label'])
    ax3.legend(loc='upper right', fontsize=FONTSIZE['legend'])
    
    # ========== Panel 4: SCR Events ==========
    ax4 = fig.add_subplot(gs[3, 0], sharex=ax1)
    plot_scr_events(ax4, data.get('eda', {}), moments)
    ax4.set_title('Skin Conductance Responses (SCR)', fontsize=FONTSIZE['title'], fontweight='bold')
    ax4.set_ylabel('Amplitude (µS)', fontsize=FONTSIZE['label'])
    ax4.set_xlabel('Time (seconds)', fontsize=FONTSIZE['label'])
    
    # Add overall title
    subject = data.get('subject', 'Unknown')
    session = data.get('session', 'Unknown')
    fig.suptitle(
        f'Physiological Signals Dashboard - Subject {subject}, Session {session}',
        fontsize=FONTSIZE['title'] + 2,
        fontweight='bold',
        y=0.995
    )
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def plot_bvp_signal(ax: plt.Axes, bvp_data: Dict, moments: list):
    """Plot BVP signal with detected peaks."""
    if not bvp_data or 'signals' not in bvp_data:
        ax.text(0.5, 0.5, 'No BVP data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE['label'])
        return
    
    offset = 0
    for moment in moments:
        if moment not in bvp_data['signals']:
            continue
        
        signals = bvp_data['signals'][moment]
        color = get_moment_color(moment)
        
        # Plot cleaned BVP signal
        time = signals['time'].values + offset
        if 'PPG_Clean' in signals.columns:
            ax.plot(time, signals['PPG_Clean'], 
                   color=color, linewidth=LINEWIDTH['normal'],
                   label=moment.capitalize(), alpha=ALPHA['line'])
        
        # Mark detected peaks
        if 'PPG_Peaks' in signals.columns:
            peaks = signals[signals['PPG_Peaks'] == 1]
            if len(peaks) > 0:
                ax.scatter(peaks['time'].values + offset, peaks['PPG_Clean'].values,
                          color=COLORS['scr'], s=MARKERSIZE['small'], 
                          marker='o', zorder=5)
        
        # Update offset for next moment
        offset += signals['time'].max() + 10  # 10s gap between moments
    
    ax.grid(True, alpha=ALPHA['fill'])


def plot_hr_signal(ax: plt.Axes, hr_data: Dict, moments: list):
    """Plot instantaneous heart rate with zones."""
    if not hr_data or 'signals' not in hr_data:
        ax.text(0.5, 0.5, 'No HR data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE['label'])
        return
    
    offset = 0
    all_hr = []
    
    for moment in moments:
        if moment not in hr_data['signals']:
            continue
        
        signals = hr_data['signals'][moment]
        color = get_moment_color(moment)
        
        # Plot HR
        time = signals['time'].values + offset
        if 'HR' in signals.columns:
            hr_values = signals['HR'].values
            all_hr.extend(hr_values)
            ax.plot(time, hr_values, 
                   color=color, linewidth=LINEWIDTH['medium'],
                   label=moment.capitalize(), alpha=ALPHA['line'])
        
        offset += signals['time'].max() + 10
    
    # Add horizontal zones
    if all_hr:
        hr_mean = np.mean(all_hr)
        ax.axhline(hr_mean, color=COLORS['gray'], linestyle='--', 
                  linewidth=LINEWIDTH['thin'], alpha=0.5, label='Mean HR')
        
        # Elevated zone (Mean + 20 BPM)
        ax.axhspan(hr_mean + 20, ax.get_ylim()[1], 
                  color=COLORS['poor'], alpha=0.1, label='Elevated')
        
        # Rest zone (Mean - 20 BPM)
        ax.axhspan(ax.get_ylim()[0], max(40, hr_mean - 20), 
                  color=COLORS['good'], alpha=0.1, label='Rest')
    
    ax.grid(True, alpha=ALPHA['fill'])


def plot_eda_signal(ax: plt.Axes, eda_data: Dict, moments: list):
    """Plot EDA tonic (line) and phasic (filled area)."""
    if not eda_data or 'signals' not in eda_data:
        ax.text(0.5, 0.5, 'No EDA data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE['label'])
        return
    
    offset = 0
    
    for moment in moments:
        if moment not in eda_data['signals']:
            continue
        
        signals = eda_data['signals'][moment]
        color = get_moment_color(moment)
        
        time = signals['time'].values + offset
        
        # Plot tonic component (baseline)
        if 'EDA_Tonic' in signals.columns:
            ax.plot(time, signals['EDA_Tonic'], 
                   color=COLORS['tonic'], linewidth=LINEWIDTH['thick'],
                   label=f'{moment.capitalize()} - Tonic', alpha=ALPHA['line'])
        
        # Plot phasic component (filled area)
        if 'EDA_Phasic' in signals.columns:
            ax.fill_between(time, 0, signals['EDA_Phasic'],
                           color=COLORS['phasic'], alpha=ALPHA['fill'],
                           label=f'{moment.capitalize()} - Phasic')
        
        offset += signals['time'].max() + 10
    
    ax.grid(True, alpha=ALPHA['fill'])


def plot_scr_events(ax: plt.Axes, eda_data: Dict, moments: list):
    """Plot SCR events as stem plot."""
    if not eda_data or 'events' not in eda_data:
        ax.text(0.5, 0.5, 'No SCR events available', ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE['label'])
        return
    
    offset = 0
    
    for i, moment in enumerate(moments):
        if moment not in eda_data['events']:
            continue
        
        events = eda_data['events'][moment]
        color = get_moment_color(moment)
        
        if len(events) == 0:
            continue
        
        # Get SCR onset times and amplitudes
        if 'SCR_Onsets' in events.columns and 'SCR_Amplitude' in events.columns:
            onsets = events['SCR_Onsets'].values + offset
            amplitudes = events['SCR_Amplitude'].values
            
            # Stem plot
            markerline, stemlines, baseline = ax.stem(
                onsets, amplitudes,
                linefmt=color, markerfmt='o', 
                basefmt='k-'
            )
            markerline.set_markersize(MARKERSIZE['medium'])
            markerline.set_color(color)
            markerline.set_label(moment.capitalize())
            stemlines.set_linewidth(LINEWIDTH['normal'])
            stemlines.set_alpha(ALPHA['overlay'])
        
        # Update offset
        if moment in eda_data['signals']:
            offset += eda_data['signals'][moment]['time'].max() + 10
    
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=ALPHA['fill'])
    ax.legend(loc='upper right', fontsize=FONTSIZE['legend'])


def plot_hr_dynamics_timeline(
    data: Dict,
    output_path: str,
    show: bool = False
) -> None:
    """
    Visualization #6: HR Dynamics Timeline.
    
    Shows HR over time with color-coded zones (rest/moderate/elevated).
    
    Args:
        data: Dictionary containing 'hr' data
        output_path: Where to save the figure (string path)
        show: Whether to display the figure
    """
    apply_plot_style()
    
    fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
    
    hr_data = data.get('hr', {})
    
    if not hr_data or 'signals' not in hr_data:
        ax.text(0.5, 0.5, 'No HR data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE['label'])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Concatenate all moments
    all_hr = []
    all_time = []
    moment_boundaries = [0]
    offset = 0
    
    for moment in ['restingstate', 'therapy']:
        if moment not in hr_data['signals']:
            continue
        
        signals = hr_data['signals'][moment]
        time = signals['time'].values + offset
        hr_values = signals['HR'].values if 'HR' in signals.columns else []
        
        all_time.extend(time)
        all_hr.extend(hr_values)
        
        offset += signals['time'].max() + 10
        moment_boundaries.append(offset - 10)
    
    if not all_hr:
        ax.text(0.5, 0.5, 'No HR data available', ha='center', va='center',
                transform=ax.transAxes, fontsize=FONTSIZE['label'])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    all_time = np.array(all_time)
    all_hr = np.array(all_hr)
    hr_mean = np.mean(all_hr)
    
    # Define zones
    zones = [
        (0, hr_mean - 10, COLORS['good'], 'Rest'),
        (hr_mean - 10, hr_mean + 10, COLORS['medium'], 'Moderate'),
        (hr_mean + 10, 200, COLORS['poor'], 'Elevated')
    ]
    
    # Fill background zones
    for y_min, y_max, color, label in zones:
        ax.axhspan(y_min, y_max, color=color, alpha=0.1, label=label)
    
    # Plot HR line
    ax.plot(all_time, all_hr, color=COLORS['hr'], 
           linewidth=LINEWIDTH['thick'], label='HR', zorder=10)
    
    # Mark moment boundaries
    for boundary in moment_boundaries[1:-1]:
        ax.axvline(boundary, color=COLORS['dark_gray'], 
                  linestyle='--', linewidth=LINEWIDTH['thin'], alpha=0.5)
    
    # Annotations
    ax.axhline(hr_mean, color=COLORS['gray'], linestyle='--', 
              linewidth=LINEWIDTH['normal'], label=f'Mean ({hr_mean:.1f} BPM)')
    
    ax.set_xlabel('Time (seconds)', fontsize=FONTSIZE['label'])
    ax.set_ylabel('Heart Rate (BPM)', fontsize=FONTSIZE['label'])
    ax.set_title(f'Heart Rate Dynamics - Subject {data.get("subject", "Unknown")}, Session {data.get("session", "Unknown")}',
                fontsize=FONTSIZE['title'], fontweight='bold')
    ax.legend(loc='upper right', fontsize=FONTSIZE['legend'])
    ax.grid(True, alpha=ALPHA['fill'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if not show:
        plt.close()


def plot_events_timeline(
    data: Dict,
    output_path: Optional[Path] = None,
    show: bool = False
) -> plt.Figure:
    """
    Visualization #10: Multi-Modal Events Timeline.
    
    Shows 4 horizontal tracks:
    - Track 1: BVP peaks + quality
    - Track 2: SCR events
    - Track 3: HR zones
    - Track 4: Session markers
    
    Args:
        data: Dictionary containing all modality data
        output_path: Where to save the figure
        show: Whether to display the figure
    
    Returns:
        Figure object
    """
    apply_plot_style()
    
    fig, axes = plt.subplots(4, 1, figsize=FIGSIZE['wide'], 
                            sharex=True, gridspec_kw={'height_ratios': [1, 1, 1, 0.5]})
    
    # TODO: Implement full timeline visualization
    # This is a simplified version
    
    for ax in axes:
        ax.text(0.5, 0.5, 'Timeline visualization - Coming soon', 
               ha='center', va='center', transform=ax.transAxes)
    
    fig.suptitle('Physiological Events Timeline', fontsize=FONTSIZE['title'], fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig
