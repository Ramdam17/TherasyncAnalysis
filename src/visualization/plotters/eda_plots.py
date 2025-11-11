"""
EDA-specific visualization plots.

Implements visualizations #4, #5, #9:
- EDA arousal profile (tonic/phasic dual-axis)
- SCR cascade (waterfall plot by amplitude)
- SCR distribution (histogram + boxplot)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from typing import Dict
import pandas as pd

from ..config import (
    COLORS, FIGSIZE, FONTSIZE, LINEWIDTH, ALPHA,
    apply_plot_style, get_moment_color, format_duration
)


def plot_eda_arousal_profile(data: Dict, output_path: str, show: bool = False) -> None:
    """
    Plot EDA arousal profile with tonic and phasic components.
    
    Visualization #4: Dual-axis plot showing:
    - Tonic EDA (baseline arousal level) - primary y-axis
    - Phasic EDA (reactive component) - secondary y-axis
    - Moment differentiation with colors
    
    Args:
        data: Dictionary containing EDA signals with keys:
            - 'eda': Dict with 'restingstate' and 'therapy' DataFrames
                     Each should have: 'time', 'eda_tonic', 'eda_phasic'
        output_path: Path to save the PNG figure
    """
    apply_plot_style()
    
    fig, ax1 = plt.subplots(figsize=FIGSIZE['wide'])
    
    # Secondary axis for phasic
    ax2 = ax1.twinx()
    
    # Track time offset for concatenation
    time_offset = 0
    
    # Plot each moment
    for moment in ['restingstate', 'therapy']:
        if moment not in data['eda']:
            continue
        
        df = data['eda'][moment]
        if df.empty:
            continue
        
        time = df['time'].values + time_offset
        tonic = df['eda_tonic'].values if 'eda_tonic' in df.columns else None
        phasic = df['eda_phasic'].values if 'eda_phasic' in df.columns else None
        
        moment_color = get_moment_color(moment)
        
        # Plot tonic on primary axis (line)
        if tonic is not None:
            ax1.plot(
                time, tonic,
                color=moment_color,
                linewidth=LINEWIDTH['signal'],
                label=f'{moment} - Tonic',
                alpha=0.9
            )
        
        # Plot phasic on secondary axis (filled area)
        if phasic is not None:
            ax2.fill_between(
                time, 0, phasic,
                color=moment_color,
                alpha=ALPHA['low'],
                label=f'{moment} - Phasic'
            )
            ax2.plot(
                time, phasic,
                color=moment_color,
                linewidth=LINEWIDTH['thin'],
                alpha=0.5
            )
        
        # Update offset for next moment
        if len(time) > 0:
            time_offset = time[-1]
    
    # Formatting
    ax1.set_xlabel('Temps (s)', fontsize=FONTSIZE['label'])
    ax1.set_ylabel('EDA Tonique (µS)', fontsize=FONTSIZE['label'], color=COLORS['tonic'])
    ax2.set_ylabel('EDA Phasique (µS)', fontsize=FONTSIZE['label'], color=COLORS['phasic'])
    
    ax1.tick_params(axis='y', labelcolor=COLORS['tonic'], labelsize=FONTSIZE['tick'])
    ax2.tick_params(axis='y', labelcolor=COLORS['phasic'], labelsize=FONTSIZE['tick'])
    ax1.tick_params(axis='x', labelsize=FONTSIZE['tick'])
    
    ax1.set_title('Profil d\'Arousal EDA - Composantes Tonique et Phasique', 
                  fontsize=FONTSIZE['title'], fontweight='bold', pad=20)
    
    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, 
              loc='upper right', fontsize=FONTSIZE['legend'])
    
    ax1.grid(True, alpha=ALPHA['medium'], linestyle='--', linewidth=LINEWIDTH['thin'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scr_cascade(data: Dict, output_path: str, show: bool = False) -> None:
    """
    Plot SCR cascade/waterfall showing SCR events by amplitude.
    
    Visualization #5: Waterfall plot where each SCR event is shown as:
    - Stacked horizontal bars sorted by amplitude
    - Color-coded by moment (restingstate/therapy)
    - X-axis shows SCR amplitude, Y-axis shows event index
    
    Args:
        data: Dictionary containing EDA events with key:
            - 'eda': Dict with 'events' sub-dict containing:
                - 'restingstate': DataFrame with 'scr_amplitude', 'scr_onset'
                - 'therapy': DataFrame with 'scr_amplitude', 'scr_onset'
        output_path: Path to save the PNG figure
    """
    apply_plot_style()
    
    fig, ax = plt.subplots(figsize=FIGSIZE['wide'])
    
    # Collect all SCR events
    all_events = []
    for moment in ['restingstate', 'therapy']:
        if 'events' not in data['eda'] or moment not in data['eda']['events']:
            continue
        
        df = data['eda']['events'][moment]
        if df.empty or 'scr_amplitude' not in df.columns:
            continue
        
        for _, row in df.iterrows():
            all_events.append({
                'amplitude': row['scr_amplitude'],
                'onset': row.get('scr_onset', 0),
                'moment': moment
            })
    
    if not all_events:
        # No events to plot
        ax.text(0.5, 0.5, 'Aucun événement SCR détecté', 
               ha='center', va='center', fontsize=FONTSIZE['title'])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    else:
        # Sort by amplitude (ascending for waterfall effect)
        events_df = pd.DataFrame(all_events)
        events_df = events_df.sort_values('amplitude', ascending=True)
        
        # Plot cascading bars
        y_positions = np.arange(len(events_df))
        
        for idx, row in events_df.iterrows():
            color = get_moment_color(row['moment'])
            y_pos = np.where(y_positions == list(events_df.index).index(idx))[0][0]
            
            ax.barh(
                y_pos,
                row['amplitude'],
                height=0.8,
                color=color,
                alpha=ALPHA['high'],
                edgecolor='black',
                linewidth=0.5
            )
        
        # Formatting
        ax.set_xlabel('Amplitude SCR (µS)', fontsize=FONTSIZE['label'])
        ax.set_ylabel('Index d\'événement (trié par amplitude)', fontsize=FONTSIZE['label'])
        ax.set_title('Cascade des Réponses Cutanées Sympathiques (SCR)', 
                    fontsize=FONTSIZE['title'], fontweight='bold', pad=20)
        
        # Legend
        rest_patch = mpatches.Patch(
            color=get_moment_color('restingstate'), 
            label='Restingstate', 
            alpha=ALPHA['high']
        )
        therapy_patch = mpatches.Patch(
            color=get_moment_color('therapy'), 
            label='Therapy', 
            alpha=ALPHA['high']
        )
        ax.legend(handles=[rest_patch, therapy_patch], 
                 loc='lower right', fontsize=FONTSIZE['legend'])
        
        ax.grid(True, alpha=ALPHA['medium'], axis='x', linestyle='--', linewidth=LINEWIDTH['thin'])
    
    ax.tick_params(labelsize=FONTSIZE['tick'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_scr_distribution(data: Dict, output_path: str, show: bool = False) -> None:
    """
    Plot SCR amplitude distribution with histogram and boxplot.
    
    Visualization #9: Combined plot showing:
    - Top panel: Histogram of SCR amplitudes by moment
    - Bottom panel: Boxplot comparing distributions
    
    Args:
        data: Dictionary containing EDA events with key:
            - 'eda': Dict with 'events' sub-dict containing:
                - 'restingstate': DataFrame with 'scr_amplitude'
                - 'therapy': DataFrame with 'scr_amplitude'
        output_path: Path to save the PNG figure
    """
    apply_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE['medium'], 
                                    height_ratios=[2, 1])
    
    # Collect SCR amplitudes by moment
    amplitudes_by_moment = {}
    for moment in ['restingstate', 'therapy']:
        if 'events' not in data['eda'] or moment not in data['eda']['events']:
            amplitudes_by_moment[moment] = []
            continue
        
        df = data['eda']['events'][moment]
        if df.empty or 'scr_amplitude' not in df.columns:
            amplitudes_by_moment[moment] = []
        else:
            amplitudes_by_moment[moment] = df['scr_amplitude'].values
    
    # Top panel: Histogram
    bins = np.linspace(
        0,
        max([max(amps) if len(amps) > 0 else 0.1 
             for amps in amplitudes_by_moment.values()]),
        30
    )
    
    for moment, amplitudes in amplitudes_by_moment.items():
        if len(amplitudes) == 0:
            continue
        
        color = get_moment_color(moment)
        ax1.hist(
            amplitudes,
            bins=bins,
            color=color,
            alpha=ALPHA['medium'],
            label=moment,
            edgecolor='black',
            linewidth=0.5
        )
    
    ax1.set_ylabel('Fréquence', fontsize=FONTSIZE['label'])
    ax1.set_title('Distribution des Amplitudes SCR', 
                 fontsize=FONTSIZE['title'], fontweight='bold', pad=20)
    ax1.legend(loc='upper right', fontsize=FONTSIZE['legend'])
    ax1.grid(True, alpha=ALPHA['medium'], axis='y', linestyle='--', linewidth=LINEWIDTH['thin'])
    ax1.tick_params(labelsize=FONTSIZE['tick'])
    
    # Bottom panel: Boxplot
    box_data = [
        amplitudes_by_moment['restingstate'],
        amplitudes_by_moment['therapy']
    ]
    box_labels = ['Restingstate', 'Therapy']
    
    bp = ax2.boxplot(
        box_data,
        labels=box_labels,
        patch_artist=True,
        notch=True,
        widths=0.6
    )
    
    # Color boxplots
    for patch, moment in zip(bp['boxes'], ['restingstate', 'therapy']):
        patch.set_facecolor(get_moment_color(moment))
        patch.set_alpha(ALPHA['high'])
    
    ax2.set_ylabel('Amplitude (µS)', fontsize=FONTSIZE['label'])
    ax2.set_xlabel('Moment', fontsize=FONTSIZE['label'])
    ax1.grid(True, alpha=ALPHA['medium'], axis='y', linestyle='--', linewidth=LINEWIDTH['thin'])
    ax2.grid(True, alpha=ALPHA['medium'], axis='y', linestyle='--', linewidth=LINEWIDTH['thin'])
    ax2.tick_params(labelsize=FONTSIZE['tick'])
    
    # Add statistics as text
    for i, (moment, amplitudes) in enumerate(amplitudes_by_moment.items()):
        if len(amplitudes) > 0:
            median = np.median(amplitudes)
            q1, q3 = np.percentile(amplitudes, [25, 75])
            ax2.text(
                i + 1, ax2.get_ylim()[1] * 0.95,
                f'n={len(amplitudes)}\nM={median:.2f}µS',
                ha='center', va='top',
                fontsize=FONTSIZE['small'],
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
