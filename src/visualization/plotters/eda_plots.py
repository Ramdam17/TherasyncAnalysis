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
    Plot EDA baseline arousal comparison between moments.
    
    Visualization #4: Quantitative comparison showing:
    - Tonic EDA levels: mean, min, max (baseline arousal)
    - Phasic variability: standard deviation (reactivity)
    
    Args:
        data: Dictionary containing EDA signals with structure:
            - 'eda': Dict with 'signals' containing moment DataFrames
                     Each should have: 'EDA_Tonic', 'EDA_Phasic'
        output_path: Path to save the PNG figure
    """
    apply_plot_style()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=FIGSIZE['wide'])
    
    eda_data = data.get('eda', {})
    if not eda_data or 'signals' not in eda_data:
        ax1.text(0.5, 0.5, 'No EDA data available', ha='center', va='center',
                transform=ax1.transAxes, fontsize=FONTSIZE['label'])
        return
    
    moments = ['restingstate', 'therapy']
    available_moments = [m for m in moments if m in eda_data.get('signals', {})]
    
    if not available_moments:
        ax1.text(0.5, 0.5, 'No EDA signals available', ha='center', va='center',
                transform=ax1.transAxes, fontsize=FONTSIZE['label'])
        return
    
    # Calculate statistics for each moment
    stats = {}
    for moment in available_moments:
        df = eda_data['signals'][moment]
        if df.empty or 'EDA_Tonic' not in df.columns or 'EDA_Phasic' not in df.columns:
            continue
        
        # Clip tonic values to 0 (physiological minimum)
        # Note: Negative tonic values indicate preprocessing artifacts
        tonic_clipped = df['EDA_Tonic'].clip(lower=0)
        
        stats[moment] = {
            'tonic_mean': tonic_clipped.mean(),
            'tonic_min': tonic_clipped.min(),
            'tonic_max': tonic_clipped.max(),
            'phasic_std': df['EDA_Phasic'].std()
        }
    
    if not stats:
        ax1.text(0.5, 0.5, 'Insufficient EDA data', ha='center', va='center',
                transform=ax1.transAxes, fontsize=FONTSIZE['label'])
        return
    
    # Panel 1: Tonic EDA levels (mean, min, max)
    x = np.arange(len(stats))
    width = 0.25
    
    for i, moment in enumerate(stats.keys()):
        moment_color = get_moment_color(moment)
        s = stats[moment]
        
        # Mean (solid bar)
        ax1.bar(i - width, s['tonic_mean'], width, 
               color=moment_color, alpha=0.9, 
               edgecolor='white', linewidth=2,
               label='Mean' if i == 0 else '')
        
        # Min (lighter bar)
        ax1.bar(i, s['tonic_min'], width,
               color=moment_color, alpha=0.5,
               edgecolor='white', linewidth=2,
               label='Min' if i == 0 else '')
        
        # Max (darker bar with hatching)
        ax1.bar(i + width, s['tonic_max'], width,
               color=moment_color, alpha=0.7, hatch='///',
               edgecolor='white', linewidth=2,
               label='Max' if i == 0 else '')
        
        # Add value labels
        ax1.text(i - width, s['tonic_mean'] + 0.05, f"{s['tonic_mean']:.2f}",
                ha='center', va='bottom', fontsize=FONTSIZE['annotation'],
                fontweight='bold', color=moment_color)
        ax1.text(i, s['tonic_min'] + 0.05, f"{s['tonic_min']:.2f}",
                ha='center', va='bottom', fontsize=FONTSIZE['annotation'],
                fontweight='bold', color=moment_color)
        ax1.text(i + width, s['tonic_max'] + 0.05, f"{s['tonic_max']:.2f}",
                ha='center', va='bottom', fontsize=FONTSIZE['annotation'],
                fontweight='bold', color=moment_color)
    
    ax1.set_xlabel('Moment', fontsize=FONTSIZE['label'], fontweight='bold')
    ax1.set_ylabel('EDA Tonic Level (µS)', fontsize=FONTSIZE['label'], fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([m.capitalize() for m in stats.keys()], 
                        fontsize=FONTSIZE['tick'], fontweight='bold')
    ax1.legend(loc='upper left', fontsize=FONTSIZE['legend'], framealpha=0.95)
    ax1.grid(True, alpha=ALPHA['fill'], axis='y', linestyle='--')
    ax1.set_title('Baseline Arousal Levels\n(Tonic EDA)', 
                 fontsize=FONTSIZE['subtitle'], fontweight='bold')
    
    # Panel 2: Phasic variability (reactivity)
    for i, moment in enumerate(stats.keys()):
        moment_color = get_moment_color(moment)
        s = stats[moment]
        
        # Phasic std as bar
        ax2.bar(i, s['phasic_std'], 0.6,
               color=moment_color, alpha=0.8,
               edgecolor='white', linewidth=2)
        
        # Add value label
        ax2.text(i, s['phasic_std'] + max([stats[m]['phasic_std'] for m in stats]) * 0.02, 
                f"{s['phasic_std']:.3f}",
                ha='center', va='bottom', fontsize=FONTSIZE['annotation'],
                fontweight='bold', color=moment_color)
    
    ax2.set_xlabel('Moment', fontsize=FONTSIZE['label'], fontweight='bold')
    ax2.set_ylabel('EDA Phasic Std Dev (µS)', fontsize=FONTSIZE['label'], fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([m.capitalize() for m in stats.keys()], 
                        fontsize=FONTSIZE['tick'], fontweight='bold')
    ax2.grid(True, alpha=ALPHA['fill'], axis='y', linestyle='--')
    ax2.set_title('Emotional Reactivity\n(Phasic Variability)', 
                 fontsize=FONTSIZE['subtitle'], fontweight='bold')
    
    # Overall title
    fig.suptitle(
        f'EDA Baseline Arousal Comparison\n'
        f'Subject {data.get("subject", "Unknown")}, Session {data.get("session", "Unknown")}',
        fontsize=FONTSIZE['title'], fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    
    if output_path:
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close(fig)


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
