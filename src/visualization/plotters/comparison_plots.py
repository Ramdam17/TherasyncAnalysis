"""
Cross-modal comparison visualization plots.

Implements visualizations #7, #8:
- Correlation matrix (cross-modal heatmap)
- Radar comparison (restingstate vs therapy)
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
import pandas as pd
import seaborn as sns

from ..config import (
    COLORS, FIGSIZE, FONTSIZE, LINEWIDTH, ALPHA,
    apply_plot_style, get_moment_color, METRIC_LABELS
)


def plot_correlation_matrix(data: Dict, output_path: str, show: bool = False) -> None:
    """
    Plot cross-modal correlation matrix heatmap.
    
    Visualization #7: Correlation heatmap showing relationships between:
    - BVP/HRV metrics
    - EDA metrics
    - HR metrics
    Across both restingstate and therapy moments.
    
    Args:
        data: Dictionary containing metrics from all modalities:
            - 'bvp': Dict with 'metrics' DataFrame
            - 'eda': Dict with 'metrics' DataFrame
            - 'hr': Dict with 'metrics' DataFrame (if available)
        output_path: Path to save the PNG figure
    """
    apply_plot_style()
    
    # Collect all metrics into a single DataFrame
    metrics_list = []
    
    for moment in ['restingstate', 'therapy']:
        moment_metrics = {'moment': moment}
        
        # BVP metrics
        if 'bvp' in data and 'metrics' in data['bvp'] and data['bvp']['metrics'] is not None:
            bvp_df = data['bvp']['metrics']
            if not bvp_df.empty:
                bvp_moment = bvp_df[bvp_df['moment'] == moment]
                if not bvp_moment.empty:
                    for col in bvp_moment.columns:
                        if col != 'moment' and pd.api.types.is_numeric_dtype(bvp_moment[col]):
                            moment_metrics[f'bvp_{col}'] = bvp_moment[col].iloc[0]
        
        # EDA metrics
        if 'eda' in data and 'metrics' in data['eda'] and data['eda']['metrics'] is not None:
            eda_df = data['eda']['metrics']
            if not eda_df.empty:
                eda_moment = eda_df[eda_df['moment'] == moment]
                if not eda_moment.empty:
                    for col in eda_moment.columns:
                        if col != 'moment' and pd.api.types.is_numeric_dtype(eda_moment[col]):
                            moment_metrics[f'eda_{col}'] = eda_moment[col].iloc[0]
        
        # HR metrics (if available)
        if 'hr' in data and 'metrics' in data['hr'] and data['hr']['metrics'] is not None:
            hr_df = data['hr']['metrics']
            if not hr_df.empty:
                hr_moment = hr_df[hr_df['moment'] == moment]
                if not hr_moment.empty:
                    for col in hr_moment.columns:
                        if col != 'moment' and pd.api.types.is_numeric_dtype(hr_moment[col]):
                            moment_metrics[f'hr_{col}'] = hr_moment[col].iloc[0]
        
        if len(moment_metrics) > 1:  # More than just 'moment'
            metrics_list.append(moment_metrics)
    
    if len(metrics_list) < 2:
        # Not enough data for correlation
        fig, ax = plt.subplots(figsize=FIGSIZE['large'])
        ax.text(0.5, 0.5, 'Données insuffisantes pour calculer les corrélations', 
               ha='center', va='center', fontsize=FONTSIZE['title'])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Create DataFrame and compute correlation
    df = pd.DataFrame(metrics_list)
    df = df.drop('moment', axis=1, errors='ignore')
    
    # Select subset of most relevant metrics (avoid overcrowding)
    key_metrics = []
    for col in df.columns:
        # Include key HRV, EDA, and HR metrics
        if any(key in col.lower() for key in [
            'mean_hr', 'sdnn', 'rmssd', 'lf_hf',
            'scr_count', 'mean_scr_amplitude', 'eda_mean',
            'hr_mean', 'hr_std'
        ]):
            key_metrics.append(col)
    
    if len(key_metrics) > 0:
        df = df[key_metrics]
    
    # Compute correlation matrix
    corr_matrix = df.corr()
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=FIGSIZE['large'])
    
    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    # Generate heatmap
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap='RdBu_r',
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={
            'shrink': 0.8,
            'label': 'Corrélation de Pearson',
            'orientation': 'vertical'
        },
        annot=True,
        fmt='.2f',
        annot_kws={'fontsize': FONTSIZE['small']},
        ax=ax
    )
    
    # Formatting
    ax.set_title('Matrice de Corrélations Cross-Modales', 
                fontsize=FONTSIZE['title'], fontweight='bold', pad=20)
    
    # Rotate labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=FONTSIZE['tick'])
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=FONTSIZE['tick'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_radar_comparison(data: Dict, output_path: str, show: bool = False) -> None:
    """
    Plot radar chart comparing restingstate vs therapy moments.
    
    Visualization #8: Radar/spider plot showing:
    - Normalized metrics across multiple dimensions
    - Direct visual comparison of physiological state between moments
    - Key metrics from BVP, EDA, and HR modalities
    
    Args:
        data: Dictionary containing metrics from all modalities:
            - 'bvp': Dict with 'metrics' DataFrame
            - 'eda': Dict with 'metrics' DataFrame
            - 'hr': Dict with 'metrics' DataFrame (if available)
        output_path: Path to save the PNG figure
    """
    apply_plot_style()
    
    # Select key metrics for radar plot
    metrics_to_plot = [
        ('bvp', 'mean_hr', 'HR Moyen'),
        ('bvp', 'sdnn', 'SDNN'),
        ('bvp', 'rmssd', 'RMSSD'),
        ('bvp', 'lf_hf_ratio', 'LF/HF Ratio'),
        ('eda', 'scr_count', 'Nb SCRs'),
        ('eda', 'mean_scr_amplitude', 'Amp SCR Moy'),
        ('eda', 'eda_mean', 'EDA Moy'),
        ('hr', 'hr_mean', 'HR (direct)'),
    ]
    
    # Collect values for each moment
    restingstate_values = []
    therapy_values = []
    labels = []
    
    for modality, metric, label in metrics_to_plot:
        if modality not in data or 'metrics' not in data[modality] or data[modality]['metrics'] is None:
            continue
        
        df = data[modality]['metrics']
        if df.empty or metric not in df.columns:
            continue
        
        rest_row = df[df['moment'] == 'restingstate']
        therapy_row = df[df['moment'] == 'therapy']
        
        if not rest_row.empty and not therapy_row.empty:
            rest_val = rest_row[metric].iloc[0]
            therapy_val = therapy_row[metric].iloc[0]
            
            # Skip if both are zero or NaN
            if pd.isna(rest_val) or pd.isna(therapy_val):
                continue
            if rest_val == 0 and therapy_val == 0:
                continue
            
            restingstate_values.append(rest_val)
            therapy_values.append(therapy_val)
            labels.append(label)
    
    if len(labels) < 3:
        # Not enough metrics for radar plot
        fig, ax = plt.subplots(figsize=FIGSIZE['medium'])
        ax.text(0.5, 0.5, 'Données insuffisantes pour le graphique radar\n(minimum 3 métriques requises)', 
               ha='center', va='center', fontsize=FONTSIZE['title'])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    # Normalize values to 0-1 range
    all_values = np.array([restingstate_values, therapy_values])
    min_vals = np.min(all_values, axis=0)
    max_vals = np.max(all_values, axis=0)
    range_vals = max_vals - min_vals
    range_vals[range_vals == 0] = 1  # Avoid division by zero
    
    rest_normalized = (np.array(restingstate_values) - min_vals) / range_vals
    therapy_normalized = (np.array(therapy_values) - min_vals) / range_vals
    
    # Setup radar chart
    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # Complete the circle
    rest_normalized = np.concatenate((rest_normalized, [rest_normalized[0]]))
    therapy_normalized = np.concatenate((therapy_normalized, [therapy_normalized[0]]))
    angles += angles[:1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=FIGSIZE['medium'], subplot_kw=dict(projection='polar'))
    
    # Plot data
    ax.plot(angles, rest_normalized, 'o-', linewidth=LINEWIDTH['signal'], 
           color=get_moment_color('restingstate'), label='Restingstate',
           markersize=8, alpha=ALPHA['high'])
    ax.fill(angles, rest_normalized, 
           color=get_moment_color('restingstate'), alpha=ALPHA['low'])
    
    ax.plot(angles, therapy_normalized, 'o-', linewidth=LINEWIDTH['signal'], 
           color=get_moment_color('therapy'), label='Therapy',
           markersize=8, alpha=ALPHA['high'])
    ax.fill(angles, therapy_normalized, 
           color=get_moment_color('therapy'), alpha=ALPHA['low'])
    
    # Formatting
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=FONTSIZE['tick'])
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=FONTSIZE['small'])
    ax.grid(True, alpha=ALPHA['medium'], linestyle='--', linewidth=LINEWIDTH['thin'])
    
    ax.set_title('Comparaison Radar: Restingstate vs Therapy', 
                fontsize=FONTSIZE['title'], fontweight='bold', pad=30)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=FONTSIZE['legend'])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
