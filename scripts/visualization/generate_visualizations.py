#!/usr/bin/env python
"""
Visualization Generation Script for TherasyncPipeline.

This script generates all 10 visualizations for a subject/session or batch processes
multiple subjects.

Usage:
    # Single subject/session
    python scripts/visualization/generate_visualizations.py --subject f01p01 --session 01
    
    # All available subjects
    python scripts/visualization/generate_visualizations.py --all
    
    # Specific visualizations only
    python scripts/visualization/generate_visualizations.py --subject f01p01 --session 01 --plots 1 2 3

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.visualization.data_loader import VisualizationDataLoader
from src.visualization.plotters.signal_plots import (
    plot_multisignal_dashboard,
    plot_hr_dynamics_timeline,
    plot_events_timeline
)
from src.visualization.plotters.hrv_plots import (
    plot_poincare_hrv,
    plot_autonomic_balance
)
from src.visualization.plotters.eda_plots import (
    plot_eda_arousal_profile,
    plot_scr_distribution
)
from src.visualization.plotters.comparison_plots import (
    plot_correlation_matrix,
    plot_radar_comparison
)
from src.visualization.config import OUTPUT_CONFIG

logger = logging.getLogger(__name__)


# Mapping of plot numbers to (filename, function) pairs
PLOT_FUNCTIONS = {
    1: ('01_dashboard_multisignals.png', plot_multisignal_dashboard),
    2: ('02_poincare_hrv.png', plot_poincare_hrv),
    3: ('03_autonomic_balance.png', plot_autonomic_balance),
    4: ('04_eda_arousal_profile.png', plot_eda_arousal_profile),
    5: ('05_scr_distribution.png', plot_scr_distribution),
    6: ('06_hr_dynamics_timeline.png', plot_hr_dynamics_timeline),
    7: ('07_correlation_matrix.png', plot_correlation_matrix),
    8: ('08_radar_comparison.png', plot_radar_comparison),
    9: ('09_scr_distribution.png', plot_scr_distribution),
    10: ('10_events_timeline.png', plot_events_timeline),
}


def setup_logging(verbose: bool = False):
    """Configure logging."""
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def generate_visualizations_for_subject(
    subject: str,
    session: str,
    plots: Optional[List[int]] = None,
    output_base: Optional[Path] = None
) -> bool:
    """
    Generate all visualizations for a single subject/session.
    
    Args:
        subject: Subject ID (e.g., 'f01p01')
        session: Session ID (e.g., '01')
        plots: List of plot numbers to generate (None = all)
        output_base: Base output directory
    
    Returns:
        True if successful, False otherwise
    """
    logger.info(f"Processing subject {subject}, session {session}")
    
    try:
        # Load data
        loader = VisualizationDataLoader()
        data = loader.load_subject_session(subject, session)
        
        if not data:
            logger.error(f"No data loaded for {subject}/{session}")
            return False
        
        # Setup output directory
        if output_base is None:
            output_base = Path(OUTPUT_CONFIG['base_path'])
        
        subject_id = data['subject_id']
        session_id = data['session_id']
        
        figures_dir = output_base / subject_id / session_id / OUTPUT_CONFIG['figures_subdir']
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Output directory: {figures_dir}")
        
        # Determine which plots to generate
        plots_to_generate = plots if plots else list(PLOT_FUNCTIONS.keys())
        
        # Generate each visualization
        success_count = 0
        for plot_num in plots_to_generate:
            if plot_num not in PLOT_FUNCTIONS:
                logger.warning(f"Plot #{plot_num} not implemented yet, skipping")
                continue
            
            filename, plot_func = PLOT_FUNCTIONS[plot_num]
            output_path = figures_dir / filename
            
            logger.info(f"Generating visualization #{plot_num}: {filename}")
            
            try:
                plot_func(data, output_path=output_path, show=False)
                logger.info(f"  ✓ Saved to {output_path}")
                success_count += 1
            except Exception as e:
                logger.error(f"  ✗ Failed to generate plot #{plot_num}: {str(e)}")
                logger.debug(f"Error details:", exc_info=True)
        
        logger.info(f"Successfully generated {success_count}/{len(plots_to_generate)} visualizations")
        
        return success_count > 0
        
    except Exception as e:
        logger.error(f"Error processing {subject}/{session}: {str(e)}")
        logger.debug("Error details:", exc_info=True)
        return False


def batch_process_all_subjects(
    plots: Optional[List[int]] = None,
    output_base: Optional[Path] = None
) -> dict:
    """
    Process all available subjects/sessions.
    
    Args:
        plots: List of plot numbers to generate (None = all)
        output_base: Base output directory
    
    Returns:
        Dictionary with processing statistics
    """
    logger.info("Starting batch processing of all subjects/sessions")
    
    loader = VisualizationDataLoader()
    subjects_sessions = loader.list_available_subjects()
    
    if not subjects_sessions:
        logger.warning("No subjects/sessions found")
        return {'total': 0, 'success': 0, 'failed': 0}
    
    logger.info(f"Found {len(subjects_sessions)} subject/session combinations")
    
    stats = {'total': len(subjects_sessions), 'success': 0, 'failed': 0}
    
    for i, (subject, session) in enumerate(subjects_sessions, 1):
        logger.info(f"[{i}/{len(subjects_sessions)}] Processing {subject}/{session}")
        
        success = generate_visualizations_for_subject(
            subject, session, plots, output_base
        )
        
        if success:
            stats['success'] += 1
        else:
            stats['failed'] += 1
    
    logger.info(f"Batch processing complete: {stats['success']}/{stats['total']} successful")
    
    return stats


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate visualizations for TherasyncPipeline preprocessed data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all visualizations for one subject/session
  python scripts/visualization/generate_visualizations.py --subject f01p01 --session 01
  
  # Generate specific visualizations only
  python scripts/visualization/generate_visualizations.py --subject f01p01 --session 01 --plots 1 2 3
  
  # Process all available subjects
  python scripts/visualization/generate_visualizations.py --all
  
  # Custom output directory
  python scripts/visualization/generate_visualizations.py --subject f01p01 --session 01 --output custom/path
        """
    )
    
    # Subject/session selection
    parser.add_argument(
        '--subject',
        type=str,
        help='Subject ID (e.g., f01p01)'
    )
    parser.add_argument(
        '--session',
        type=str,
        help='Session ID (e.g., 01)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all available subjects/sessions'
    )
    
    # Visualization options
    parser.add_argument(
        '--plots',
        type=int,
        nargs='+',
        help='Specific plot numbers to generate (default: all implemented plots)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Custom output base directory (default: data/derivatives/visualization)'
    )
    
    # Logging
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Validate arguments
    if not args.all and (not args.subject or not args.session):
        parser.error("Either --all or both --subject and --session must be specified")
    
    # Parse output path
    output_base = Path(args.output) if args.output else None
    
    # Execute
    if args.all:
        stats = batch_process_all_subjects(args.plots, output_base)
        sys.exit(0 if stats['success'] > 0 else 1)
    else:
        success = generate_visualizations_for_subject(
            args.subject, args.session, args.plots, output_base
        )
        sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
