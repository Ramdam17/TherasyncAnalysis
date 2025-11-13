#!/usr/bin/env python3
"""
Epoch all physiological signals for batch processing.

This script processes preprocessed physiological signals and adds epoch columns
for time-windowed analysis. Supports batch processing of multiple subjects/sessions.

Usage:
    # Single session
    python scripts/physio/epoching/epoch_all_signals.py --subject g01p01 --session 01
    
    # Batch process all sessions
    python scripts/physio/epoching/epoch_all_signals.py --batch
    
    # Dry run (show what would be processed)
    python scripts/physio/epoching/epoch_all_signals.py --batch --dry-run

Authors: Lena Adel, Remy Ramadour
Date: November 2025
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.core.config_loader import ConfigLoader
from src.physio.epoching.epoch_bids_writer import EpochBIDSWriter

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path, verbose: bool = False):
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"epoch_all_signals_{timestamp}.log"
    
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger.info(f"Logging to: {log_file}")


def find_all_sessions(raw_data_path: Path) -> List[Tuple[str, str]]:
    """
    Find all subject/session pairs in raw data.
    
    Returns:
        List of (subject, session) tuples
    """
    sessions = []
    
    for sub_dir in sorted(raw_data_path.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        
        subject = sub_dir.name.replace("sub-", "")
        
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue
            
            session = ses_dir.name.replace("ses-", "")
            sessions.append((subject, session))
    
    return sessions


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Epoch physiological signals for time-windowed analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single session
  python scripts/physio/epoching/epoch_all_signals.py --subject g01p01 --session 01
  
  # Batch process all sessions
  python scripts/physio/epoching/epoch_all_signals.py --batch
  
  # Dry run to see what would be processed
  python scripts/physio/epoching/epoch_all_signals.py --batch --dry-run
  
  # Process specific subjects only
  python scripts/physio/epoching/epoch_all_signals.py --batch --subjects g01p01 g01p02
        """
    )
    
    parser.add_argument('--subject', type=str, help='Subject ID (e.g., g01p01)')
    parser.add_argument('--session', type=str, help='Session ID (e.g., 01)')
    parser.add_argument('--batch', action='store_true', help='Process all subjects/sessions')
    parser.add_argument('--subjects', nargs='+', help='Process only these subjects')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be processed without executing')
    parser.add_argument('--config', type=str, default='config/config.yaml', help='Path to config file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Load config
    config = ConfigLoader(args.config)
    paths = config.get("paths", {})
    
    # Setup logging
    log_dir = Path(paths.get("logs", "log"))
    setup_logging(log_dir, args.verbose)
    
    logger.info("="*80)
    logger.info("EPOCH ALL SIGNALS")
    logger.info("="*80)
    logger.info(f"Dry run: {args.dry_run}")
    logger.info(f"Config: {args.config}")
    
    # Initialize writer
    writer = EpochBIDSWriter(args.config)
    
    # Determine sessions to process
    if args.batch:
        raw_data_path = Path(paths.get("rawdata", "data/raw"))
        sessions = find_all_sessions(raw_data_path)
        
        # Filter by subjects if specified
        if args.subjects:
            sessions = [(s, ses) for s, ses in sessions if s in args.subjects]
        
        logger.info(f"Found {len(sessions)} sessions to process")
    elif args.subject and args.session:
        sessions = [(args.subject, args.session)]
        logger.info(f"Processing single session: {args.subject}/ses-{args.session}")
    else:
        parser.error("Must specify either --batch or both --subject and --session")
        return 1
    
    # Process sessions
    total_stats = {'processed': 0, 'skipped': 0, 'failed': 0}
    successful_sessions = 0
    failed_sessions = []
    
    for i, (subject, session) in enumerate(sessions, 1):
        logger.info(f"\n[{i}/{len(sessions)}] Processing sub-{subject}/ses-{session}")
        
        if args.dry_run:
            logger.info("  DRY RUN - would process this session")
            continue
        
        try:
            stats = writer.process_session(subject, session)
            
            # Update totals
            for key in total_stats:
                total_stats[key] += stats[key]
            
            if stats['processed'] > 0 or stats['skipped'] > 0:
                successful_sessions += 1
            else:
                failed_sessions.append(f"sub-{subject}/ses-{session}")
        
        except Exception as e:
            logger.error(f"Failed to process sub-{subject}/ses-{session}: {e}")
            failed_sessions.append(f"sub-{subject}/ses-{session}")
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"Sessions processed:  {successful_sessions}/{len(sessions)}")
    logger.info(f"Files processed:     {total_stats['processed']}")
    logger.info(f"Files skipped:       {total_stats['skipped']}")
    logger.info(f"Files failed:        {total_stats['failed']}")
    
    if failed_sessions:
        logger.warning(f"\nFailed sessions ({len(failed_sessions)}):")
        for session in failed_sessions:
            logger.warning(f"  - {session}")
    
    logger.info("="*80)
    
    return 0 if not failed_sessions else 1


if __name__ == "__main__":
    sys.exit(main())
