#!/usr/bin/env python3
"""
Clean BVP processing outputs for fresh pipeline runs.

This utility script safely removes processed BVP derivatives and logs,
allowing for clean re-runs during testing and development.
"""

import argparse
import shutil
import sys
from pathlib import Path
from typing import List, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config_loader import ConfigLoader
import logging

# Set up basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def confirm_deletion(path: Path, force: bool = False) -> bool:
    """
    Confirm deletion with user unless force flag is set.
    
    Args:
        path: Path to be deleted
        force: Skip confirmation if True
        
    Returns:
        True if deletion should proceed, False otherwise
    """
    if force:
        return True
    
    response = input(f"Delete {path}? [y/N]: ").strip().lower()
    return response in ['y', 'yes']


def clean_bvp_derivatives(
    config: ConfigLoader,
    subject: Optional[str] = None,
    session: Optional[str] = None,
    force: bool = False,
    dry_run: bool = False
) -> int:
    """
    Clean BVP processing derivatives.
    
    Args:
        config: Configuration loader instance
        subject: Specific subject to clean (None = all subjects)
        session: Specific session to clean (None = all sessions)
        force: Skip confirmation prompts
        dry_run: Show what would be deleted without actually deleting
        
    Returns:
        Number of items deleted
    """
    derivatives_path = Path(config.get('paths.derivatives'))
    bvp_derivatives = derivatives_path / 'therasync-bvp'
    
    if not bvp_derivatives.exists():
        logger.info(f"No BVP derivatives found at {bvp_derivatives}")
        return 0
    
    deleted_count = 0
    
    # Clean specific subject/session
    if subject:
        subject_path = bvp_derivatives / subject
        
        if not subject_path.exists():
            logger.warning(f"Subject {subject} not found in derivatives")
            return 0
        
        if session:
            # Clean specific session
            session_path = subject_path / session
            if session_path.exists():
                logger.info(f"{'[DRY RUN] Would delete' if dry_run else 'Deleting'}: {session_path}")
                
                if not dry_run and confirm_deletion(session_path, force):
                    shutil.rmtree(session_path)
                    deleted_count += 1
                    logger.info(f"Deleted session: {session_path}")
        else:
            # Clean all sessions for subject
            logger.info(f"{'[DRY RUN] Would delete' if dry_run else 'Deleting'} all sessions for {subject}")
            
            if not dry_run and confirm_deletion(subject_path, force):
                shutil.rmtree(subject_path)
                deleted_count += 1
                logger.info(f"Deleted subject: {subject_path}")
    else:
        # Clean all derivatives
        logger.info(f"{'[DRY RUN] Would delete' if dry_run else 'Deleting'} all BVP derivatives")
        
        if not dry_run and confirm_deletion(bvp_derivatives, force):
            shutil.rmtree(bvp_derivatives)
            deleted_count += 1
            logger.info(f"Deleted all BVP derivatives: {bvp_derivatives}")
    
    return deleted_count


def clean_logs(
    config: ConfigLoader,
    force: bool = False,
    dry_run: bool = False
) -> int:
    """
    Clean log files.
    
    Args:
        config: Configuration loader instance
        force: Skip confirmation prompts
        dry_run: Show what would be deleted without actually deleting
        
    Returns:
        Number of log files deleted
    """
    log_path = Path(config.get('paths.logs', 'log'))
    
    if not log_path.exists():
        logger.info(f"No log directory found at {log_path}")
        return 0
    
    log_files = list(log_path.glob('*.log*'))
    
    if not log_files:
        logger.info("No log files found")
        return 0
    
    logger.info(f"Found {len(log_files)} log file(s)")
    
    deleted_count = 0
    
    if dry_run:
        logger.info("[DRY RUN] Would delete the following log files:")
        for log_file in log_files:
            logger.info(f"  - {log_file}")
        return len(log_files)
    
    if not force:
        response = input(f"Delete {len(log_files)} log file(s)? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            logger.info("Log deletion cancelled")
            return 0
    
    for log_file in log_files:
        try:
            log_file.unlink()
            deleted_count += 1
            logger.debug(f"Deleted log file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to delete {log_file}: {e}")
    
    logger.info(f"Deleted {deleted_count} log file(s)")
    return deleted_count


def clean_cache(
    force: bool = False,
    dry_run: bool = False
) -> int:
    """
    Clean Python cache files (__pycache__ directories).
    
    Args:
        force: Skip confirmation prompts
        dry_run: Show what would be deleted without actually deleting
        
    Returns:
        Number of cache directories deleted
    """
    project_root = Path(__file__).parent.parent
    cache_dirs = list(project_root.rglob('__pycache__'))
    
    if not cache_dirs:
        logger.info("No cache directories found")
        return 0
    
    logger.info(f"Found {len(cache_dirs)} cache director(y/ies)")
    
    if dry_run:
        logger.info("[DRY RUN] Would delete the following cache directories:")
        for cache_dir in cache_dirs:
            logger.info(f"  - {cache_dir}")
        return len(cache_dirs)
    
    if not force:
        response = input(f"Delete {len(cache_dirs)} cache director(y/ies)? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            logger.info("Cache deletion cancelled")
            return 0
    
    deleted_count = 0
    for cache_dir in cache_dirs:
        try:
            shutil.rmtree(cache_dir)
            deleted_count += 1
            logger.debug(f"Deleted cache directory: {cache_dir}")
        except Exception as e:
            logger.error(f"Failed to delete {cache_dir}: {e}")
    
    logger.info(f"Deleted {deleted_count} cache director(y/ies)")
    return deleted_count


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Clean BVP processing outputs for fresh pipeline runs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to see what would be deleted
  python scripts/clean_outputs.py --dry-run
  
  # Clean all derivatives and logs (with confirmation)
  python scripts/clean_outputs.py --derivatives --logs
  
  # Clean specific subject/session
  python scripts/clean_outputs.py --derivatives --subject sub-f01p01 --session ses-01
  
  # Force clean without confirmation
  python scripts/clean_outputs.py --all --force
  
  # Clean only Python cache files
  python scripts/clean_outputs.py --cache
        """
    )
    
    # What to clean
    parser.add_argument(
        '--derivatives', '-d',
        action='store_true',
        help='Clean BVP processing derivatives'
    )
    
    parser.add_argument(
        '--logs', '-l',
        action='store_true',
        help='Clean log files'
    )
    
    parser.add_argument(
        '--cache', '-c',
        action='store_true',
        help='Clean Python cache files (__pycache__)'
    )
    
    parser.add_argument(
        '--all', '-a',
        action='store_true',
        help='Clean everything (derivatives, logs, and cache)'
    )
    
    # Scope filters
    parser.add_argument(
        '--subject', '-s',
        help='Clean specific subject only (for derivatives)'
    )
    
    parser.add_argument(
        '--session', '-e',
        help='Clean specific session only (requires --subject)'
    )
    
    # Options
    parser.add_argument(
        '--config', '-cfg',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Skip confirmation prompts'
    )
    
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be deleted without actually deleting'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logger.setLevel('DEBUG')
    
    # Validate arguments
    if args.session and not args.subject:
        parser.error("--session requires --subject")
    
    if not any([args.derivatives, args.logs, args.cache, args.all]):
        parser.error("Must specify at least one of: --derivatives, --logs, --cache, or --all")
    
    # Load configuration
    try:
        config = ConfigLoader(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Perform cleaning
    total_deleted = 0
    
    try:
        if args.dry_run:
            logger.info("=== DRY RUN MODE - No files will be deleted ===")
        
        # Clean derivatives
        if args.all or args.derivatives:
            logger.info("Cleaning BVP derivatives...")
            count = clean_bvp_derivatives(
                config,
                subject=args.subject,
                session=args.session,
                force=args.force,
                dry_run=args.dry_run
            )
            total_deleted += count
        
        # Clean logs
        if args.all or args.logs:
            logger.info("Cleaning log files...")
            count = clean_logs(config, force=args.force, dry_run=args.dry_run)
            total_deleted += count
        
        # Clean cache
        if args.all or args.cache:
            logger.info("Cleaning Python cache...")
            count = clean_cache(force=args.force, dry_run=args.dry_run)
            total_deleted += count
        
        # Summary
        if args.dry_run:
            logger.info(f"=== DRY RUN COMPLETE: Would delete {total_deleted} item(s) ===")
        else:
            logger.info(f"=== CLEANING COMPLETE: Deleted {total_deleted} item(s) ===")
        
        return 0
        
    except KeyboardInterrupt:
        logger.warning("Cleaning interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Cleaning failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
