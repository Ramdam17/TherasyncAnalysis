#!/usr/bin/env python3
"""
Analyze Raw RR Interval Synchrony for Real vs Pseudo Dyads.

Computes 5 coupling metrics (windowed correlation, phase locking value,
spectral coherence, cross-recurrence, transfer entropy) on raw RR interval
time series and runs 3-tier statistical comparison with FDR correction.

Usage:
    uv run python scripts/physio/synchrony/analyze_rr_synchrony.py --task therapy
    uv run python scripts/physio/synchrony/analyze_rr_synchrony.py --task therapy --methods plv_lf plv_hf

Authors: Guillaume Dumas
Date: March 2026
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.core.config_loader import ConfigLoader
from src.core.logger_setup import setup_logging
from src.physio.dppa.dyad_config_loader import DyadConfigLoader
from src.physio.dppa.synchrony_stats import (
    compare_real_vs_pseudo_synchrony,
    generate_synchrony_report,
)
from src.physio.synchrony.rr_loader import RRLoader
from src.physio.synchrony.rr_synchrony_stats import compute_rr_synchrony_for_all_dyads
from src.physio.synchrony.methods import (
    compute_windowed_correlation,
    compute_phase_locking_value,
    compute_spectral_coherence,
    compute_crqa,
    compute_transfer_entropy,
)

logger = logging.getLogger(__name__)


def _build_metrics() -> dict[str, dict]:
    """Build the full metrics dictionary."""
    return {
        "windowed_correlation": {
            "fn": compute_windowed_correlation,
            "value_key": "frac_significant",
            "one_sided": True,
        },
        "plv_lf": {
            "fn": lambda r1, r2: compute_phase_locking_value(r1, r2, band=(0.04, 0.15)),
            "value_key": "plv",
            "one_sided": True,
        },
        "plv_hf": {
            "fn": lambda r1, r2: compute_phase_locking_value(r1, r2, band=(0.15, 0.4)),
            "value_key": "plv",
            "one_sided": True,
        },
        "coherence_lf": {
            "fn": compute_spectral_coherence,
            "value_key": "lf_coherence",
            "one_sided": True,
        },
        "coherence_hf": {
            "fn": compute_spectral_coherence,
            "value_key": "hf_coherence",
            "one_sided": True,
        },
        "crqa_determinism": {
            "fn": compute_crqa,
            "value_key": "determinism",
            "one_sided": True,
        },
        "transfer_entropy": {
            "fn": compute_transfer_entropy,
            "value_key": "te_1_to_2",
            "one_sided": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze raw RR interval synchrony metrics"
    )
    parser.add_argument(
        "--task",
        choices=["therapy", "restingstate"],
        required=True,
        help="Task to analyze",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Subset of methods to run (default: all)",
    )
    parser.add_argument("--config", type=str, default=None, help="Config file path")
    args = parser.parse_args()

    setup_logging(log_dir=Path("log"))
    config = ConfigLoader(args.config)
    rr_loader = RRLoader(args.config)
    dyad_config = DyadConfigLoader()

    all_metrics = _build_metrics()

    if args.methods:
        metrics = {k: v for k, v in all_metrics.items() if k in args.methods}
        if not metrics:
            logger.error(
                f"No valid methods found. Available: {list(all_metrics.keys())}"
            )
            sys.exit(1)
    else:
        metrics = all_metrics

    results_dict: dict[str, dict] = {}

    for name, spec in metrics.items():
        logger.info(f"Computing {name}...")
        df = compute_rr_synchrony_for_all_dyads(
            rr_loader,
            dyad_config,
            args.task,
            metric_fn=spec["fn"],
            value_key=spec["value_key"],
        )
        if len(df) == 0:
            logger.warning(f"No valid dyad data for {name}")
            continue

        tier_results = compare_real_vs_pseudo_synchrony(
            df,
            value_col="metric_value",
            one_sided=spec["one_sided"],
        )
        results_dict[name] = tier_results

    report = generate_synchrony_report(args.task, results_dict)
    print(report)

    # Save report
    derivatives_dir = Path(config.get("paths.derivatives_dir", "data/derivatives"))
    output_dir = derivatives_dir / "dppa" / "inter_session" / "stats"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"rr_synchrony_stats_{args.task}.txt"
    output_path.write_text(report)
    logger.info(f"Report saved to {output_path}")


if __name__ == "__main__":
    main()
