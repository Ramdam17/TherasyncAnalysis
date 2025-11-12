# Therasync Pipeline

A comprehensive pipeline for analyzing physiological data from family therapy sessions.

**Authors**: Lena Adel, Remy Ramadour

## Overview

The Therasync project records families during therapeutic sessions to analyze physiological synchrony and emotional dynamics. This pipeline processes multi-modal physiological data from Empatica E4 wearable devices:

- **Blood Volume Pulse (BVP)**: Heart rate variability and autonomic nervous system analysis
- **Electrodermal Activity (EDA)**: Arousal and stress response measurement
- **Heart Rate (HR)**: Cardiovascular dynamics and quality metrics

The pipeline features a fully harmonized, modular architecture with comprehensive BIDS-compliant outputs, automated batch processing, and rich visualizations for clinical and research applications.

## Data Structure

The project follows a BIDS-inspired format with modular preprocessing outputs:
- **Subjects**: `sub-fXXpYY` (family XX, participant YY)
- **Sessions**: `ses-01`, `ses-02`, etc.
- **Tasks/Moments**: `task-restingstate`, `task-therapy`

```
data/
├── raw/                  # Raw BIDS-formatted data
│   └── sub-fXXpYY/      # Family XX, participant YY
│       └── ses-XX/      # Session number
│           └── physio/  # Physiological recordings
│               ├── *_task-restingstate_recording-bvp.tsv
│               ├── *_task-restingstate_recording-eda.tsv
│               └── *_task-therapy_recording-*.tsv
└── derivatives/          # Processed data outputs
    └── preprocessing/    # Preprocessing derivatives (modular structure)
        └── sub-fXXpYY/
            └── ses-XX/
                ├── bvp/  # BVP/HRV preprocessing outputs
                ├── eda/  # EDA preprocessing outputs
                └── hr/   # Heart rate preprocessing outputs
```

## Pipeline Components

### ✅ Implemented Pipelines (Production Ready)

#### 1. BVP Pipeline (Sprint 2)
- Blood volume pulse signal cleaning (NeuroKit2, Elgendi method)
- Peak detection and quality assessment
- 20 HRV metrics extraction (time-domain, frequency-domain, non-linear)
- BIDS-compliant output formatting (9 files per subject/session)
- **CLI**: `poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject <id> --session <num>`

#### 2. EDA Pipeline (Sprint 3)
- Electrodermal activity signal cleaning (NeuroKit2)
- cvxEDA tonic/phasic decomposition
- SCR (Skin Conductance Response) detection and analysis
- 23 comprehensive metrics (SCR peaks, tonic/phasic components, temporal patterns)
- BIDS-compliant output (13 files per subject/session)
- **CLI**: `poetry run python scripts/physio/preprocessing/preprocess_eda.py --subject <id> --session <num>`
- **Validated on**: 5 real subjects from 2 families
- **Performance**: 0.5-1 second per subject/session

#### 3. HR Pipeline (Sprint 4)
- Heart rate extraction from BVP signal
- Automated outlier detection and quality assessment
- Gap interpolation for missing data
- 26 comprehensive HR metrics (mean, variability, stability, quality)
- Per-moment processing (restingstate and therapy)
- BIDS-compliant output (14 files per subject/session: 7 per moment)
- **CLI**: `poetry run python scripts/physio/preprocessing/preprocess_hr.py --subject <id> --session <num>`
- **Validated on**: 49/51 subjects (96% success rate)

#### 4. Visualization Pipeline (Sprint 5)
- 6 comprehensive visualizations per subject/session:
  1. **Multi-signal Dashboard**: BVP, HR, EDA synchronized overview with moment markers
  2. **Poincaré Plot**: HRV non-linear dynamics (SD1/SD2 analysis)
  3. **Autonomic Balance**: LF/HF ratio evolution across moments
  4. **EDA Arousal Profile**: Tonic/phasic components with SCR event markers
  5. **SCR Distribution**: Amplitude histogram and temporal statistics
  6. **HR Dynamics Timeline**: Heart rate evolution with rest/moderate/elevated zones
- YAML-configured plot styles and parameters
- Automatic data loading from BIDS preprocessing derivatives
- Publication-quality output (300 DPI PNG)
- **CLI**: `poetry run python scripts/visualization/generate_visualizations.py --subject <id> --session <num>`
- **Validated on**: 306 visualizations generated (51 sessions × 6 plots, 100% success)

#### 5. Batch Processing & Analysis (Sprint 5-6)
- **Batch Preprocessing**: Process all subjects/sessions with automatic error handling
- **Batch Visualization**: Generate all plots for preprocessed data in parallel
- **Quality Analysis**: Automated quality report generation with 114 signal quality flags
- Comprehensive error tracking and detailed logging
- Dry-run mode for validation before execution
- Subject filtering and selective processing
- Progress tracking with timing statistics
- **CLI Preprocessing**: `poetry run python scripts/batch/run_all_preprocessing.py`
- **CLI Visualization**: `poetry run python scripts/batch/run_all_visualizations.py`
- **CLI Quality Report**: `poetry run python scripts/analysis/generate_quality_report.py`
- **Production Stats**: 49/51 sessions preprocessed (96%), 306/306 visualizations generated (100%)

#### 6. DPPA Pipeline (Dyadic Poincaré Plot Analysis)
- **Poincaré Centroid Computation**: Calculate centroids per participant/session/epoch from RR intervals
- **Inter-Session Analysis**: Quantify synchrony across all sessions (~1,275 dyad pairs)
- **Intra-Family Analysis**: Measure within-family synchrony during same session (81 dyad pairs)
- **ICD Calculation**: Compute Inter-Centroid Distances using Euclidean formula
- **Rectangular CSV Export**: Epochs × dyads format for statistical analysis
- **5 Core Modules**: PoincareCalculator, CentroidLoader, ICDCalculator, DyadConfigLoader, DPPAWriter
- **CLI Poincaré**: `poetry run python scripts/physio/dppa/compute_poincare.py --batch`
- **CLI DPPA**: `poetry run python scripts/physio/dppa/compute_dppa.py --mode both --task all --batch`
- **Validated on**: 51 sessions (606 centroid files), 2,514 ICD pairs (100% success)
- **Methods**: nsplit120 (inter-session), sliding 30s/5s (intra-family)

#### 7. DPPA Visualizations (nsplit120)
- **4-Subplot Dyadic Visualizations**: ICD + SD1/SD2 metrics across epochs
- **ICD Time Series (Top)**: Full-width plot showing therapy vs resting baseline with trendline
- **SD1 (Bottom Left)**: Short-term variability for both subjects (0-600 ms normalized)
- **SD2 (Bottom Center)**: Long-term variability for both subjects (0-600 ms normalized)
- **SD1/SD2 Ratio (Bottom Right)**: Autonomic balance metric (0-3.0 normalized)
- **3 Visualization Modules**: DyadICDLoader, DyadCentroidLoader, DyadPlotter
- **CLI Single**: `poetry run python scripts/physio/dppa/plot_dyad.py --dyad f01p01_ses-01_vs_f01p02_ses-01 --method nsplit120`
- **CLI Batch**: `poetry run python scripts/physio/dppa/plot_dyad.py --batch --mode inter --method nsplit120`
- **Production Results**: 1176 inter-session figures (100% success, 309 MB, ~7 minutes)
- **Output**: `data/derivatives/dppa/figures/nsplit120/` (PNG 12×8 inches, 150 DPI)
- **Testing**: 25 tests (20 unit + 5 CLI integration, 100% passing)

### 🔄 Harmonized Modular Architecture (Phase 2 Complete)

All three BIDS writers (BVP, EDA, HR) now use identical code patterns and structure:

**Harmonization Benefits:**
- ✅ Consistent helper method signatures across all modalities
- ✅ Unified variable naming conventions (`subject_dir`, `base_filename`, `signals_tsv`, `signals_json`)
- ✅ Identical file writing patterns for signals, events, metrics, and summaries
- ✅ Standardized BIDS path construction
- ✅ Easier maintenance and future extensions

**Code Organization:**
```
src/physio/preprocessing/   # Preprocessing modules
├── base_bids_writer.py    # Shared BIDS functionality
├── bvp_*.py               # BVP pipeline (9 output files/session)
├── eda_*.py               # EDA pipeline (13 output files/session)
└── hr_*.py                # HR pipeline (14 output files/session)

src/visualization/          # Visualization modules
├── data_loader.py         # BIDS-compliant data loading
├── config.py              # Plot styling and configuration
└── plotters/              # Visualization generators
    ├── signal_plots.py    # Multi-signal dashboard, HR timeline
    ├── hrv_plots.py       # Poincaré, autonomic balance
    └── eda_plots.py       # EDA arousal, SCR distribution
```
src/physio/preprocessing/   # Preprocessing modules
├── base_bids_writer.py    # Shared BIDS functionality
├── bvp_*.py               # BVP pipeline (9 output files/session)
├── eda_*.py               # EDA pipeline (13 output files/session)
└── hr_*.py                # HR pipeline (14 output files/session)

src/visualization/          # Visualization modules
├── data_loader.py         # BIDS-compliant data loading
├── config.py              # Plot styling and configuration
└── plotters/              # Visualization generators
    ├── signal_plots.py    # Multi-signal dashboard, HR timeline
    ├── hrv_plots.py       # Poincaré, autonomic balance
    └── eda_plots.py       # EDA arousal, SCR distribution

scripts/physio/preprocessing/  # CLI scripts for preprocessing
├── preprocess_bvp.py     # BVP preprocessing
├── preprocess_eda.py     # EDA preprocessing
└── preprocess_hr.py      # HR preprocessing

scripts/visualization/     # CLI scripts for visualization
└── generate_visualizations.py  # Single subject/session plots

scripts/batch/             # Batch processing automation
├── run_all_preprocessing.py    # Batch preprocessing (all sessions)
└── run_all_visualizations.py   # Batch visualization (all sessions)

scripts/analysis/          # Analysis and quality control
└── generate_quality_report.py  # Quality metrics analysis
```

## Quick Start

### Prerequisites

- Python 3.8+
- Poetry for dependency management

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd therasync

# Install dependencies
poetry install

# Activate environment
poetry shell
```

### Configuration

Edit `config/config.yaml` to customize processing parameters:

- **Moments**: Define task periods (currently: restingstate, therapy)
- **Processing methods**: Configure cleaning and feature extraction
- **Output settings**: Customize BIDS derivatives structure

### Usage

#### BVP Preprocessing
```bash
# Process single subject/session (subject ID without 'sub-' prefix)
poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject f01p01 --session 01

# Process with custom config
poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject f01p01 --session 01 --config config/config.yaml

# Check outputs
tree data/derivatives/preprocessing/sub-f01p01/ses-01/bvp/
```

#### EDA Preprocessing
```bash
# Process single subject/session  
poetry run python scripts/physio/preprocessing/preprocess_eda.py --subject f01p01 --session 01

# View metrics
cat data/derivatives/preprocessing/sub-f01p01/ses-01/eda/*_desc-eda-metrics_physio.tsv

# Check SCR events
head data/derivatives/preprocessing/sub-f01p01/ses-01/eda/*_desc-scr_events.tsv
```

#### HR Preprocessing
```bash
# Process single subject/session
poetry run python scripts/physio/preprocessing/preprocess_hr.py --subject f01p01 --session 01

# View metrics
cat data/derivatives/preprocessing/sub-f01p01/ses-01/hr/*_hr-metrics.tsv
```

#### Batch Processing
```bash
# Preprocessing: Process all subjects and sessions
poetry run python scripts/batch/run_all_preprocessing.py

# Options
poetry run python scripts/batch/run_all_preprocessing.py --dry-run        # Preview without execution
poetry run python scripts/batch/run_all_preprocessing.py --skip-existing  # Skip already processed
poetry run python scripts/batch/run_all_preprocessing.py --subjects f01p01 f02p01  # Specific subjects

# Visualization: Generate all plots for preprocessed data
poetry run python scripts/batch/run_all_visualizations.py

# Options
poetry run python scripts/batch/run_all_visualizations.py --dry-run      # Preview
poetry run python scripts/batch/run_all_visualizations.py --plots 1 2 3  # Specific visualizations
poetry run python scripts/batch/run_all_visualizations.py --subjects f01p01  # Specific subjects

# Clean all outputs
poetry run python scripts/utils/clean_outputs.py --derivatives --force

# Typical workflow
poetry run python scripts/batch/run_all_preprocessing.py --skip-existing
poetry run python scripts/batch/run_all_visualizations.py
```

#### Single Subject Processing
```bash
# Process complete pipeline for one subject/session
subject=f01p01
session=01

# 1. Preprocessing
poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject $subject --session $session
poetry run python scripts/physio/preprocessing/preprocess_eda.py --subject $subject --session $session
poetry run python scripts/physio/preprocessing/preprocess_hr.py --subject $subject --session $session

# 2. Visualization
poetry run python scripts/visualization/generate_visualizations.py --subject $subject --session $session

# 3. Check outputs
tree data/derivatives/preprocessing/sub-$subject/ses-$session/
tree data/derivatives/visualization/sub-$subject/ses-$session/figures/
```

## Documentation

Comprehensive documentation is available in the `docs/` directory:

### Essential Guides
- **[Quick Reference](docs/quick_reference.md)**: Command reference card and common workflows
- **[API Reference](docs/api_reference.md)**: Complete API documentation for all modules
- **[Testing Guide](docs/testing_guide.md)**: Testing framework and validation procedures
- **[Troubleshooting](docs/troubleshooting.md)**: Common issues and solutions
- **[Environment Setup](docs/environment_setup.md)**: Installation and configuration
- **[Resources](docs/resources.md)**: External resources and references

## Development

### Project Status

**Current Version**: 1.0.0  
**Last Update**: November 11, 2025  
**Status**: Production Ready

| Phase | Status | Description | Impact |
|-------|--------|-------------|--------|
| Sprint 1 | ✅ Complete | Project setup, configuration, core utilities | Foundation |
| Sprint 2 | ✅ Complete | BVP preprocessing pipeline (HRV analysis) | 9 files/session |
| Sprint 3 | ✅ Complete | EDA preprocessing pipeline (arousal analysis) | 13 files/session |
| Sprint 4 | ✅ Complete | HR extraction and comprehensive metrics | 14 files/session |
| Sprint 5 | ✅ Complete | Visualization pipeline (6 plots/session) | 306 visualizations |
| **Phase 2** | ✅ **Complete** | **Code harmonization & refactoring** | **30+ files unified** |
| **Production** | ✅ **Complete** | **Batch processing & quality analysis** | **96% success rate** |

**Current Capabilities:**
- ✅ 34/34 unit tests passing (100%)
- ✅ 49/51 sessions preprocessed successfully (96%)
- ✅ 306/306 visualizations generated (100%)
- ✅ 114 quality flags tracked across all modalities
- ✅ Fully harmonized BIDS-compliant architecture
- ✅ Automated batch processing with error handling
- ✅ Comprehensive quality reporting

**Latest Achievement**: Phase 2 harmonization complete - all three BIDS writers (BVP, EDA, HR) now use identical code patterns, enabling consistent outputs and easier maintenance. Visualization pipeline successfully integrated with per-moment HR data structure.

### Sprint-based Development

The project follows a sprint-based workflow:
- Each sprint gets a dedicated branch: `sprint-N/description`
- See `TODO.md` for current sprint tasks and progress
- All code must be reviewed before merging to master

### Code Standards

- **Language**: All documentation and code must be in English
- **Testing**: >80% test coverage required
- **Style**: Follow PEP 8, use Black formatter
- **Documentation**: Comprehensive docstrings for all functions

### Testing

```bash
# Run tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

## Contributing

1. Check `TODO.md` for current tasks and priorities
2. Create feature branch from appropriate sprint branch
3. Follow coding standards and write tests
4. Submit PR with clear description

## Research Context

This pipeline supports research into:
- **Physiological synchrony** in family therapy
- **Emotional regulation** during therapeutic interventions
- **Alliance dynamics** between family members and therapists
- **Multi-modal integration** of physiological, verbal, and observational data

## Citation

*Citation information will be added upon publication*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Copyright (c) 2025 Lena Adel, Remy Ramadour**

---

For detailed development progress, see `TODO.md`.
For technical documentation, see `docs/` directory.