# Therasync Pipeline

A comprehensive pipeline for analyzing physiological data from family therapy sessions.

**Authors**: Lena Adel, Remy Ramadour

## Overview

The Therasync project records families during therapeutic sessions to analyze physiological synchrony and emotional dynamics. This pipeline processes multi-modal data including:

- **Physiological data**: BVP, EDA, Heart Rate (from Empatica E4 devices)
- **Transcript analysis**: Conversation analysis from video recordings
- **Alliance analysis**: Therapist-provided ratings of therapeutic alliance and emotional expression

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
- Heart rate extraction and validation
- Outlier detection and gap interpolation
- 26 comprehensive HR metrics (mean, variability, stability, quality)
- Integrated with BVP pipeline for HRV analysis
- BIDS-compliant output (7 files per subject/session)
- **CLI**: `poetry run python scripts/physio/preprocessing/preprocess_hr.py --subject <id> --session <num>`

#### 4. Visualization Pipeline (Sprint 5)
- 6 core visualizations per subject/session:
  1. **Multi-signal Dashboard**: BVP, EDA, HR overview with moment markers
  2. **Poincaré Plot**: HRV non-linear dynamics (SD1/SD2)
  3. **Autonomic Balance**: LF/HF ratio evolution over time
  4. **EDA Arousal Profile**: Tonic/phasic components with SCR events
  5. **SCR Distribution**: Amplitude histogram and statistics
  6. **HR Dynamics Timeline**: Heart rate evolution with variability bands
- YAML-configured plot styles and parameters
- Automatic data loading from preprocessing derivatives
- **CLI**: `poetry run python scripts/visualization/generate_visualizations.py --subject <id> --session <num>`

#### 5. Batch Processing (Sprint 5)
- **Preprocessing**: Process all subjects/sessions in one command
- **Visualization**: Generate all plots for preprocessed data
- Comprehensive error tracking and logging
- Dry-run mode for validation
- Subject filtering and selective execution
- **CLI Preprocessing**: `poetry run python scripts/batch/run_all_preprocessing.py`
- **CLI Visualization**: `poetry run python scripts/batch/run_all_visualizations.py`

### 🔄 Modular Architecture

The codebase is organized for extensibility:

```
src/physio/preprocessing/   # Preprocessing modules (current focus)
├── bvp_*.py               # BVP pipeline components
├── eda_*.py               # EDA pipeline components  
└── hr_*.py                # HR pipeline components

src/visualization/          # Visualization modules (Sprint 5)
├── data_loader.py         # Load preprocessed data
├── config.py              # Plot styling and configuration
└── plotters/              # Visualization generators
    ├── signal_plots.py    # Multi-signal dashboard, timeline
    ├── hrv_plots.py       # HRV analysis plots
    └── eda_plots.py       # EDA/arousal visualizations

scripts/physio/preprocessing/  # CLI scripts
├── preprocess_bvp.py     # BVP preprocessing
├── preprocess_eda.py     # EDA preprocessing
└── preprocess_hr.py      # HR preprocessing

scripts/visualization/     # Visualization scripts
└── generate_visualizations.py  # Single subject/session visualization

scripts/batch/             # Batch processing scripts
├── run_all_preprocessing.py    # Batch preprocessing
└── run_all_visualizations.py   # Batch visualization

Future modules (planned):
├── src/physio/synchrony/      # Dyadic synchrony analysis
└── src/physio/emotion/        # Emotion recognition
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
- **[API Reference](docs/api_reference.md)**: Complete API documentation for all modules (BVP + EDA)
- **[Troubleshooting Guide](docs/troubleshooting.md)**: Common issues and solutions
- **[Quick Reference](docs/quick_reference.md)**: Command reference card

### Sprint Documentation
- **[Sprint 3 Summary](docs/sprint3_summary.md)**: Complete EDA pipeline implementation details
- **[EDA Testing Results](docs/eda_testing_results.md)**: Real data validation report (5 subjects)
- **[Technical Decisions](docs/technical_decisions_validation.md)**: Formalized design decisions

### Research & Methods
- **[BVP Preprocessing Research](docs/bvp_preprocessing_research.md)**: Method comparison and selection
- **[EDA Preprocessing Research](docs/eda_preprocessing_research.md)**: EDA method survey
- **[BVP Metrics Research](docs/bvp_metrics_research.md)**: 40+ HRV metrics documented
- **[EDA Metrics Research](docs/eda_metrics_research.md)**: 23 EDA metrics documented

### Design Decisions
- **[BVP Decisions](docs/bvp_decisions.md)**: BVP pipeline design choices
- **[EDA Decisions](docs/eda_decisions.md)**: EDA pipeline design choices

## Development

### Project Status

**Current Version**: 0.5.0  
**Last Update**: November 11, 2025

| Sprint | Status | Description | Files Changed |
|--------|--------|-------------|---------------|
| Sprint 1 | ✅ Complete | Project setup, configuration, core utilities | - |
| Sprint 2 | ✅ Complete | BVP preprocessing pipeline | 8 files |
| Sprint 3 | ✅ Complete | EDA preprocessing pipeline | 18 files (+6705 lines) |
| Sprint 4 | ✅ Complete | HR extraction and metrics analysis | 8 files (+2500 lines) |
| **Refactor** | ✅ **Complete** | **Modular architecture restructuring** | **30+ files** |
| **Sprint 5** | ✅ **Complete** | **Visualization + Batch Processing** | **15+ files (+3000 lines)** |

**Latest**: Complete visualization pipeline with 6 core plots per session, plus batch processing scripts for automated preprocessing and visualization of all subjects. Successfully processed 50/51 sessions (98% success rate) and generated 300 visualizations.

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