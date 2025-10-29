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

### 🔄 Modular Architecture

The codebase is organized for extensibility:

```
src/physio/preprocessing/   # Preprocessing modules (current focus)
├── bvp_*.py               # BVP pipeline components
├── eda_*.py               # EDA pipeline components  
└── hr_*.py                # HR pipeline components

scripts/physio/preprocessing/  # CLI scripts
├── preprocess_bvp.py     # BVP preprocessing
├── preprocess_eda.py     # EDA preprocessing
└── preprocess_hr.py      # HR preprocessing

Future modules (planned):
├── src/physio/synchrony/      # Dyadic synchrony analysis
├── src/physio/emotion/        # Emotion recognition
└── src/visualization/         # Plotting and reporting
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
# Clean all outputs
poetry run python scripts/utils/clean_outputs.py --derivatives --force

# Process multiple subjects
for subject in f01p01 f02p01; do
  for session in 01 02; do
    poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject $subject --session $session
    poetry run python scripts/physio/preprocessing/preprocess_eda.py --subject $subject --session $session
    poetry run python scripts/physio/preprocessing/preprocess_hr.py --subject $subject --session $session
  done
done
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

**Current Version**: 0.3.0  
**Last Update**: October 28, 2025

| Sprint | Status | Description | Files Changed |
|--------|--------|-------------|---------------|
| Sprint 1 | ✅ Complete | Project setup, configuration, core utilities | - |
| Sprint 2 | ✅ Complete | BVP preprocessing pipeline | 8 files |
| Sprint 3 | ✅ Complete | EDA preprocessing pipeline | 18 files (+6705 lines) |
| Sprint 4 | ✅ Complete | HR extraction and metrics analysis | 8 files (+2500 lines) |
| **Refactor** | ✅ **Complete** | **Modular architecture restructuring** | **30+ files** |

**Latest**: Restructured codebase into modular architecture with `src/physio/preprocessing/` and `scripts/physio/preprocessing/` for better extensibility and future development (synchrony analysis, emotion recognition, visualization modules).

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