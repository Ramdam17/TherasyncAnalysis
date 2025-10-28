# Therasync Pipeline

A comprehensive pipeline for analyzing physiological data from family therapy sessions.

**Authors**: Lena Adel, Remy Ramadour

## Overview

The Therasync project records families during therapeutic sessions to analyze physiological synchrony and emotional dynamics. This pipeline processes multi-modal data including:

- **Physiological data**: BVP, EDA, Heart Rate (from Empatica E4 devices)
- **Transcript analysis**: Conversation analysis from video recordings
- **Alliance analysis**: Therapist-provided ratings of therapeutic alliance and emotional expression

## Data Structure

The project follows a specialized BIDS format:
- **Subjects**: `sub-fXXpYY` (family XX, participant YY)
- **Sessions**: `ses-01`, `ses-02`, etc.
- **Tasks/Moments**: `task-restingstate`, `task-therapy`

```
data/
├── sub-fXXpYY/           # Family XX, participant YY
│   └── ses-XX/           # Session number
│       └── physio/       # Physiological recordings
│           ├── *_task-restingstate_recording-bvp.tsv
│           ├── *_task-restingstate_recording-bvp.json
│           ├── *_task-therapy_recording-bvp.tsv
│           └── *_task-therapy_recording-bvp.json
└── derivatives/          # Processed data outputs
    └── therasync-physio/ # This pipeline's outputs
```

## Pipeline Components

### Current Phase: Physiological Preprocessing

1. **BVP Pipeline**: Blood volume pulse cleaning and metrics extraction
2. **EDA Pipeline**: Electrodermal activity processing and feature extraction  
3. **HR Pipeline**: Heart rate derivation and HRV analysis

### Future Phases

- **DPPA Analysis**: Dyadic Poincaré Plot Analysis for synchrony
- **Emotion Analysis**: Integration with transcript and alliance data
- **Visualization**: Comprehensive plotting and reporting tools

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

```bash
# Process single subject/session
python scripts/preprocess_physio.py --subject sub-f01p01 --session ses-01

# Batch processing
python scripts/preprocess_physio.py --batch --config config/config.yaml
```

## Development

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