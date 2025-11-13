# Copilot Instructions for TherasyncPipeline Project

**Project Version:** v0.3.0 (Modular Architecture)  
**Last Updated:** October 28, 2025  
**Status:** Production-ready preprocessing pipelines (BVP, EDA, HR)

## General Guidelines

### Language and Documentation
- **ALL code documentation MUST be in English** (docstrings, comments, variable names, function names)
- **ALL commit messages MUST be in English**
- Use clear, descriptive naming conventions following PEP 8
- Every function/class must have a comprehensive docstring

### Code Quality Standards
- **Maximum file length: 200 lines** (excluding docstrings and comments)
- If a file exceeds 200 lines, split it into multiple modules
- Follow SOLID principles and keep functions focused on single responsibilities
- Use type hints for all function signatures
- All modules must follow the standard initialization pattern (see Module Initialization Pattern above)

### Testing Requirements
- **ALL functions MUST have corresponding unit tests**
- Tests must be located in the `tests/` directory
- Use pytest framework
- Aim for >80% code coverage
- Test edge cases and error conditions

### File Operations - CRITICAL RULES
⚠️ **NEVER create, modify, or delete ANY file without explicit user approval**
- Always ask before creating new files
- Always ask before modifying existing files
- Always ask before deleting files
- Show the user what changes you intend to make first

### Logging Standards
- Use Python's `logging` module (not print statements)
- Log files must be stored in `log/` directory
- Use appropriate log levels:
  - DEBUG: Detailed diagnostic information
  - INFO: General information about execution
  - WARNING: Warnings about potential issues
  - ERROR: Error messages for failures
  - CRITICAL: Critical failures requiring immediate attention
- Include timestamps and module names in logs
- Rotate logs to prevent excessive file sizes

## Project Structure

```
TherasyncPipeline/
├── src/
│   ├── core/                          # Core utilities (ConfigLoader, BIDS utils, logger)
│   └── physio/
│       └── preprocessing/             # Preprocessing modules (CURRENT: BVP, EDA, HR)
│           ├── bvp_loader.py
│           ├── bvp_cleaner.py
│           ├── bvp_metrics.py
│           ├── bvp_bids_writer.py
│           ├── eda_loader.py
│           ├── eda_cleaner.py
│           ├── eda_metrics.py
│           ├── eda_bids_writer.py
│           ├── hr_loader.py
│           ├── hr_cleaner.py
│           ├── hr_metrics.py
│           └── hr_bids_writer.py
├── scripts/
│   ├── physio/
│   │   └── preprocessing/             # CLI scripts for preprocessing
│   │       ├── preprocess_bvp.py
│   │       ├── preprocess_eda.py
│   │       └── preprocess_hr.py
│   └── utils/                         # Utility scripts
│       └── clean_outputs.py
├── tests/                             # Unit and integration tests (34/34 passing)
│   ├── test_bvp_pipeline.py
│   ├── test_eda_pipeline.py
│   ├── test_hr_pipeline.py
│   └── test_config/                   # Test configuration files
├── docs/                              # Documentation
│   ├── bvp_preprocessing_research.md
│   ├── bvp_decisions.md
│   ├── bvp_metrics_research.md
│   ├── eda_preprocessing_research.md
│   ├── eda_decisions.md
│   ├── eda_metrics_research.md
│   ├── hr_metrics_research.md
│   ├── api_reference.md
│   ├── testing_guide.md
│   ├── troubleshooting.md
│   ├── technical_decisions_validation.md
│   ├── resources.md
│   └── environment_setup.md
├── config/                            # Configuration files
│   └── config.yaml                    # Main configuration
├── notebooks/                         # Jupyter notebooks (analysis/visualization)
├── data/                              # Input and output data (gitignored)
│   ├── sourcedata/                    # Raw BIDS data
│   │   └── sub-{subject}/
│   │       └── ses-{session}/
│   │           ├── physio/            # Physiological recordings
│   │           └── ...
│   └── derivatives/                   # Processed outputs
│       └── preprocessing/
│           └── sub-{subject}/
│               └── ses-{session}/
│                   ├── bvp/           # BVP outputs (9 files)
│                   ├── eda/           # EDA outputs (13 files)
│                   └── hr/            # HR outputs (7 files)
├── log/                               # Log files (gitignored)
├── pyproject.toml                     # Poetry dependencies
├── README.md                          # Main documentation
├── QUICKREF.md                        # Quick reference guide
└── TODO.md                            # Project TODO list
```

## Modular Architecture Principles

### Module Organization
- **Preprocessing modules** are in `src/physio/preprocessing/`
- **Future modules** will be organized by functionality:
  - `src/physio/synchrony/` - Dyadic analysis modules
  - `src/physio/emotion/` - Emotion recognition modules
  - `src/visualization/` - Visualization and reporting
- Each modality (BVP, EDA, HR) has 4 core modules:
  - `*_loader.py` - Load raw data
  - `*_cleaner.py` - Clean and process signals
  - `*_metrics.py` - Extract metrics
  - `*_bids_writer.py` - Write BIDS-compliant output

### Module Initialization Pattern
All preprocessing modules follow this consistent pattern:
```python
from pathlib import Path
from typing import Optional, Union
from src.core.config_loader import ConfigLoader

class ModuleName:
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize with optional config path."""
        self.config = ConfigLoader(config_path)
        # ... rest of initialization
```

**CRITICAL:** Never deviate from this pattern. All modules must:
- Accept `config_path` parameter (not ConfigLoader instance)
- Create ConfigLoader internally
- Use Optional[Union[str, Path]] type hint

### CLI Script Pattern
All preprocessing scripts follow this pattern:
```python
#!/usr/bin/env python3
"""Script description."""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.physio.preprocessing.modality_loader import ModalityLoader
# ... other imports

def main():
    # Parse arguments
    # Load config
    # Process data
    # Log results

if __name__ == "__main__":
    main()
```

**Usage:**
```bash
poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject g01p01 --session 01
```

Note: No PYTHONPATH manipulation needed; scripts handle path setup internally.

## Workflow
1. Always start by understanding the requirements
2. Plan the architecture before coding
3. Write tests before or alongside implementation (TDD encouraged)
4. Document as you code
5. Ask for approval before making file changes
6. Commit small, logical changes with descriptive messages

## Development Standards

### Test Requirements
- **ALL functions MUST have corresponding unit tests**
- Tests must be located in the `tests/` directory
- Use pytest framework
- Current status: **34/34 tests passing (100%)**
- Test each module in isolation and as part of full pipeline
- Test edge cases and error conditions
- Maintain >80% code coverage

### Current Testing Status
- ✅ BVP Pipeline: All tests passing
- ✅ EDA Pipeline: All tests passing  
- ✅ HR Pipeline: All tests passing
- ✅ End-to-end validation: All pipelines verified on real data

## Python Style Guide
- Follow PEP 8
- Use Black formatter (line length: 88)
- Use isort for import sorting
- Use pylint/flake8 for linting
- Maximum function complexity: 10 (cyclomatic complexity)

## Error Handling
- Use specific exception types
- Always log errors with full context
- Provide helpful error messages to users
- Fail gracefully with meaningful feedback

## Performance Guidelines
- Profile code when processing large datasets
- Use generators for memory efficiency
- Parallelize when beneficial (multiprocessing/threading)
- Cache expensive computations when appropriate

## Security
- Never commit sensitive data
- Sanitize user inputs
- Use secure random for any cryptographic needs
- Keep dependencies updated

## Version Control
- Commit frequently with meaningful messages
- Use conventional commit format: `type(scope): description`
  - Types: feat, fix, docs, style, refactor, test, chore
- Keep commits atomic and focused
- Branch for features/experiments

### Branching Strategy
- **Feature branches**: `feature/description` for new features
- **Refactoring branches**: `refactor/description` for architecture changes
- **Bugfix branches**: `fix/description` for bug fixes
- Create branch from `master` before starting work
- Make all commits on the feature branch
- Request user approval before merging to master
- Use `git mv` to preserve file history when reorganizing
- Never push directly to master without approval

### Recent Major Refactoring
- **Branch:** `refactor/restructure-for-modularity` (October 2025)
- **Scope:** Complete modular architecture restructuring
- **Status:** Complete, ready to merge
- **Changes:** Reorganized all preprocessing modules into modular structure
- **Tests:** 34/34 passing (100%)
- **Validation:** All three pipelines tested end-to-end

## BIDS Compliance

### Output Structure
All processed data must follow BIDS derivative format:
```
data/derivatives/preprocessing/
└── sub-{subject}/
    └── ses-{session}/
        ├── bvp/
        │   ├── sub-{subject}_ses-{session}_task-restingstate_desc-bvp-processed_physio.tsv.gz
        │   ├── sub-{subject}_ses-{session}_task-restingstate_desc-bvp-metrics_physio.tsv
        │   └── ... (9 files total)
        ├── eda/
        │   ├── sub-{subject}_ses-{session}_task-restingstate_desc-eda-processed_physio.tsv.gz
        │   ├── sub-{subject}_ses-{session}_task-restingstate_desc-scr_events.tsv
        │   └── ... (13 files total)
        └── hr/
            ├── sub-{subject}_ses-{session}_task-restingstate_desc-hr-processed_physio.tsv.gz
            └── ... (7 files total)
```

### File Naming Convention
- Format: `sub-{subject}_ses-{session}_task-{moment}_desc-{description}_{suffix}.{ext}`
- Moments: `restingstate`, `therapy`
- Suffixes: `physio`, `events`, `metrics`
- Extensions: `.tsv`, `.tsv.gz`, `.json`

## Project Status (October 2025)

### Completed Phases
1. ✅ **Project Setup** - Configuration, utilities, documentation structure
2. ✅ **BVP Pipeline** - Blood volume pulse preprocessing (18 HRV metrics)
3. ✅ **EDA Pipeline** - Electrodermal activity preprocessing (23 metrics)
4. ✅ **HR Pipeline** - Heart rate preprocessing (basic HR metrics)
5. ✅ **Modular Refactoring** - Complete architecture restructuring

### Production-Ready Components
- All three preprocessing pipelines (BVP, EDA, HR)
- BIDS-compliant output generation
- Comprehensive unit tests (34/34 passing)
- Complete documentation (README, QUICKREF, research docs)
- CLI scripts for all modalities

### Next Development Opportunities
- Synchrony analysis module (`src/physio/synchrony/`)
- Emotion recognition module (`src/physio/emotion/`)
- Visualization module (`src/visualization/`)
- Advanced preprocessing features (artifact detection, adaptive filtering)
- Batch processing automation

Remember: Quality over speed. Ask questions when requirements are unclear.