# Environment Setup Guide

**Last Updated:** November 11, 2025  
**Project Version:** v1.0.0 (Production Ready)  

This guide helps you set up the development environment for the Therasync Pipeline project.

## Prerequisites

- **Python 3.8+** (tested with Python 3.10, 3.11, 3.12)
- **Poetry** for dependency management (required)
- **Git** for version control
- **Linux/macOS/Windows** (primarily tested on Linux)
- **Disk space**: ~2.5 GB for full dataset processing
- **Memory**: <2 GB RAM for typical processing

## Installation Methods

### Method 1: Poetry (Recommended)

Poetry manages dependencies and virtual environments automatically:

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Clone the repository
git clone https://github.com/Ramdam17/TherasyncAnalysis.git
cd TherasyncPipeline

# Install all dependencies
poetry install

# Activate the virtual environment
poetry shell

# Verify installation
poetry run python -c "import neurokit2; import pandas; import numpy; print('All packages installed successfully!')"
```

### Method 2: Standard pip installation (Alternative)

If you prefer not to use Poetry:

```bash
# Clone the repository
git clone https://github.com/Ramdam17/TherasyncAnalysis.git
cd TherasyncPipeline

# Create a virtual environment
python -m venv therasync-env

# Activate virtual environment
# On Linux/Mac:
source therasync-env/bin/activate
# On Windows:
therasync-env\Scripts\activate

# Install dependencies from pyproject.toml
pip install -e .

# Verify installation
python -c "import neurokit2; import pandas; import numpy; print('All packages installed successfully!')"
```

## Development Setup

### Code Quality Tools

The project includes several code quality tools. Set them up:

```bash
# Format code with Black
black src/ scripts/ tests/

# Sort imports with isort
isort src/ scripts/ tests/

# Lint with flake8
flake8 src/ scripts/ tests/

# Comprehensive linting with pylint
pylint src/
```

### Pre-commit Hooks (Optional but Recommended)

```bash
# Install pre-commit
pip install pre-commit

# Set up git hooks
pre-commit install
```

### Testing Environment

```bash
# Run all tests (34 tests, should all pass)
poetry run pytest tests/ -v

# Run tests with coverage
poetry run pytest --cov=src tests/

# Run specific test file
poetry run pytest tests/test_bvp_pipeline.py -v

# Verify all pipelines work
poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject g01p01 --session 01
poetry run python scripts/physio/preprocessing/preprocess_eda.py --subject g01p01 --session 01
poetry run python scripts/physio/preprocessing/preprocess_hr.py --subject g01p01 --session 01
poetry run python scripts/visualization/generate_visualizations.py --subject g01p01 --session 01
```
poetry run pytest --cov=src tests/

# Run specific test file
poetry run pytest tests/test_bvp_pipeline.py
poetry run pytest tests/test_eda_pipeline.py
poetry run pytest tests/test_hr_pipeline.py

# Run tests in verbose mode
poetry run pytest -v tests/

# Current status: 34/34 tests passing (100%)
```

## IDE Configuration

### VS Code (Recommended)

Install these extensions:
- **Python** (Microsoft)
- **Pylance** (Microsoft) 
- **Black Formatter** (Microsoft)
- **isort** (Microsoft)

Add to your VS Code settings.json:

```json
{
    "python.formatting.provider": "black",
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.pylintEnabled": true,
    "editor.formatOnSave": true,
    "python.sortImports.args": ["--profile", "black"]
}
```

### PyCharm

1. Set interpreter to your virtual environment
2. Enable Black as external tool
3. Configure isort integration
4. Set up pytest as test runner

## Environment Verification

Run this verification script to ensure everything is working:

```python
# verification.py
import sys
import pkg_resources

required_packages = [
    'numpy', 'pandas', 'scipy', 'neurokit2', 
    'pyyaml', 'matplotlib', 'seaborn', 'plotly',
    'joblib', 'tqdm', 'pytest'
]

print(f"Python version: {sys.version}")
print("\\nPackage versions:")

for package in required_packages:
    try:
        version = pkg_resources.get_distribution(package).version
        print(f"✓ {package}: {version}")
    except pkg_resources.DistributionNotFound:
        print(f"✗ {package}: NOT INSTALLED")

print("\\nEnvironment verification complete!")
```

## Common Issues and Solutions

### Issue: NeuroKit2 installation fails
**Solution**: Ensure you have the latest pip and try:
```bash
pip install --upgrade pip
pip install neurokit2
```

### Issue: Permission errors on Windows
**Solution**: Run command prompt as administrator or use:
```bash
pip install --user -e .
```

### Issue: Import errors with matplotlib on headless systems
**Solution**: Set matplotlib backend:
```python
import matplotlib
matplotlib.use('Agg')  # For headless systems
```

### Issue: Out of memory with large datasets
**Solution**: Configure processing parameters in `config/config.yaml`:
```yaml
processing:
  n_jobs: 1  # Reduce parallel processes
  chunk_size: 1000  # Process in smaller chunks
```

## Data Directory Setup

The pipeline expects data in BIDS format and outputs to a modular derivatives structure:

```bash
# Create log directory for runtime logs
mkdir -p log

# Data structure is already organized as:
# data/
# ├── sourcedata/              # Raw BIDS data (input)
# │   └── sub-{subject}/
# │       └── ses-{session}/
# │           └── physio/      # BVP, EDA, HR files
# └── derivatives/             # Processed data (output)
#     └── preprocessing/
#         └── sub-{subject}/
#             └── ses-{session}/
#                 ├── bvp/     # BVP outputs (9 files)
#                 ├── eda/     # EDA outputs (13 files)
#                 └── hr/      # HR outputs (7 files)
```

## Running the Preprocessing Pipelines

After successful installation, you can run the preprocessing scripts:

```bash
# BVP preprocessing
poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject g01p01 --session 01

# EDA preprocessing
poetry run python scripts/physio/preprocessing/preprocess_eda.py --subject g01p01 --session 01

# HR preprocessing
poetry run python scripts/physio/preprocessing/preprocess_hr.py --subject g01p01 --session 01

# Clean outputs (if needed)
poetry run python scripts/utils/clean_outputs.py --subject g01p01 --session 01
```

## Configuration

Copy and customize the configuration file:

```bash
# The config file is already created at config/config.yaml
# Edit it according to your specific needs:
nano config/config.yaml  # or your preferred editor
```

## Next Steps

After successful environment setup:

1. **Run tests**: `poetry run pytest tests/` to ensure everything works (should see 34/34 passing)
2. **Check code quality**: Run linting tools (Black, isort, flake8)
3. **Review configuration**: Customize `config/config.yaml` for your needs
4. **Run a test subject**: Try preprocessing a single subject with one of the scripts above
5. **Review documentation**: Check `README.md` and `QUICKREF.md` for usage examples
6. **Explore next development**: Review `TODO.md` for available features to implement

## Project Status

Current implementation includes:
- ✅ **BVP Pipeline**: Blood volume pulse preprocessing with 18 HRV metrics
- ✅ **EDA Pipeline**: Electrodermal activity preprocessing with 23 metrics
- ✅ **HR Pipeline**: Heart rate preprocessing with basic HR metrics
- ✅ **BIDS Compliance**: All outputs follow BIDS derivative format
- ✅ **Comprehensive Testing**: 34/34 unit tests passing
- ✅ **Modular Architecture**: Easy to extend with new modules

## Troubleshooting

If you encounter issues:

1. **Check Python version**: Ensure Python 3.9+ is installed (`python --version`)
2. **Verify dependencies**: Run `poetry show` to see installed packages
3. **Ensure proper virtual environment activation**: `poetry shell`
4. **Check file permissions**: Ensure you have write access to data/ and log/ directories
5. **Review error logs**: Check `log/` directory for detailed error messages
6. **Clean and reinstall**: `poetry env remove python3 && poetry install`

For additional help, refer to:
- **README.md**: Main project documentation with usage examples
- **QUICKREF.md**: Quick reference guide for common commands
- **docs/troubleshooting.md**: Comprehensive troubleshooting guide
- **docs/api_reference.md**: Complete API documentation
- **NeuroKit2 documentation**: https://neurokit2.readthedocs.io/
- **Project TODO**: Check `TODO.md` for known issues and future features
- **GitHub Issues**: https://github.com/Ramdam17/TherasyncAnalysis/issues
- **Contact**: remy.ramadour.labs@gmail.com