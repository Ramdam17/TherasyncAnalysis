# Environment Setup Guide

This guide helps you set up the development environment for the Therasync Pipeline project.

## Prerequisites

- **Python 3.8+** (recommended: Python 3.9 or 3.10)
- **Git** for version control
- **pip** for package management

## Installation Methods

### Method 1: Standard pip installation (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd therasync

# Create a virtual environment
python -m venv therasync-env

# Activate virtual environment
# On Linux/Mac:
source therasync-env/bin/activate
# On Windows:
therasync-env\Scripts\activate

# Install the project in development mode
pip install -e .

# Verify installation
python -c "import neurokit2; import pandas; import numpy; print('All packages installed successfully!')"
```

### Method 2: Poetry (Alternative)

If you prefer Poetry for dependency management:

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Clone and setup
git clone <repository-url>
cd therasync

# Install dependencies
poetry install

# Activate environment
poetry shell
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
# Run all tests
pytest tests/

# Run tests with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_config_loader.py

# Run tests in verbose mode
pytest -v tests/
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

The pipeline expects data in BIDS format. Create the derivatives directory:

```bash
# Create derivatives directory structure
mkdir -p data/derivatives/therasync-physio
mkdir -p log
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

1. **Run tests**: `pytest tests/` to ensure everything works
2. **Check code quality**: Run linting tools
3. **Review configuration**: Customize `config/config.yaml`
4. **Start development**: Begin with Sprint 1 tasks in `TODO.md`

## Troubleshooting

If you encounter issues:

1. Check Python version compatibility
2. Verify all dependencies are installed
3. Ensure proper virtual environment activation
4. Check file permissions
5. Review error logs in `log/` directory

For additional help, refer to:
- **NeuroKit2 documentation**: https://neurokit2.readthedocs.io/
- **Project issues**: Check the TODO.md for known issues
- **Contact**: remy.ramadour.labs@gmail.com