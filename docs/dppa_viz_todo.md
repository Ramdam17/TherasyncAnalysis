# DPPA Visualization TODO

**Branch:** `feature/dppa-viz`  
**Status:** Planning Phase  
**Created:** 2025-11-12  
**Objective:** Create visualization system for DPPA analysis with dyad-level temporal plots

---

## Requirements Summary

### Main Goal
Generate 4-subplot figures for each dyad (nsplit method) showing:
1. **Row 1 (full width):** ICD evolution over time
   - **Therapy data:** time series in red
   - **Resting state baseline:** horizontal line (single epoch value)
   - **Trendline:** linear fit with slope coefficient displayed
2. **Row 2, Col 1:** SD1 evolution per subject
   - **Therapy data:** time series, one color per subject
   - **Resting state baselines:** horizontal lines for each subject
3. **Row 2, Col 2:** SD2 evolution per subject
   - **Therapy data:** time series, one color per subject
   - **Resting state baselines:** horizontal lines for each subject
4. **Row 2, Col 3:** SD1/SD2 ratio evolution per subject
   - **Therapy data:** time series, one color per subject
   - **Resting state baselines:** horizontal lines for each subject

**Key Design Elements:**
- Resting state = single epoch → horizontal baseline (all subplots)
- ICD trendline with slope coefficient (subplot 1 only)
- No vertical separator bars (different approach than initially planned)

### Technical Requirements
- Follow `.github/copilot-instructions.md` rules:
  - Max 200 lines per file
  - English documentation/code
  - Type hints everywhere
  - Comprehensive docstrings
  - Unit tests for ALL functions
  - Logging (not print)
- Test-driven development approach
- BIDS-compliant output paths
- **Configuration:** Load plot settings from `config/config.yaml`
  - DPI: 150
  - Figure size: 12x8 inches (adjustable)
  - Output format: PNG only
  - Color scheme: configurable

---

## Architecture Plan

### Module Structure (`src/physio/dppa/`)
```
src/physio/dppa/
├── __init__.py (existing)
├── poincare_calculator.py (existing)
├── centroid_loader.py (existing)
├── dyad_config_loader.py (existing)
├── icd_calculator.py (existing)
├── dppa_writer.py (existing)
├── dyad_icd_loader.py (NEW)        # Load ICD data for a specific dyad
├── dyad_centroid_loader.py (NEW)   # Load centroid data for dyad members
└── dyad_plotter.py (NEW)           # Generate 4-subplot visualization
```

### CLI Script (`scripts/physio/dppa/`)
```
scripts/physio/dppa/
├── compute_poincare.py (existing)
├── compute_dppa.py (existing)
└── plot_dyad.py (NEW)              # CLI for dyad visualization
```

### Test Files (`tests/`)
```
tests/
├── test_dppa.py (existing - 22 tests)
└── test_dppa_viz.py (NEW)          # Tests for visualization modules
```

---

## Detailed Task List

### Phase 1: Data Loading Modules

#### Task 1.1: Create `dyad_icd_loader.py`
**File:** `src/physio/dppa/dyad_icd_loader.py`  
**Responsibility:** Load ICD time series for a specific dyad (both tasks)

**Requirements:**
- [ ] Class `DyadICDLoader` with ConfigLoader pattern
- [ ] Method `load_icd(dyad_pair: str, task: str, method: str) -> pd.DataFrame`
  - Input: `dyad_pair="f01p01_ses-01_vs_f01p02_ses-01"`, `task="restingstate"` or `"therapy"`, `method="nsplit120"`
  - Output: DataFrame with columns: `epoch_id`, `icd_value`
  - Path: `data/derivatives/dppa/inter_session/inter_session_icd_task-{task}_method-{method}.csv`
- [ ] Method `load_both_tasks(dyad_pair: str, method: str) -> dict[str, pd.DataFrame]`
  - Load restingstate and therapy ICDs
  - Return: `{"restingstate": df_resting, "therapy": df_therapy}`
  - Restingstate will have 1 epoch, therapy will have multiple
- [ ] Method `parse_dyad_info(dyad_pair: str) -> dict`
  - Extract: `subject1`, `session1`, `subject2`, `session2`
  - Return: `{"sub1": "f01p01", "ses1": "01", "sub2": "f01p02", "ses2": "01"}`
- [ ] Error handling: file not found, invalid dyad format, missing columns
- [ ] Logging: INFO for successful load, ERROR for failures
- [ ] Type hints and comprehensive docstrings
- [ ] Max 200 lines

**Dependencies:** pandas, pathlib, logging, ConfigLoader

#### Task 1.2: Create `dyad_centroid_loader.py`
**File:** `src/physio/dppa/dyad_centroid_loader.py`  
**Responsibility:** Load Poincaré centroid data for both dyad members (both tasks)

**Requirements:**
- [ ] Class `DyadCentroidLoader` with ConfigLoader pattern
- [ ] Method `load_centroids(dyad_info: dict, task: str, method: str) -> tuple[pd.DataFrame, pd.DataFrame]`
  - Input: dyad_info from DyadICDLoader, task (`restingstate` or `therapy`), method
  - Output: (df_subject1, df_subject2)
  - Each DataFrame columns: `epoch_id`, `centroid_x`, `centroid_y`, `sd1`, `sd2`, `sd_ratio`, `n_intervals`
  - Path pattern: `data/derivatives/dppa/sub-{sub}/ses-{ses}/poincare/sub-{sub}_ses-{ses}_task-{task}_method-{method}_desc-poincare_physio.tsv`
- [ ] Method `load_both_tasks(dyad_info: dict, method: str) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]`
  - Load centroids for both tasks
  - Return: `{"restingstate": (df_sub1_rest, df_sub2_rest), "therapy": (df_sub1_therapy, df_sub2_therapy)}`
  - Restingstate will have 1 epoch per subject, therapy will have multiple
- [ ] Method `validate_epoch_alignment(df1: pd.DataFrame, df2: pd.DataFrame) -> bool`
  - Check both subjects have same epoch_ids
  - Log WARNING if mismatch
- [ ] Error handling: missing files, invalid format, misaligned epochs
- [ ] Logging: INFO for loads, WARNING for alignment issues
- [ ] Type hints and comprehensive docstrings
- [ ] Max 200 lines

**Dependencies:** pandas, pathlib, logging, ConfigLoader

#### Task 1.3: Unit Tests for Data Loaders
**File:** `tests/test_dppa_viz.py` (Part 1)

**Requirements:**
- [ ] Test class `TestDyadICDLoader`
  - [ ] `test_load_icd_valid_dyad()` - Load real ICD data
  - [ ] `test_load_icd_missing_file()` - FileNotFoundError
  - [ ] `test_parse_dyad_info_valid()` - Extract subject/session info
  - [ ] `test_parse_dyad_info_invalid_format()` - ValueError
- [ ] Test class `TestDyadCentroidLoader`
  - [ ] `test_load_centroids_valid_dyad()` - Load both subjects
  - [ ] `test_load_centroids_missing_file()` - Handle missing subject data
  - [ ] `test_validate_epoch_alignment_match()` - Aligned epochs
  - [ ] `test_validate_epoch_alignment_mismatch()` - Misaligned epochs warning
- [ ] Use pytest fixtures for test data paths
- [ ] Mock ConfigLoader if needed

**Target:** 8+ tests, 100% passing

---

### Phase 2: Visualization Module

#### Task 2.1: Create `dyad_plotter.py`
**File:** `src/physio/dppa/dyad_plotter.py`  
**Responsibility:** Generate 4-subplot dyad visualization

**Requirements:**
- [ ] Class `DyadPlotter` with ConfigLoader pattern
- [ ] Method `plot_dyad(icd_data: dict, centroid_data: dict, dyad_info: dict, method: str, output_path: Path) -> None`
  - Input data structure:
    - `icd_data = {"restingstate": df_rest, "therapy": df_therapy}`
    - `centroid_data = {"restingstate": (df_sub1_rest, df_sub2_rest), "therapy": (df_sub1_therapy, df_sub2_therapy)}`
  - Create figure with 2 rows:
    - Row 1: Single subplot (full width) for ICD
    - Row 2: 3 subplots (SD1, SD2, SD1/SD2)
  - Use matplotlib GridSpec for layout
  - Load settings from config: DPI (150), figure size (12x8), colors
  - Color scheme:
    - Therapy data: red
    - Resting baselines: blue (dashed horizontal lines)
    - Subject 1: #1f77b4
    - Subject 2: #ff7f0e
  - Labels, legend, title with dyad info
  - Save as PNG (150 DPI from config)
- [ ] Method `_calculate_trendline(therapy_icd: pd.DataFrame) -> tuple[np.ndarray, float]`
  - Compute linear regression on therapy ICD data
  - Return: (fitted_values, slope_coefficient)
  - Use numpy.polyfit or scipy.stats.linregress
- [ ] Method `_plot_icd_subplot(ax, icd_data, trendline_data)` 
  - Plot therapy ICD time series (red)
  - Plot resting baseline (blue dashed horizontal)
  - Plot trendline (black dashed)
  - Display slope coefficient on plot (e.g., "slope = -0.05")
- [ ] Method `_plot_sd1_subplot(ax, centroid_data)` 
  - Plot therapy SD1 for both subjects (colored lines)
  - Plot resting baselines (dashed horizontal)
- [ ] Method `_plot_sd2_subplot(ax, centroid_data)` 
  - Plot therapy SD2 for both subjects (colored lines)
  - Plot resting baselines (dashed horizontal)
- [ ] Method `_plot_ratio_subplot(ax, centroid_data)` 
  - Plot therapy SD1/SD2 ratio for both subjects (colored lines)
  - Plot resting baselines (dashed horizontal)
- [ ] Error handling: invalid data, plotting failures
- [ ] Logging: INFO for successful plots, ERROR for failures
- [ ] Type hints and comprehensive docstrings
- [ ] Max 200 lines (may need to split if exceeds)

**Dependencies:** matplotlib, pandas, pathlib, logging, ConfigLoader, numpy, scipy (optional)

**Notes:**
- If file exceeds 200 lines, split into:
  - `dyad_plotter.py` - Main class and orchestration
  - `dyad_plot_utils.py` - Helper functions for individual subplots

#### Task 2.2: Unit Tests for Plotter
**File:** `tests/test_dppa_viz.py` (Part 2)

**Requirements:**
- [ ] Test class `TestDyadPlotter`
  - [ ] `test_plot_dyad_creates_file()` - Output file exists
  - [ ] `test_determine_phase_boundary()` - Correct boundary detection
  - [ ] `test_plot_with_valid_data()` - No errors on real data
  - [ ] `test_plot_with_misaligned_epochs()` - Handle gracefully
  - [ ] `test_plot_saves_png_and_pdf()` - Both formats created
- [ ] Mock matplotlib.pyplot.savefig to avoid file I/O in tests
- [ ] Use temporary directories for output

**Target:** 5+ tests, 100% passing

---

### Phase 3: CLI Script

#### Task 3.1: Create `plot_dyad.py`
**File:** `scripts/physio/dppa/plot_dyad.py`  
**Responsibility:** Command-line interface for dyad visualization

**Requirements:**
- [ ] Argparse interface:
  - `--dyad`: Dyad pair string (e.g., `f01p01_ses-01_vs_f01p02_ses-01`)
  - `--method`: Method name (`nsplit120`, `sliding_duration30s_step5s`, or `all`)
  - `--output-dir`: Custom output directory (default: `data/derivatives/dppa/figures/`)
  - `--batch`: Process all dyads from config
  - `--config`: Custom config path
- [ ] Single dyad mode (loads BOTH tasks automatically):
  ```bash
  poetry run python scripts/physio/dppa/plot_dyad.py \
    --dyad f01p01_ses-01_vs_f01p02_ses-01 \
    --method nsplit120
  ```
- [ ] Batch mode (all dyads, both tasks):
  ```bash
  poetry run python scripts/physio/dppa/plot_dyad.py --batch --method nsplit120
  ```
- [ ] Output path pattern: `data/derivatives/dppa/figures/sub-{sub1}_ses-{ses1}_vs_sub-{sub2}_ses-{ses2}_method-{method}_desc-dyad_viz.png`
- [ ] Progress logging for batch mode
- [ ] Error handling: continue on failure in batch mode
- [ ] Summary statistics at end (N processed, N failed)
- [ ] Follow CLI pattern from copilot-instructions.md
- [ ] Max 200 lines

**Dependencies:** argparse, sys, pathlib, logging, DyadICDLoader, DyadCentroidLoader, DyadPlotter, DyadConfigLoader (for batch mode)

#### Task 3.2: Integration Tests for CLI
**File:** `tests/test_dppa_viz.py` (Part 3)

**Requirements:**
- [ ] Test class `TestPlotDyadCLI`
  - [ ] `test_cli_single_dyad()` - Run CLI with valid arguments
  - [ ] `test_cli_batch_mode()` - Process multiple dyads
  - [ ] `test_cli_invalid_dyad()` - Handle errors gracefully
  - [ ] `test_cli_missing_data()` - Skip missing files
- [ ] Use subprocess to call CLI script
- [ ] Verify output files created

**Target:** 4+ tests, 100% passing

---

### Phase 4: Documentation & Finalization

#### Task 4.1: Update API Reference
**File:** `docs/api_reference.md`

**Requirements:**
- [ ] Add section "DPPA Visualization Modules"
- [ ] Document `DyadICDLoader` with examples
- [ ] Document `DyadCentroidLoader` with examples
- [ ] Document `DyadPlotter` with examples
- [ ] Add CLI usage examples for `plot_dyad.py`

#### Task 4.2: Update Quick Reference
**File:** `docs/quick_reference.md`

**Requirements:**
- [ ] Add "Visualizing Dyadic Analysis" section
- [ ] Single dyad visualization command
- [ ] Batch visualization command
- [ ] Example output interpretation

#### Task 4.3: Update README
**File:** `README.md`

**Requirements:**
- [ ] Add DPPA Visualization to features list
- [ ] Example visualization command in quickstart

#### Task 4.4: Update Testing Guide
**File:** `docs/testing_guide.md`

**Requirements:**
- [ ] Add test section for DPPA visualization
- [ ] Commands to run viz tests
- [ ] Expected test coverage

---

## Testing Strategy

### Test Coverage Goals
- **Unit Tests:** >80% code coverage for all new modules
- **Integration Tests:** CLI end-to-end tests with real data
- **Visual Tests:** Manual verification of plot outputs (sample images)

### Test Data
- Use existing DPPA outputs from `data/derivatives/dppa/`
- Test with:
  - `f01p01_ses-01_vs_f01p02_ses-01` (therapy, nsplit120)
  - Edge cases: single epoch, missing data, misaligned epochs

### Running Tests
```bash
# All tests
poetry run pytest tests/test_dppa_viz.py -v

# Specific test class
poetry run pytest tests/test_dppa_viz.py::TestDyadICDLoader -v

# With coverage
poetry run pytest tests/test_dppa_viz.py --cov=src/physio/dppa --cov-report=term-missing
```

---

## Implementation Phases Summary

### Phase 1: Data Loading (Est. 2-3 hours)
- [x] Plan architecture
- [ ] Implement `DyadICDLoader`
- [ ] Implement `DyadCentroidLoader`
- [ ] Write 8+ unit tests
- [ ] Verify all tests pass

### Phase 2: Visualization (Est. 3-4 hours)
- [ ] Implement `DyadPlotter`
- [ ] Test with sample dyad
- [ ] Refine plot aesthetics
- [ ] Write 5+ unit tests
- [ ] Verify all tests pass

### Phase 3: CLI (Est. 1-2 hours)
- [ ] Implement `plot_dyad.py`
- [ ] Test single dyad mode
- [ ] Test batch mode
- [ ] Write 4+ integration tests
- [ ] Verify all tests pass

### Phase 4: Documentation (Est. 1 hour)
- [ ] Update API reference
- [ ] Update quick reference
- [ ] Update README
- [ ] Update testing guide

### Total Estimated Time: 7-10 hours

---

## Open Questions

1. **Phase separation detection:**
   - ~~Should we read metadata to find resting/therapy boundary?~~
   - ~~Or assume midpoint of epochs?~~
   - **Decision: RESOLVED** - Resting state is always 1 epoch → horizontal baseline. Therapy is time series.

2. **Color scheme:**
   - Use default matplotlib colors or custom palette?
   - **Proposed:** Subject 1 = `#1f77b4` (blue), Subject 2 = `#ff7f0e` (orange)
   - **Decision: CONFIRMED** ✅

3. **Output formats:**
   - ~~Save PNG only, PDF only, or both?~~
   - **Decision: CONFIRMED** - PNG only, 150 DPI (from config) ✅

4. **Figure size:**
   - Standard size: 12x8 inches?
   - **Decision: CONFIRMED** - Start with 12x8, adjustable later ✅

5. **Batch mode organization:**
   - Group by family? By session? Flat structure?
   - **Proposed:** Flat with descriptive filenames
   - **Decision:** [TO BE CONFIRMED]

6. **Trendline algorithm:**
   - Use numpy.polyfit (degree 1) or scipy.stats.linregress?
   - **Proposed:** numpy.polyfit for simplicity
   - **Decision:** [TO BE CONFIRMED]

---

## Success Criteria

- [ ] All modules respect 200-line limit
- [ ] All functions have type hints and docstrings
- [ ] 17+ unit/integration tests, 100% passing
- [ ] Total test count: 56 (existing) + 17 (new) = 73 tests
- [ ] Code coverage >80% for new modules
- [ ] CLI works in single and batch mode
- [ ] Generated figures are clear and publication-ready
- [ ] Documentation complete and accurate
- [ ] All code in English
- [ ] All commits follow conventional format
- [ ] User approval before merge to master

---

## Notes

- Follow TDD: Write tests before or alongside implementation
- Commit frequently with descriptive messages
- Ask for approval before creating/modifying files
- Use logging, not print statements
- Validate with real data throughout development
