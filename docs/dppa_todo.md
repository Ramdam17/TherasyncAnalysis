# DPPA Development TODO List

**Project**: Dyadic Poincaré Plot Analysis (DPPA)  
**Branch**: `feature/dppa`  
**Target Version**: v1.2.0  
**Last Updated**: 2025-11-12

---

## Overview

Implementation of DPPA module for analyzing physiological synchrony between dyads using Inter-Centroid Distances (ICD) computed from Poincaré plot centroids.

**3-Step Architecture**:
1. **STEP 1**: Compute Poincaré centroids per participant/session/epoch
2. **STEP 2**: Generate dyad mappings (inter-session: all pairs, intra-family: same session)
3. **STEP 3**: Calculate ICD between dyad pairs

---

## Progress Tracker

### ✅ Phase 1: Planning & Setup (2/2 completed)

- [x] **Create DPPA feature branch**
  - Status: ✅ DONE (2025-11-12)
  - Branch: `feature/dppa` created from master
  - All DPPA development will happen on this branch

- [x] **Design DPPA module architecture (REVISED)**
  - Status: ✅ DONE (2025-11-12)
  - Updated architecture with 3-step process
  - Modules: `poincare_calculator.py`, `centroid_loader.py`, `icd_calculator.py`, `dyad_config_loader.py`, `dppa_writer.py`
  - Max 200 lines per file (per copilot-instructions.md)

---

### ✅ Phase 2: Configuration & Infrastructure (1/1 completed)

- [x] **Generate dyad configuration file**
  - Status: ✅ DONE (2025-11-12)
  - File: `config/dppa_dyads.yaml`
  - Task: Scan `data/derivatives/epoched/` to identify all families/sessions/tasks
  - Auto-generate config with:
    - `inter_session`: all pairs, all tasks (nsplit120 method)
    - `intra_family`: same session, all tasks (sliding method)
  - Result: 6 families, 51 sessions total, 2 tasks (restingstate, therapy)

---

### ✅ Phase 3: Core Modules Implementation (1/5 completed)

- [x] **Implement PoincareCalculator module (STEP 1)**
  - Status: ✅ DONE (2025-11-12)
  - File: `src/physio/dppa/poincare_calculator.py`
  - Task: For each participant/session/task/method, compute Poincaré centroids per epoch
  - Columns: `epoch_id`, `centroid_x` (mean RRₙ), `centroid_y` (mean RRₙ₊₁), `sd1`, `sd2`, `sd_ratio`, `n_intervals`
  - Handle NaN for empty epochs
  - Tested: ✓ f01p01, ✓ f02p01, ✓ nsplit120, ✓ sliding methods

- [ ] **Implement CentroidLoader module**
  - Status: ⏳ NOT STARTED
  - File: `src/physio/dppa/centroid_loader.py`
  - Task: Load pre-computed Poincaré centroid files
  - Provide methods to retrieve centroid data by subject/session/task/method
  - Cache loaded data for performance

- [ ] **Implement ICDCalculator module (STEP 3)**
  - Status: ⏳ NOT STARTED
  - File: `src/physio/dppa/icd_calculator.py`
  - Task: Calculate Inter-Centroid Distances
  - Formula: `ICD = √[(x̄₁ - x̄₂)² + (ȳ₁ - ȳ₂)²]`
  - Input: two centroid series from CentroidLoader
  - Handle NaN propagation (if either centroid is NaN, ICD = NaN)

- [ ] **Implement DyadConfigLoader module (STEP 2)**
  - Status: ⏳ NOT STARTED
  - File: `src/physio/dppa/dyad_config_loader.py`
  - Task: Load and parse `config/dppa_dyads.yaml`
  - Methods: `get_inter_session_pairs()`, `get_intra_family_pairs()`
  - Validate configuration structure
  - Support multiple tasks (restingstate, therapy, future tasks)

- [ ] **Implement DPPAWriter module**
  - Status: ⏳ NOT STARTED
  - File: `src/physio/dppa/dppa_writer.py`
  - Task: Export ICD results to CSV
  - Inter-session: rectangular CSV per task (120 rows × ~1275 columns)
  - Intra-family: single CSV per task with `dyad_id` column (variable rows)
  - Include metadata in JSON sidecars

---

### 🔄 Phase 4: CLI Scripts (0/2 completed)

- [ ] **Create compute_poincare.py CLI script (STEP 1)**
  - Status: ⏳ NOT STARTED
  - File: `scripts/physio/dppa/compute_poincare.py`
  - CLI flags: `--subject`, `--session`, `--batch`
  - Compute Poincaré centroids for all epoching methods
  - Follow CLI pattern from copilot-instructions.md
  - Horodated logging

- [ ] **Create compute_dppa.py CLI script (STEP 3)**
  - Status: ⏳ NOT STARTED
  - File: `scripts/physio/dppa/compute_dppa.py`
  - CLI flags: `--mode {inter|intra|both}`, `--task {restingstate|therapy|all}`, `--batch`
  - Orchestrates: centroid loading, dyad pairing, ICD calculation, CSV export

---

### 🔄 Phase 5: Testing (0/5 completed)

- [ ] **Write comprehensive tests**
  - Status: ⏳ NOT STARTED
  - File: `tests/test_dppa.py`
  - Unit tests for all DPPA modules
  - Test: Poincaré centroid calculation (RRₙ vs RRₙ₊₁ pairing), ICD computation, NaN handling, dyad pairing, multi-task support, CSV export
  - Aim for >80% coverage

- [ ] **Test Poincaré calculation on single session**
  - Status: ⏳ NOT STARTED
  - Command: `compute_poincare.py` on `sub-f01p01/ses-01` for all tasks/methods
  - Validate: `centroid_x`, `centroid_y`, `sd1`, `sd2` values
  - Manual inspection of Poincaré plot coordinates

- [ ] **Batch compute all Poincaré centroids**
  - Status: ⏳ NOT STARTED
  - Command: `compute_poincare.py --batch`
  - Process: all 51 sessions × 2+ tasks × 3 methods = ~300 files
  - Verify BIDS structure in `data/derivatives/dppa/poincare/`

- [ ] **Test ICD calculation on single dyad**
  - Status: ⏳ NOT STARTED
  - Command: `compute_dppa.py` on one intra-family dyad (f01p01-ses01 vs f01p02-ses01, task-therapy, sliding method)
  - Validate: ICD values and CSV format

- [ ] **Run batch DPPA inter-session mode**
  - Status: ⏳ NOT STARTED
  - Command: `--mode inter --task all --batch`
  - Compute: ~1275 dyad pairs × 2+ tasks with nsplit120
  - Verify: rectangular CSV outputs per task (120 rows × 1275 columns)

---

### 🔄 Phase 6: Validation & Documentation (0/3 completed)

- [ ] **Run batch DPPA intra-family mode**
  - Status: ⏳ NOT STARTED
  - Command: `--mode intra --task all --batch`
  - Compute: all intra-family dyads × 2+ tasks with sliding method
  - Verify: CSV with `dyad_id` column per task

- [ ] **Update documentation**
  - Status: ⏳ NOT STARTED
  - Files: `docs/api_reference.md`, `docs/resources.md`
  - Document: all DPPA modules, output structures, ICD formula, Poincaré centroid calculation, multi-task support
  - Include: interpretation guidelines

- [ ] **Commit changes with descriptive messages**
  - Status: ⏳ NOT STARTED
  - Atomic commits:
    1. `config/dppa_dyads.yaml`
    2. poincare_calculator module
    3. ICD modules
    4. CLI scripts
    5. tests
    6. docs
  - Format: `feat(dppa): description`

---

### 🔄 Phase 7: Finalization (0/1 completed)

- [ ] **Merge feature/dppa to master**
  - Status: ⏳ NOT STARTED
  - After user approval, merge with `--no-ff`
  - Push to GitHub
  - Tag as `v1.2.0` (DPPA module release)

---

## Summary Statistics

- **Total Tasks**: 19
- **Completed**: 4 (21%)
- **In Progress**: 0 (0%)
- **Not Started**: 15 (79%)

**Next Task**: Implement CentroidLoader module (`src/physio/dppa/centroid_loader.py`)

---

## Key Technical Decisions

1. **Distance Metric**: Euclidean distance (simple formula), not Poincaré distance with SD1/SD2
2. **Empty Epochs**: NaN (no interpolation)
3. **Intra-family Format**: Single CSV with `dyad_id` column (option 3)
4. **Auto-generation**: Config file generated automatically
5. **Tasks to Compare**: Both `restingstate` AND `therapy` (for baseline comparison)
6. **Naming Format**: `f01p01-ses01_f02p03-ses02` (explicit)
7. **Config Separation**: `config/dppa_dyads.yaml` separate from main config
8. **3-Step Process**: (1) Compute Poincaré centroids, (2) Generate dyad mappings, (3) Calculate ICDs

---

## Notes

- All modules must follow copilot-instructions.md guidelines:
  - Max 200 lines per file
  - Type hints required
  - Tests mandatory (>80% coverage)
  - All code/docs in English
  - Ask before creating/modifying files
  - Module initialization: `config_path` parameter

- Expected outputs:
  - ~300 Poincaré centroid files (51 sessions × 2 tasks × 3 methods)
  - ~1275 inter-session dyad pairs (51 choose 2)
  - Variable intra-family dyads (depends on family size per session)
