# Therasync Pipeline - TODO List

**Last Updated:** November 11, 2025  
**Current Version:** v0.5.0  
**Current Phase:** Visualization & Batch Processing Complete

---

## ✅ Completed Work

### Phase 1: Project Setup and Configuration - COMPLETE
### Phase 1: Project Setup and Configuration - COMPLETE
- ✅ Created modular project structure (src/, scripts/, tests/, config/, docs/)
- ✅ Initialized config.yaml with BVP, EDA, and HR parameters
- ✅ Setup Poetry environment with all dependencies (neurokit2, pandas, numpy, scipy, pyyaml, pytest)
- ✅ Implemented core utilities:
  - ConfigLoader for YAML configuration management
  - BIDS utilities for compliant file operations
  - Logger setup for comprehensive logging
- ✅ Created comprehensive README.md and documentation structure

### Phase 2: BVP (Blood Volume Pulse) Pipeline - COMPLETE
**Research & Decision Making:**
- ✅ Researched BVP preprocessing methods (docs/bvp_preprocessing_research.md)
- ✅ **DECISION 1:** Selected NeuroKit2 automatic pipeline with elgendi peak detection
- ✅ Researched BVP/HRV metrics (docs/bvp_metrics_research.md - 40+ metrics catalogued)
- ✅ **DECISION 2:** Selected Extended Set (~18 HRV metrics)
- ✅ All decisions documented in docs/bvp_decisions.md

**Implementation:**
- ✅ Created BVP preprocessing modules:
  - `src/physio/preprocessing/bvp_loader.py` - Load Empatica BVP data (64 Hz)
  - `src/physio/preprocessing/bvp_cleaner.py` - PPG signal processing with elgendi
  - `src/physio/preprocessing/bvp_metrics.py` - Extract 18 HRV metrics
  - `src/physio/preprocessing/bvp_bids_writer.py` - BIDS-compliant output
- ✅ Created CLI script: `scripts/physio/preprocessing/preprocess_bvp.py`
- ✅ Comprehensive unit tests (tests/test_bvp_pipeline.py)
- ✅ End-to-end validation: Successfully processes subjects, outputs 9 BIDS files per session

### Phase 3: EDA (Electrodermal Activity) Pipeline - COMPLETE
**Research & Decision Making:**
- ✅ Researched EDA preprocessing methods (docs/eda_preprocessing_research.md)
- ✅ **DECISION 3:** Selected NeuroKit2 with cvxEDA decomposition (gold standard)
- ✅ Researched EDA metrics (docs/eda_metrics_research.md - 40+ metrics catalogued)
- ✅ **DECISION 4:** Selected Extended Set (23 metrics: 9 SCR + 5 Tonic + 6 Phasic + 3 Metadata)
- ✅ All decisions documented in docs/eda_decisions.md

**Implementation:**
- ✅ Created EDA preprocessing modules:
  - `src/physio/preprocessing/eda_loader.py` - Load Empatica EDA data (4 Hz)
  - `src/physio/preprocessing/eda_cleaner.py` - Tonic/phasic decomposition with cvxEDA
  - `src/physio/preprocessing/eda_metrics.py` - Extract 23 EDA metrics
  - `src/physio/preprocessing/eda_bids_writer.py` - BIDS-compliant output (5 file types)
- ✅ Created CLI script: `scripts/physio/preprocessing/preprocess_eda.py`
- ✅ Comprehensive unit tests (tests/test_eda_pipeline.py)
- ✅ End-to-end validation: Successfully processes subjects, outputs 13 BIDS files per session
- ✅ Real data testing: 5 subjects validated with physiologically reasonable SCR rates

### Phase 4: HR (Heart Rate) Pipeline - COMPLETE
**Research & Decision Making:**
- ✅ Researched HR processing methods (docs/hr_metrics_research.md)
- ✅ Selected basic HR metrics (non-HRV, complementary to BVP pipeline)

**Implementation:**
- ✅ Created HR preprocessing modules:
  - `src/physio/preprocessing/hr_loader.py` - Load direct HR data (1 Hz)
  - `src/physio/preprocessing/hr_cleaner.py` - HR signal cleaning
  - `src/physio/preprocessing/hr_metrics.py` - Extract basic HR metrics
  - `src/physio/preprocessing/hr_bids_writer.py` - BIDS-compliant output
- ✅ Created CLI script: `scripts/physio/preprocessing/preprocess_hr.py`
- ✅ Comprehensive unit tests (tests/test_hr_pipeline.py)
- ✅ End-to-end validation: Successfully processes subjects, outputs 7 BIDS files per session

### Phase 5: Modular Architecture Refactoring - COMPLETE ✨
**Major restructuring completed October 28, 2025**

**Architecture Changes:**
- ✅ Restructured entire codebase into modular preprocessing architecture
- ✅ New directory structure for future extensibility:
  - `src/physio/preprocessing/` - All preprocessing modules (BVP, EDA, HR)
  - `scripts/physio/preprocessing/` - All CLI scripts
  - `scripts/utils/` - General utilities (clean_outputs.py)
- ✅ Preserved git history using `git mv` for all file moves

**Code Consistency:**
- ✅ Standardized all module initialization signatures:
  - All modules now accept `Optional[Union[str, Path]] config_path`
  - Consistent ConfigLoader instantiation pattern across BVP, EDA, HR
- ✅ Fixed EDA module signatures to match BVP/HR pattern
- ✅ Unified BIDS output structure: `data/derivatives/preprocessing/sub-{subject}/ses-{session}/{modality}/`

**Testing & Validation:**
- ✅ All 34 unit tests passing (100% pass rate)
- ✅ Updated test configurations and expectations
- ✅ Fixed metric names to match NeuroKit2 conventions
- ✅ All three pipelines validated end-to-end:
  - BVP: 9 BIDS files per session
  - EDA: 13 BIDS files per session
  - HR: 7 BIDS files per session

**Documentation:**
- ✅ Updated README.md with new modular structure
- ✅ Updated QUICKREF.md with correct command syntax
- ✅ All three pipelines documented as production-ready
- ✅ Clarified new output directory structure
- ✅ Removed PYTHONPATH requirements from all examples

**Git Management:**
- ✅ Branch: refactor/restructure-for-modularity
- ✅ Comprehensive commit with detailed message (commit: 99e8ab7)
- ✅ 7 files changed, 340 insertions(+), 206 deletions(-)
- ✅ Merged to master

### Phase 6: Visualization Pipeline - COMPLETE ✨
**Sprint 5 completed November 11, 2025**

**Implementation:**
- ✅ Created visualization modules:
  - `src/visualization/data_loader.py` - Load preprocessed BVP, EDA, HR data
  - `src/visualization/config.py` - Plot styling and configuration
  - `src/visualization/plotters/signal_plots.py` - Multi-signal dashboard, timeline
  - `src/visualization/plotters/hrv_plots.py` - Poincaré, autonomic balance
  - `src/visualization/plotters/eda_plots.py` - Arousal profile, SCR distribution
- ✅ Created 6 core visualizations per subject/session:
  1. Multi-signal Dashboard - BVP, EDA, HR overview
  2. Poincaré Plot - HRV non-linear dynamics (SD1/SD2)
  3. Autonomic Balance - LF/HF ratio timeline
  4. EDA Arousal Profile - Tonic/phasic with SCR events
  5. SCR Distribution - Amplitude histogram + statistics
  6. HR Dynamics Timeline - Heart rate evolution + variability
- ✅ YAML-configured plot styles (DPI, colors, figure sizes)
- ✅ Automatic moment detection for visualization markers
- ✅ Created CLI script: `scripts/visualization/generate_visualizations.py`

**Batch Processing:**
- ✅ Created batch preprocessing script: `scripts/batch/run_all_preprocessing.py`
  - Scans all subjects/sessions in `data/raw/`
  - Sequential BVP → EDA → HR pipeline execution
  - Comprehensive error tracking and logging
  - Options: `--dry-run`, `--skip-existing`, `--subjects`, `--verbose`
- ✅ Created batch visualization script: `scripts/batch/run_all_visualizations.py`
  - Generates 6 plots for all preprocessed sessions
  - Per-plot error tracking
  - Options: `--dry-run`, `--plots`, `--subjects`, `--verbose`
- ✅ Horodated log files: `log/batch_*_YYYYMMDD_HHMMSS.log`
- ✅ Keyboard interrupt handling (Ctrl+C shows partial results)

**Testing & Validation:**
- ✅ Successfully processed 50/51 sessions (98% success rate)
  - 1 failed session: sub-f02p05/ses-03 (empty HR data in raw file)
- ✅ Generated 300 visualizations (50 sessions × 6 plots)
- ✅ Processing time: ~45 minutes preprocessing + ~3 minutes visualization
- ✅ All plots validated on real data from 29 subjects

**Documentation:**
- ✅ Created QUICKSTART.md - Fast processing guide
- ✅ Updated README.md with visualization and batch processing sections
- ✅ Created docs/quick_reference.md - Comprehensive command reference
- ✅ Updated TODO.md with Sprint 5 completion

**Git Management:**
- ✅ Branch: batch/full-pipeline-execution
- ✅ Ready to commit, merge, and push

---

## 🚀 Next Development Phase

### High Priority

### Option 1: Enhanced Batch Processing
- [ ] Add parallel processing for faster execution (multiprocessing)
- [ ] Implement resume capability after interruption
- [ ] Add email notifications on completion/errors
- [ ] Create dashboard for monitoring batch progress

### Option 2: Advanced Visualizations
- [ ] Add interactive HTML visualizations (Plotly)
- [ ] Create comprehensive PDF reports per subject
- [ ] Add group-level statistical visualizations
- [ ] Implement quality control dashboards

### Medium Priority

### Option 3: Synchrony Analysis Module
Create `src/physio/synchrony/` for dyadic physiological analysis:
- [ ] Research dyadic synchrony methods (DPPA, windowed cross-correlation, etc.)
- [ ] Implement synchrony computation modules
- [ ] Create visualization tools for synchrony patterns
- [ ] Add CLI script for synchrony analysis

### Option 2: Emotion Recognition Module
Create `src/physio/emotion/` for emotion classification:
- [ ] Research emotion recognition from physiological signals
- [ ] Implement feature extraction for emotion detection
- [ ] Train/integrate emotion classification models
- [ ] Add CLI script for emotion analysis

### Option 3: Visualization & Reporting Module
Create `src/visualization/` for data visualization and reporting:
- [ ] Implement signal visualization tools
- [ ] Create summary report generators
- [ ] Add quality control visualizations
- [ ] Generate HTML/PDF reports for analyses

### Option 4: Advanced Preprocessing Features
Enhance existing preprocessing modules:
- [ ] Add real-time quality metrics during processing
- [ ] Implement advanced artifact detection
- [ ] Add motion artifact handling for Empatica data
- [ ] Create preprocessing validation reports

### Low Priority

### Option 5: Data Export & Integration

---

## 📋 Key Decisions Summary

### ✅ All Major Decisions Completed and Validated

**DECISION 1 - BVP Preprocessing Method**
- Selected: Automatic NeuroKit2 with elgendi peak detection
- Rationale: Best balance of accuracy and automation for HRV analysis
- Status: ✅ Implemented, tested, production-ready

**DECISION 2 - BVP Metrics Selection**
- Selected: Extended Set (~18 HRV metrics)
- Rationale: Comprehensive coverage without redundancy
- Status: ✅ Implemented, tested, production-ready

**DECISION 3 - EDA Preprocessing Method**
- Selected: cvxEDA decomposition (gold standard)
- Rationale: Research-grade quality, optimal for 4 Hz sampling
- Configuration: SCR threshold = 0.01 μS
- Status: ✅ Implemented, tested, production-ready

**DECISION 4 - EDA Metrics Selection**
- Selected: Extended Set (23 metrics)
- Rationale: Balances comprehensiveness with clinical interpretability
- Status: ✅ Implemented, tested, production-ready

---

## 📊 Current Status

**Version:** v0.5.0 (Visualization & Batch Processing)  
**Branch:** batch/full-pipeline-execution  
**Tests:** 34/34 passing (100%)  
**Pipelines:** 5/5 complete (BVP, EDA, HR, Visualization, Batch)  
**Documentation:** Complete and up-to-date  
**Data Processed:** 50/51 sessions successfully processed

**Production Ready:**
- ✅ BVP preprocessing (9 files per session)
- ✅ EDA preprocessing (13 files per session)
- ✅ HR preprocessing (7 files per session)
- ✅ Visualization generation (6 plots per session)
- ✅ Batch processing (automated pipeline)

**Performance:**
- Preprocessing: ~1 minute per session
- Visualization: ~3 seconds per session
- Total throughput: ~65 seconds per subject/session
- Success rate: 98% (50/51 sessions)

**Ready for:**
- Commit and push to GitHub
- Merge to master
- Tag release as v0.5.0
- Begin next development phase

---

## 🎯 Immediate Next Steps

1. **Commit Sprint 5 changes**
   ```bash
   git add .
   git commit -m "feat: add visualization pipeline and batch processing

   - Created 6 core visualizations per subject/session
   - Implemented batch preprocessing script (run_all_preprocessing.py)
   - Implemented batch visualization script (run_all_visualizations.py)
   - Added comprehensive logging with horodated files
   - Processed 50/51 sessions successfully (98% success rate)
   - Generated 300 visualizations (6 plots × 50 sessions)
   - Created QUICKSTART.md and updated documentation
   - Updated README, TODO, and quick_reference.md
   
   Sprint 5 complete: 15+ files, +3000 lines"
   ```

2. **Push and merge to master**
   ```bash
   git push origin batch/full-pipeline-execution
   # Create PR or merge directly to master
   git checkout master
   git merge batch/full-pipeline-execution
   git tag v0.5.0
   git push origin master --tags
   ```

3. **Choose next development phase**
   - Review high-priority options (Enhanced Batch Processing, Advanced Visualizations)
   - Discuss with team which module to build next
   - Create new feature branch
   - Begin implementation

---

*Pipeline now includes complete preprocessing, visualization, and batch processing capabilities for physiological data analysis.*