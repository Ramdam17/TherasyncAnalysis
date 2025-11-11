# Therasync Pipeline - TODO List

**Last Updated:** November 11, 2025  
**Current Version:** v1.0.0  
**Current Status:** Production Ready - Phase 2 Harmonization Complete

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
  1. Multi-signal Dashboard - BVP, HR, EDA synchronized overview
  2. Poincaré Plot - HRV non-linear dynamics (SD1/SD2)
  3. Autonomic Balance - LF/HF ratio timeline
  4. EDA Arousal Profile - Tonic/phasic with SCR events
  5. SCR Distribution - Amplitude histogram + statistics
  6. HR Dynamics Timeline - HR evolution with rest/moderate/elevated zones
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
- ✅ Created quality analysis script: `scripts/analysis/generate_quality_report.py`
  - Comprehensive signal quality assessment
  - 114 quality flags tracked across all modalities
  - Per-session and aggregate statistics

**Testing & Validation:**
- ✅ Successfully processed 49/51 sessions (96% success rate)
  - 2 failed sessions: Missing/empty source data (expected)
- ✅ Generated 306 visualizations (51 sessions × 6 plots, 100% success)
- ✅ Processing time: ~3 minutes preprocessing + ~3 minutes visualization
- ✅ All plots validated on real data from 29 subjects
- ✅ 34/34 unit tests passing (100%)

**Documentation:**
- ✅ Created QUICKSTART.md - Fast processing guide
- ✅ Updated README.md with visualization and batch processing sections
- ✅ Created comprehensive QUICKREF.md - One-page command reference
- ✅ Updated TODO.md with Sprint 5 completion

### Phase 7: Code Harmonization (Phase 2) - COMPLETE ✨
**Completed November 11, 2025**

**Architecture Harmonization:**
- ✅ Unified all three BIDS writers (BVP, EDA, HR) with identical code patterns
- ✅ Standardized helper method signatures across all modalities:
  - `_write_physio_file()` - Consistent signal TSV writing
  - `_write_physio_metadata()` - Consistent JSON sidecar creation
  - `_write_events_file()` - Unified event TSV writing
  - `_write_events_metadata()` - Unified event JSON sidecar
  - `_write_metrics_file()` - Standardized metrics TSV output
  - `_write_metrics_metadata()` - Standardized metrics JSON sidecar
  - `_write_summary_file()` - Consistent summary generation
- ✅ Unified variable naming conventions:
  - `subject_dir` - Subject/session directory path
  - `base_filename` - BIDS-compliant file prefix
  - `signals_tsv` - Signal TSV file path
  - `signals_json` - Signal JSON file path
- ✅ Consistent BIDS path construction across all writers
- ✅ Fixed EDA JSON/TSV bug (JSON was overwriting TSV)

**Visualization Integration:**
- ✅ Updated data loader for new HR format:
  - Changed from compressed combined file to per-moment uncompressed files
  - Load `task-{moment}_desc-processed_recording-hr.tsv` for each moment
  - Load `desc-hr-summary.json` for metrics
- ✅ Updated visualization plotters for per-moment HR structure:
  - `plot_hr_signal()` - Iterate over restingstate/therapy moments
  - `plot_hr_dynamics_timeline()` - Use per-moment data structure
  - Both use `HR_Clean` column from new format

**Testing & Validation:**
- ✅ All 34 unit tests passing after harmonization
- ✅ Full preprocessing run: 49/51 sessions successful (96%)
- ✅ Complete visualization regeneration: 306/306 successful (100%)
- ✅ Quality report: 114 flags identified (same profile as before)
- ✅ No new issues introduced by harmonization

**Git Management:**
- ✅ Branch: `refactor/code-cleanup`
- ✅ Commits:
  - `fc8fcba` - Phase 2 COMPLETE: All three BIDS writers fully harmonized
  - `e95f727` - Visualization pipeline updated for Phase 2 HR format
  - `78e8209` - Documentation updated for Phase 2 completion
- ✅ Ready to merge to master

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

**Version:** v1.0.0 (Production Ready)  
**Branch:** refactor/code-cleanup  
**Tests:** 34/34 passing (100%)  
**Pipelines:** All complete and harmonized  
**Documentation:** Complete and updated  
**Data Processed:** 49/51 sessions successfully (96% success rate)

**Production Ready Components:**
- ✅ BVP preprocessing (9 files per session)
- ✅ EDA preprocessing (13 files per session)
- ✅ HR preprocessing (14 files per session: 7 per moment)
- ✅ Visualization generation (6 plots per session, 100% success)
- ✅ Batch processing (automated pipeline with error handling)
- ✅ Quality analysis (114 quality flags tracked)
- ✅ Code harmonization (identical patterns across all BIDS writers)

**Performance Benchmarks:**
- Preprocessing: ~3 seconds per session (~3 minutes for 51 sessions)
- Visualization: ~3.4 seconds per session (~3 minutes for 51 sessions)
- Quality Report: <1 second for full dataset
- Total throughput: ~7 seconds per subject/session
- Success rates: 96% preprocessing, 100% visualization

**System Health:**
- ✅ All modular architecture refactoring complete
- ✅ All three BIDS writers fully harmonized
- ✅ Visualization pipeline integrated with new HR format
- ✅ Comprehensive error handling and logging
- ✅ BIDS-compliant outputs validated
- ✅ No regressions introduced

**Ready for:**
- ✅ Merge to master
- ✅ Tag release as v1.0.0
- ✅ Begin next development phase (synchrony analysis or advanced features)

---

## 🎯 Immediate Next Steps

1. **✅ DONE - Phase 2 Harmonization Complete**
   - All BIDS writers harmonized with identical code patterns
   - Visualization pipeline updated for new HR format
   - Documentation updated
   - All tests passing

2. **Merge to master**
   ```bash
   git checkout master
   git merge refactor/code-cleanup
   git tag v1.0.0 -m "Production release: Harmonized BIDS writers + full visualization pipeline"
   git push origin master --tags
   ```

3. **Choose next development phase**
   - Option A: Synchrony Analysis Module (dyadic physiological analysis)
   - Option B: Advanced Visualizations (interactive HTML, PDF reports)
   - Option C: Performance Optimization (parallel processing, caching)
   - Option D: Data Quality Enhancement (advanced artifact detection)

---

*Pipeline is production-ready with complete preprocessing, visualization, batch processing, and quality analysis capabilities. All code harmonized and fully tested.*