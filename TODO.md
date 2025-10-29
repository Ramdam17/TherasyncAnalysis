# Therasync Pipeline - TODO List

**Last Updated:** October 28, 2025  
**Current Version:** v0.3.0  
**Current Phase:** Post-Refactoring - Ready for Next Development Phase

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
- ✅ Ready to push and merge

---

## 🚀 Next Development Phase

### Option 1: Synchrony Analysis Module
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
- [ ] Add artifact detection and rejection
- [ ] Implement adaptive filtering options
- [ ] Add motion artifact handling
- [ ] Create preprocessing quality metrics

### Option 5: Batch Processing & Automation
Improve workflow automation:
- [ ] Create master preprocessing script for all modalities
- [ ] Implement parallel processing for multiple subjects
- [ ] Add progress tracking and error recovery
- [ ] Create configuration templates for different study designs

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

**Version:** v0.3.0 (Modular Architecture)  
**Branch:** refactor/restructure-for-modularity  
**Tests:** 34/34 passing (100%)  
**Pipelines:** 3/3 production-ready (BVP, EDA, HR)  
**Documentation:** Complete and up-to-date  

**Ready for:**
- Push to GitHub
- Merge to master
- Begin next development phase

---

## 🎯 Immediate Next Steps

1. **Push current branch to GitHub**
   ```bash
   git push origin refactor/restructure-for-modularity
   ```

2. **Create Pull Request or Merge to Master**
   - Review changes
   - Merge refactoring branch
   - Tag release as v0.3.0

3. **Update remaining documentation files**
   - Update all research/decision docs with new paths
   - Review and update troubleshooting guide
   - Update API reference if needed

4. **Choose next development phase**
   - Discuss with team which module to build next
   - Create new feature branch
   - Begin implementation

---

*Architecture now supports easy addition of new modules for synchrony analysis, emotion recognition, visualization, and more.*