# Therasync Pipeline - TODO List

## Current Phase: Physiological Preprocessing Pipeline

### Sprint 1: Project Setup and Configuration
- [ ] **Setup project structure and config**
  - Create src/, scripts/, tests/, config/, log/, notebooks/ directories
  - Initialize config.yaml with parameters for moments (resting_state, therapy), preprocessing methods, and BIDS structure settings
  - Create initial README.md with project overview

- [ ] **Create Sprint 1 branch**
  - Create git branch 'sprint-1/project-setup-and-config' from master
  - All subsequent commits for this sprint will be on this branch

- [ ] **Design config.yaml structure**
  - Define YAML structure for: moments definition (resting_state/therapy), BVP preprocessing parameters, EDA preprocessing parameters, HR extraction parameters, BIDS output settings, and file paths
  - Include extensibility for future 5-moment studies

- [ ] **Setup dependencies and environment**
  - Update pyproject.toml with required packages: neurokit2, pandas, numpy, scipy, pyyaml, pytest, etc.
  - Create environment setup documentation

- [ ] **Create core utilities module**
  - Implement src/core/config_loader.py for YAML config loading
  - Implement src/core/bids_utils.py for BIDS-compliant file operations
  - Implement src/core/logger_setup.py for logging configuration

- [ ] **Sprint 1 commit and push**
  - Commit project setup, config structure, and core utilities to sprint-1 branch
  - Push to remote repository

### Sprint 2: BVP Pipeline
- [ ] **Create Sprint 2 branch for BVP pipeline**
  - Create git branch 'sprint-2/bvp-preprocessing' from master after merging sprint-1

- [ ] **Implement BVP data loader**
  - Create src/physio/bvp_loader.py to load Empatica BVP data files
  - Handle TSV format with JSON metadata
  - Include moment-based segmentation (resting_state vs therapy)

- [ ] **Research BVP cleaning methods**
  - Investigate Neurokit2 BVP cleaning options and compile comprehensive list
  - Document findings for user review

- [ ] **🔥 DECISION: Select BVP cleaning method**
  - Review cleaning methods with user and select preferred approach
  - Update config.yaml with chosen method and parameters
  - **Requires user input and discussion**

- [ ] **Implement BVP cleaning module**
  - Create src/physio/bvp_cleaner.py implementing the selected cleaning method using Neurokit2
  - Make it configurable via YAML parameters

- [ ] **Research BVP metrics extraction**
  - Compile comprehensive list of all BVP-derived metrics available in Neurokit2
  - Document each metric with descriptions for user review

- [ ] **🔥 DECISION: Select BVP metrics to extract**
  - Review available BVP metrics with user and select which ones to include
  - Update config.yaml with selected metrics list

- [ ] **Implement BVP metrics extraction**
  - Create src/physio/bvp_metrics.py to extract selected metrics using Neurokit2
  - Implement for both resting_state and therapy moments as configured in YAML

- [ ] **Create BVP BIDS output formatter**
  - Implement src/physio/bvp_bids_writer.py to save processed BVP data and metrics in BIDS-compliant format under data/derivatives/
  - Include TSV/JSONL metadata files

- [ ] **Create BVP pipeline script**
  - Create scripts/preprocess_bvp.py that orchestrates the complete BVP pipeline
  - load → clean → extract metrics → save in BIDS format
  - Make it configurable via config.yaml

- [ ] **Write BVP pipeline tests**
  - Create comprehensive unit tests in tests/test_bvp_pipeline.py
  - Cover all BVP processing functions, edge cases and error conditions

- [ ] **Sprint 2 commit and push**
  - Commit complete BVP pipeline to sprint-2 branch
  - Push to remote repository

### Sprint 3: EDA Pipeline
- [ ] **Create Sprint 3 branch for EDA pipeline**
  - Create git branch 'sprint-3/eda-preprocessing' from master after merging sprint-2

- [ ] **Implement EDA data loader**
  - Create src/physio/eda_loader.py to load Empatica EDA data files
  - Handle TSV format with JSON metadata
  - Include moment-based segmentation

- [ ] **Research EDA cleaning methods**
  - Investigate Neurokit2 EDA cleaning options and compile comprehensive list
  - Document findings for user review

- [ ] **🔥 DECISION: Select EDA cleaning method**
  - Review EDA cleaning methods with user and select preferred approach
  - Update config.yaml with chosen method and parameters

- [ ] **Implement EDA cleaning module**
  - Create src/physio/eda_cleaner.py implementing the selected cleaning method using Neurokit2
  - Make it configurable via YAML parameters

- [ ] **Research EDA metrics extraction**
  - Compile comprehensive list of all EDA-derived metrics available in Neurokit2
  - SCR peaks, SCL, phasic/tonic components, etc.
  - Document for user review

- [ ] **🔥 DECISION: Select EDA metrics to extract**
  - Review available EDA metrics with user and select which ones to include
  - Update config.yaml with selected metrics list

- [ ] **Implement EDA metrics extraction**
  - Create src/physio/eda_metrics.py to extract selected metrics using Neurokit2
  - Implement for both resting_state and therapy moments as configured in YAML

- [ ] **Create EDA BIDS output formatter**
  - Implement src/physio/eda_bids_writer.py to save processed EDA data and metrics in BIDS-compliant format under data/derivatives/
  - Include TSV/JSONL metadata files

- [ ] **Create EDA pipeline script**
  - Create scripts/preprocess_eda.py that orchestrates the complete EDA pipeline
  - load → clean → extract metrics → save in BIDS format

- [ ] **Write EDA pipeline tests**
  - Create comprehensive unit tests in tests/test_eda_pipeline.py
  - Cover all EDA processing functions, edge cases and error conditions

- [ ] **Sprint 3 commit and push**
  - Commit complete EDA pipeline to sprint-3 branch
  - Push to remote repository

### Sprint 4: HR Pipeline
- [ ] **Create Sprint 4 branch for HR pipeline**
  - Create git branch 'sprint-4/hr-preprocessing' from master after merging sprint-3

- [ ] **Implement HR extraction from BVP**
  - Create src/physio/hr_extractor.py to derive heart rate from BVP data using Neurokit2
  - Handle moment-based segmentation and quality assessment

- [ ] **Research HR cleaning and processing**
  - Investigate Neurokit2 HR processing options (artifact removal, interpolation methods, etc.)
  - Compile list of available methods for user review

- [ ] **🔥 DECISION: Select HR processing method**
  - Review HR processing methods with user and select preferred approach (e.g., template matching)
  - Update config.yaml with chosen method and parameters

- [ ] **Implement HR cleaning module**
  - Create src/physio/hr_cleaner.py implementing the selected HR processing method using Neurokit2
  - Make it configurable via YAML parameters

- [ ] **Research HR metrics extraction**
  - Compile comprehensive list of all HR-derived metrics available in Neurokit2
  - HRV time-domain, frequency-domain, nonlinear measures
  - Document for user review

- [ ] **🔥 DECISION: Select HR metrics to extract**
  - Review available HR metrics with user and select which ones to include
  - Update config.yaml with selected metrics list

- [ ] **Implement HR metrics extraction**
  - Create src/physio/hr_metrics.py to extract selected metrics using Neurokit2
  - Implement for both resting_state and therapy moments as configured in YAML

- [ ] **Create HR BIDS output formatter**
  - Implement src/physio/hr_bids_writer.py to save processed HR data and metrics in BIDS-compliant format under data/derivatives/
  - Include TSV/JSONL metadata files

- [ ] **Create HR pipeline script**
  - Create scripts/preprocess_hr.py that orchestrates the complete HR pipeline
  - extract from BVP → clean → extract metrics → save in BIDS format

- [ ] **Write HR pipeline tests**
  - Create comprehensive unit tests in tests/test_hr_pipeline.py
  - Cover all HR processing functions, edge cases and error conditions

- [ ] **Sprint 4 commit and push**
  - Commit complete HR pipeline to sprint-4 branch
  - Push to remote repository

### Sprint 5: Integration and Finalization
- [ ] **Create Sprint 5 branch for integration**
  - Create git branch 'sprint-5/pipeline-integration' from master after merging sprint-4

- [ ] **Create master preprocessing script**
  - Create scripts/preprocess_physio.py that orchestrates all three pipelines (BVP, EDA, HR)
  - Complete physiological preprocessing of a subject/session

- [ ] **Implement batch processing capabilities**
  - Add functionality to process multiple subjects/sessions in batch mode
  - Include progress tracking and error handling for large datasets

- [ ] **Create integration tests**
  - Create tests/test_integration.py to test the complete pipeline end-to-end with sample data
  - Verify BIDS compliance and data integrity

- [ ] **Update documentation**
  - Update README.md with complete usage instructions, configuration options, and examples
  - Document the BIDS derivative structure and metadata format

- [ ] **Final TODO list update**
  - Update this TODO list to mark all items as completed
  - Prepare for next phase (DPPA analysis)
  - Archive completed sprint branches

- [ ] **Sprint 5 commit and final merge**
  - Commit integration work to sprint-5 branch
  - Request user approval for final merge to master
  - Complete physiological preprocessing phase

## Key Decision Points Summary
🔥 **Decision points requiring user collaboration:**
1. BVP cleaning method selection
2. BVP metrics selection  
3. EDA cleaning method selection
4. EDA metrics selection
5. HR processing method selection
6. HR metrics selection

## Next Phase
After completing this TODO list, we will move to:
- DPPA (Dyadic Poincaré Plot Analysis) implementation
- Visualization pipeline development
- Advanced physiological analysis methods

---
*Last updated: [Date] by [User/Copilot]*