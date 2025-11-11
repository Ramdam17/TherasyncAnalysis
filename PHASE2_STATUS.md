# Phase 2 Refactoring - Status Update

**Date**: 2024-11-11  
**Branch**: `refactor/code-cleanup`  
**Status**: Step 2 COMPLETE ✅

---

## Completed Work

### ✅ Step 1: Base Class Implementation (DONE)
- Created `src/physio/preprocessing/base_bids_writer.py`
- Abstract class `PhysioBIDSWriter` with unified interface
- Shared utilities for all modalities

### ✅ Step 2: HR Writer & Preprocessing Refactoring (DONE)

#### HRBIDSWriter Changes
- ✅ Inherits from `PhysioBIDSWriter`
- ✅ Implements `save_processed_data()` API
- ✅ Multi-moment processing (restingstate + therapy)
- ✅ Column names harmonized: `HR_Raw`, `HR_Clean`, `HR_Quality`, `HR_Outliers`, `HR_Interpolated`
- ✅ File naming: `desc-processed_recording-hr.tsv` (uncompressed)
- ✅ Moment-based files instead of combined
- ✅ Return type: `Dict[str, List[Path]]`

#### preprocess_hr.py Changes
- ✅ Updated to use `save_processed_data()` API
- ✅ Processes both moments (restingstate + therapy) in single run
- ✅ Column renaming in preprocessing script
- ✅ Proper error handling and logging

#### Test Results (sub-f01p01/ses-01)
```
✅ 14 files generated (7 per moment)
✅ File naming correct: task-restingstate, task-therapy
✅ Column names: HR_Raw, HR_Clean, HR_Quality, HR_Outliers, HR_Interpolated
✅ Files uncompressed (.tsv, not .tsv.gz)
✅ Metadata JSON files complete
✅ Processing successful: 60 samples (restingstate) + 2789 samples (therapy)
```

**Output Structure**:
```
hr/
├── sub-f01p01_ses-01_task-restingstate_desc-processed_recording-hr.tsv
├── sub-f01p01_ses-01_task-restingstate_desc-processed_recording-hr.json
├── sub-f01p01_ses-01_task-restingstate_events.tsv
├── sub-f01p01_ses-01_task-restingstate_events.json
├── sub-f01p01_ses-01_task-restingstate_desc-hr-metrics.tsv
├── sub-f01p01_ses-01_task-restingstate_desc-hr-metrics.json
├── sub-f01p01_ses-01_task-restingstate_desc-hr-summary.json
├── sub-f01p01_ses-01_task-therapy_desc-processed_recording-hr.tsv
├── sub-f01p01_ses-01_task-therapy_desc-processed_recording-hr.json
├── sub-f01p01_ses-01_task-therapy_events.tsv
├── sub-f01p01_ses-01_task-therapy_events.json
├── sub-f01p01_ses-01_task-therapy_desc-hr-metrics.tsv
├── sub-f01p01_ses-01_task-therapy_desc-hr-metrics.json
└── sub-f01p01_ses-01_task-therapy_desc-hr-summary.json
```

---

## Remaining Work

### ⏳ Step 3: BVP Writer Refactoring (NEXT)

**Files to Modify**:
- `src/physio/preprocessing/bvp_bids_writer.py`
- `scripts/physio/preprocessing/preprocess_bvp.py`

**Changes Required**:

1. **BVPBIDSWriter Inheritance**
   - Inherit from `PhysioBIDSWriter`
   - Implement `_get_modality_name()` → `'bvp'`
   - Implement `save_processed_data()` with unified signature

2. **Simplify Input Format**
   - Current: `Dict[str, Tuple[pd.DataFrame, Dict]]` (DataFrame + processing_info)
   - New: `Dict[str, pd.DataFrame]` (processing_info goes to metadata param)

3. **Column Verification**
   - Current columns: `PPG_Clean`, `PPG_Quality`, etc. (already correct!)
   - Verify all use uppercase convention
   - Add `PPG_Raw` if missing

4. **File Naming**
   - Current: `_physio.tsv` → already correct format
   - Verify: `_desc-processed_recording-bvp.tsv` pattern
   - Check if already using `desc-` prefix

5. **Return Type**
   - Change: `Dict[str, List[str]]` → `Dict[str, List[Path]]`

6. **Update preprocess_bvp.py**
   - Call new API: `save_processed_data()`
   - Pass processing info through metadata parameter
   - Verify moment-based processing

**Estimated Time**: 1-2 hours

---

### ⏳ Step 4: EDA Writer Refactoring

**Files to Modify**:
- `src/physio/preprocessing/eda_bids_writer.py`
- `scripts/physio/preprocessing/preprocess_eda.py`

**Changes Required**:

1. **EDABIDSWriter Inheritance**
   - Inherit from `PhysioBIDSWriter`
   - Implement `_get_modality_name()` → `'eda'`
   - Implement `save_processed_data()` with unified signature

2. **Add Quality Column**
   - Implement `EDA_Quality` column calculation
   - Based on: tonic stability, signal-to-noise, phasic variance
   - Add to output TSV files

3. **Column Verification**
   - Current: `EDA_Raw`, `EDA_Tonic`, `EDA_Phasic`
   - Add: `EDA_Clean`, `EDA_Quality`
   - Ensure uppercase convention throughout

4. **File Naming**
   - Verify: `_desc-processed_recording-eda.tsv` pattern
   - Check SCR events files: `_desc-scr_events.tsv`

5. **Return Type**
   - Change: strings → `Path` objects

**Estimated Time**: 2-3 hours (quality calculation more complex)

---

### ⏳ Step 5: Integration Testing

**Test Subjects**:
1. `sub-f01p01/ses-01` - Already tested with HR ✅
2. `sub-f02p01/ses-01` - Test all three modalities
3. `sub-f03p01/ses-01` - Verify consistency

**Test Checklist**:
- [ ] Run all three preprocessors on same subject
- [ ] Verify file structure consistency across modalities
- [ ] Check column name consistency
- [ ] Validate with existing visualization pipeline
- [ ] Compare metrics values (should match previous processing)
- [ ] Check error handling (missing files, bad data)
- [ ] Performance benchmarking

**Validation Script**:
```bash
# Process one subject with all modalities
poetry run python scripts/physio/preprocessing/preprocess_bvp.py --subject f02p01 --session 01
poetry run python scripts/physio/preprocessing/preprocess_eda.py --subject f02p01 --session 01
poetry run python scripts/physio/preprocessing/preprocess_hr.py --subject f02p01 --session 01

# Verify output structure
find data/derivatives/preprocessing/sub-f02p01/ses-01/ -type f | sort

# Run visualizations
poetry run python scripts/physio/visualization/visualize_physio.py --subject f02p01 --session 01
```

---

## Known Issues / Notes

### Metric Names Still Lowercase
- HR metrics currently use lowercase: `hr_mean`, `hr_std`, etc.
- **Planned**: Phase 4 will rename to uppercase: `HR_Mean`, `HR_Std`
- **Impact**: None for now, Phase 4 will handle this systematically

### HR Metrics Extraction Simplified
- Currently using `_extract_basic_metrics()` fallback
- Generates only 6 basic metrics instead of full 25 metrics
- **Reason**: HRMetricsExtractor still expects old column names
- **Solution Options**:
  1. Update HRMetricsExtractor to use new column names (Phase 3)
  2. Keep basic metrics for now, full extraction in Phase 3
  3. Add adapter in preprocessing script to convert back (not recommended)

### BVP/EDA Writers Status Unknown
- Need to check current column naming
- Need to verify if already using `desc-` prefix
- May already be partially compliant

---

## Migration Strategy for Full Reprocessing

Once Steps 3-5 complete:

### 1. Test Phase (1-3 subjects)
```bash
# Delete test subjects
rm -rf data/derivatives/preprocessing/sub-f01p01/
rm -rf data/derivatives/preprocessing/sub-f02p01/
rm -rf data/derivatives/preprocessing/sub-f03p01/

# Reprocess with new pipeline
poetry run python scripts/batch/run_all_preprocessing.py --subjects f01p01,f02p01,f03p01

# Validate
poetry run python scripts/analysis/generate_quality_report.py
poetry run python scripts/batch/run_all_visualizations.py --subjects f01p01,f02p01,f03p01
```

### 2. Full Reprocessing (51 sessions)
```bash
# Backup current outputs (optional)
mv data/derivatives/preprocessing data/derivatives/preprocessing_backup_$(date +%Y%m%d)

# Full reprocessing
poetry run python scripts/batch/run_all_preprocessing.py --verbose 2>&1 | tee full_reprocessing.log

# Validation
poetry run python scripts/analysis/generate_quality_report.py
poetry run python scripts/batch/run_all_visualizations.py

# Compare with backup (metrics should be identical)
```

**Estimated Time**:
- Test phase: 30 minutes
- Full reprocessing: 2-3 hours
- Validation: 1 hour
- **Total**: ~4 hours

---

## Success Criteria for Phase 2

- [x] Base class created and documented
- [x] HR writer refactored and tested
- [ ] BVP writer refactored and tested
- [ ] EDA writer refactored and tested
- [ ] All three modalities use unified API
- [ ] File naming consistent across modalities
- [ ] Column naming consistent (uppercase with modality prefix)
- [ ] Integration tests pass
- [ ] No regression in data quality
- [ ] Documentation updated

**Current Progress**: 2/5 steps complete (40%)

---

## Next Immediate Action

**Start Step 3: BVP Writer Refactoring**

1. Read `bvp_bids_writer.py` to assess current state
2. Check if already using `desc-` prefix in filenames
3. Verify column naming convention
4. Adapt to inherit from `PhysioBIDSWriter`
5. Update `preprocess_bvp.py` to use new API
6. Test on one subject
7. Document changes

**Command to Start**:
```bash
# Read current BVP writer
cat src/physio/preprocessing/bvp_bids_writer.py | head -200

# Check current BVP output structure
ls -lh data/derivatives/preprocessing/sub-f01p01/ses-01/bvp/
```
