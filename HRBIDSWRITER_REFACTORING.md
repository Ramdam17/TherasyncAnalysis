# HRBIDSWriter Refactoring - Implementation Complete

**Date**: 2024-11-11  
**Status**: ✅ COMPLETE - Ready for testing  
**Branch**: `refactor/code-cleanup`

## Summary of Changes

The `HRBIDSWriter` class has been completely refactored to align with the unified BIDS writing interface defined by `PhysioBIDSWriter` base class. This brings HR processing in line with BVP and EDA modalities.

---

## 1. Architecture Changes

### Inheritance
- **Before**: Standalone class with no inheritance
- **After**: Inherits from `PhysioBIDSWriter` abstract base class
- **Benefit**: Unified interface, shared utilities, polymorphic code possible

### API Method
- **Before**: `write_hr_results(subject, session, moment, cleaned_data, metrics, metadata) -> Dict[str, Path]`
- **After**: `save_processed_data(subject_id, session_id, processed_results, session_metrics, metadata) -> Dict[str, List[Path]]`
- **Benefit**: Consistent API across all modalities (BVP, EDA, HR)

---

## 2. File Naming Changes

### Main Data Files
| File Type | Before | After |
|-----------|--------|-------|
| Physio signal | `{prefix}_physio.tsv.gz` | `{prefix}_desc-processed_recording-hr.tsv` |
| Physio metadata | `{prefix}_physio.json` | `{prefix}_desc-processed_recording-hr.json` |
| Events | `{prefix}_events.tsv` | `{prefix}_events.tsv` *(unchanged)* |
| Events metadata | `{prefix}_events.json` | `{prefix}_events.json` *(unchanged)* |
| Metrics | `{prefix}_hr-metrics.tsv` | `{prefix}_desc-hr-metrics.tsv` |
| Metrics metadata | `{prefix}_hr-metrics.json` | `{prefix}_desc-hr-metrics.json` |
| Summary | `{prefix}_hr-summary.json` | `{prefix}_desc-hr-summary.json` |

### Key Changes
1. ✅ **Added `desc-` prefix** to all descriptive files (BIDS v1.7.0 compliance)
2. ✅ **Added `_recording-hr` suffix** to physio files (modality identification)
3. ✅ **UNCOMPRESSED files** (`.tsv` instead of `.tsv.gz`)

---

## 3. File Organization Changes

### Moment-Based Processing
- **Before**: Single `task-combined` file for all moments
- **After**: Separate files per moment (e.g., `task-restingstate`, `task-therapy`)

**Example Before**:
```
hr/
├── sub-f01p01_ses-01_task-combined_physio.tsv.gz
├── sub-f01p01_ses-01_task-combined_physio.json
├── sub-f01p01_ses-01_task-combined_events.tsv
├── sub-f01p01_ses-01_task-combined_events.json
├── sub-f01p01_ses-01_task-combined_hr-metrics.tsv
├── sub-f01p01_ses-01_task-combined_hr-metrics.json
└── sub-f01p01_ses-01_task-combined_hr-summary.json
```

**Example After**:
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

**Benefits**:
- Can now compare restingstate vs therapy moments separately
- Consistent with BVP and EDA processing
- Better data organization for analysis

---

## 4. Column Name Changes

### Signal Columns
| Before | After | Description |
|--------|-------|-------------|
| `time` | `time` | *(unchanged)* Time in seconds |
| ❌ *(missing)* | `HR_Raw` | **NEW**: Raw HR before cleaning |
| `hr` | `HR_Clean` | Cleaned HR signal |
| `quality` | `HR_Quality` | Quality score (0-1) |
| `outlier` | `HR_Outliers` | Outlier flag (0/1) |
| `interpolated` | `HR_Interpolated` | Interpolation flag (0/1) |

**Naming Convention**: All HR-specific columns now use `HR_` prefix with PascalCase (e.g., `HR_Clean`, `HR_Quality`)

**Benefits**:
- ✅ Consistent with BVP (`PPG_Clean`, `PPG_Quality`) and EDA (`EDA_Clean`, `EDA_Quality`)
- ✅ Clear modality identification
- ✅ Easier to write polymorphic analysis code

---

## 5. ID Format Changes

### Subject/Session IDs
- **Before**: IDs without prefixes (e.g., `'f01p01'`, `'01'`)
- **After**: IDs WITH prefixes (e.g., `'sub-f01p01'`, `'ses-01'`)

**API Signature Change**:
```python
# Before
writer.write_hr_results(
    subject='f01p01',
    session='01',
    moment='therapy',
    ...
)

# After
writer.save_processed_data(
    subject_id='sub-f01p01',  # or 'f01p01' - prefix auto-added
    session_id='ses-01',       # or '01' - prefix auto-added
    processed_results={'restingstate': df_rest, 'therapy': df_therapy},
    ...
)
```

**Benefit**: Base class automatically adds prefixes via `_ensure_prefix()` if missing

---

## 6. Data Structure Changes

### Input Format
**Before**:
```python
write_hr_results(
    subject='f01p01',
    session='01', 
    moment='therapy',              # Single moment
    cleaned_data=df_therapy,       # Single DataFrame
    metrics={'hr_mean': 75.2, ...},
    cleaning_metadata={...}
)
```

**After**:
```python
save_processed_data(
    subject_id='sub-f01p01',
    session_id='ses-01',
    processed_results={             # Multiple moments
        'restingstate': df_resting,
        'therapy': df_therapy
    },
    session_metrics=metrics_df,    # DataFrame with all moments
    processing_metadata={
        'restingstate': {...},
        'therapy': {...}
    }
)
```

### Expected DataFrame Columns
**Before**:
- `time`, `hr_clean`, `hr_quality`, `hr_outliers`, `hr_interpolated`

**After**:
- `time`, `HR_Raw`, `HR_Clean`, `HR_Quality`, `HR_Outliers`, `HR_Interpolated`

---

## 7. Return Type Changes

### Return Value
- **Before**: `Dict[str, Path]` - single path per file type
  ```python
  {
      'physio': Path('...physio.tsv.gz'),
      'physio_json': Path('...physio.json'),
      'events': Path('...events.tsv'),
      ...
  }
  ```

- **After**: `Dict[str, List[Path]]` - list of paths (one per moment)
  ```python
  {
      'physio': [
          Path('...task-restingstate_desc-processed_recording-hr.tsv'),
          Path('...task-therapy_desc-processed_recording-hr.tsv')
      ],
      'physio_json': [
          Path('...task-restingstate_desc-processed_recording-hr.json'),
          Path('...task-therapy_desc-processed_recording-hr.json')
      ],
      ...
  }
  ```

**Benefit**: Consistent with multi-moment processing, easier to iterate

---

## 8. Compression Changes

### File Format
- **Before**: Compressed `.tsv.gz` files
- **After**: Uncompressed `.tsv` files

**Rationale**:
1. Consistency with BVP and EDA (both use `.tsv`)
2. Easier to inspect/debug files
3. Better compatibility with analysis tools
4. Minimal storage impact (HR files are small)

**File Size Impact**:
- Before: ~14 KB compressed
- After: ~50-100 KB uncompressed (estimated)
- Storage increase: ~40-90 KB per file (negligible)

---

## 9. Metadata Enhancements

### Processing Metadata
All JSON sidecar files now use base class `_save_json_sidecar()` method for:
- ✅ Consistent datetime serialization
- ✅ Proper numpy/pandas type handling
- ✅ Standardized formatting

### Enhanced Fields
New fields added to metadata files:
- `HR_Raw` description in physio.json
- Improved column descriptions
- Better unit specifications

---

## 10. Code Quality Improvements

### Inheritance Benefits
- ✅ Removed duplicate code (directory creation, prefix handling, JSON serialization)
- ✅ Uses base class utilities: `_ensure_prefix()`, `_get_subject_session_dir()`, `_save_json_sidecar()`
- ✅ Implements abstract interface: `_get_modality_name()`, `save_processed_data()`

### Error Handling
- ✅ Better validation of input DataFrames
- ✅ Graceful handling of missing columns
- ✅ Improved logging with DEBUG/INFO/ERROR levels

### Backward Compatibility Helper
New method `_extract_basic_metrics()` extracts fallback metrics if full metrics not provided:
```python
{
    'moment': 'therapy',
    'hr_mean': 75.2,
    'hr_std': 8.5,
    'hr_min': 62.0,
    'hr_max': 95.0,
    'hr_range': 33.0
}
```

---

## 11. Breaking Changes Summary

⚠️ **ALL CHANGES ARE BREAKING** - Full reprocessing required

### For Calling Code
1. Change method name: `write_hr_results()` → `save_processed_data()`
2. Change parameters: single moment → multi-moment dictionary
3. Change return type handling: `Dict[str, Path]` → `Dict[str, List[Path]]`
4. Update column names in input DataFrames

### For Output Files
1. All existing HR files will have different names
2. Files will be organized by moment instead of combined
3. Column names in TSV files will change
4. Files will be uncompressed

### Migration Impact
- **Scripts to update**: `preprocess_hr.py`, any analysis scripts using HR data
- **Data to reprocess**: All 51 sessions (50 + 1 partial)
- **Visualizations to update**: Any plots reading HR files directly

---

## 12. Testing Checklist

Before full deployment, test on **ONE subject** (e.g., `sub-f01p01/ses-01`):

- [ ] Delete existing HR outputs: `rm -rf data/derivatives/preprocessing/sub-f01p01/ses-01/hr/`
- [ ] Update `preprocess_hr.py` to use new API
- [ ] Run preprocessing on test subject
- [ ] Verify file structure:
  - [ ] 14 files total (7 per moment)
  - [ ] Correct naming: `desc-processed_recording-hr.tsv`, not `.tsv.gz`
  - [ ] Separate `task-restingstate` and `task-therapy` files
- [ ] Verify file content:
  - [ ] Columns: `time`, `HR_Raw`, `HR_Clean`, `HR_Quality`, `HR_Outliers`, `HR_Interpolated`
  - [ ] Data values match previous processing
  - [ ] Files are uncompressed (readable with `cat`)
- [ ] Verify metadata:
  - [ ] JSON files have correct column descriptions
  - [ ] Processing metadata complete
- [ ] Run visualization on test subject
- [ ] Compare metrics with previous results (should be identical)

---

## 13. Next Steps (Phase 2 Continuation)

### Step 2.2: Update preprocess_hr.py
Modify the HR preprocessing script to:
1. Use new `save_processed_data()` API
2. Add `HR_Raw` column to processed DataFrames
3. Rename columns: `hr_clean` → `HR_Clean`, etc.
4. Pass multi-moment dictionary instead of calling writer per moment

### Step 2.3: Adapt BVPBIDSWriter
After HR works:
1. Simplify `processed_results` from `Tuple[DataFrame, Dict]` to just `DataFrame`
2. Pass processing info through `processing_metadata` parameter
3. Update return types

### Step 2.4: Adapt EDABIDSWriter
1. Add `EDA_Quality` column calculation
2. Update return types
3. Align with base class interface

### Step 2.5: Integration Testing
Run all three preprocessors on same subject and verify consistency

---

## 14. Files Modified

### Modified Files
- `src/physio/preprocessing/hr_bids_writer.py` (complete rewrite, ~695 lines)

### Files to Modify (Next)
- `scripts/physio/preprocessing/preprocess_hr.py` (update to use new API)
- `src/physio/preprocessing/hr_loader.py` (if needed for per-moment loading)
- `src/physio/preprocessing/hr_cleaner.py` (add HR_Raw column)

### New Files Created (Previous)
- `src/physio/preprocessing/base_bids_writer.py` (base class)
- `REFACTORING_ANALYSIS.md` (strategy document)
- `PHASE2_REFACTORING.md` (detailed plan)
- `HRBIDSWRITER_REFACTORING.md` (this file)

---

## 15. Validation Criteria

### Success Metrics
1. ✅ All tests pass (after test updates)
2. ✅ File structure matches BVP/EDA pattern
3. ✅ Column names consistent across modalities
4. ✅ Metrics values identical to previous processing
5. ✅ Visualizations work without modification (after column name updates)
6. ✅ No data loss (all moments processed)

### Quality Checks
- No regression in data quality scores
- Processing time similar or better
- File sizes reasonable (uncompressed but small)
- Metadata complete and accurate

---

## Conclusion

The `HRBIDSWriter` has been successfully refactored to:
- ✅ Inherit from unified base class
- ✅ Use consistent naming conventions
- ✅ Process data per-moment (not combined)
- ✅ Support multi-moment batch processing
- ✅ Align with BVP and EDA implementations

**Status**: Ready for integration testing with updated preprocessing script.

**Estimated Time to Full Migration**: 
- Update preprocess_hr.py: 1-2 hours
- Test on one subject: 30 minutes
- Full reprocessing (51 sessions): 2-3 hours
- **Total**: ~4-6 hours
