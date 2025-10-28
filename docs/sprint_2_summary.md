# Sprint 2 Summary: BVP Processing Pipeline

**Authors**: Lena Adel, Remy Ramadour

## 🎯 Objectives Completed

Sprint 2 successfully implemented a comprehensive BVP (Blood Volume Pulse) processing pipeline for the TherasyncPipeline project, establishing the foundation for physiological data analysis in family therapy research.

## ✅ Deliverables

### Core Processing Modules

1. **BVP Data Loader** (`src/physio/bvp_loader.py`)
   - BIDS-compliant data loading with TSV/JSON pair handling
   - Moment-based segmentation for therapy session analysis
   - Robust validation and error handling for missing/corrupted data
   - Metadata extraction and preservation

2. **BVP Signal Cleaner** (`src/physio/bvp_cleaner.py`)
   - NeuroKit2 integration with elgendi method for peak detection
   - Templatematch quality assessment for signal reliability
   - Comprehensive preprocessing pipeline with validation
   - Configurable quality thresholds and processing parameters

3. **HRV Metrics Extractor** (`src/physio/bvp_metrics.py`)
   - Essential HRV metrics set (12 carefully selected metrics)
   - Session-level analysis with moment comparison capabilities
   - Placeholder architecture for future epoched analysis (30-second windows)
   - Time-domain, frequency-domain, and nonlinear metrics

4. **BIDS Output Writer** (`src/physio/bvp_bids_writer.py`)
   - BIDS-compliant derivatives structure under `data/derivatives/therasync-bvp/`
   - Comprehensive metadata generation and preservation
   - Dataset descriptions and processing provenance
   - TSV/JSON output pairs with proper BIDS naming conventions

### Pipeline Infrastructure

1. **Main Processing Script** (`scripts/preprocess_bvp.py`)
   - CLI interface with argparse for user-friendly operation
   - Single subject and batch processing modes
   - Comprehensive error handling and progress reporting
   - Flexible configuration and parameter override capabilities

2. **Configuration System** (`config/`)
   - YAML-based configuration with validation
   - Example configuration file for easy setup
   - Essential HRV metrics selection for optimal performance
   - Comprehensive documentation of all parameters

3. **Testing Framework** (`tests/test_bvp_pipeline.py`)
   - Unit tests for all processing components
   - Integration tests for complete pipeline workflow
   - Mock-based testing for external dependencies
   - Edge case and error condition coverage

### Technical Infrastructure

1. **Poetry Package Management**
   - Converted project to proper Poetry structure
   - Dependency management with lock file
   - Development and production dependency separation
   - Virtual environment isolation

2. **Import Structure Fix**
   - Resolved relative import issues for proper package structure
   - Absolute imports from `src` package
   - Proper module initialization and organization

## 📊 HRV Metrics Implemented

### Time-Domain Metrics (5)
- **HRV_MeanNN**: Mean of RR intervals (cardiac rhythm baseline)
- **HRV_SDNN**: Standard deviation of RR intervals (overall HRV)
- **HRV_RMSSD**: Root mean square of successive differences (parasympathetic activity)
- **HRV_pNN50**: Percentage of RR intervals >50ms different (parasympathetic tone)
- **HRV_CVNN**: Coefficient of variation (normalized variability)

### Frequency-Domain Metrics (4)
- **HRV_LF**: Low frequency power (0.04-0.15 Hz, sympathetic + parasympathetic)
- **HRV_HF**: High frequency power (0.15-0.4 Hz, parasympathetic)
- **HRV_LFHF**: LF/HF ratio (autonomic balance indicator)
- **HRV_TP**: Total power (overall autonomic activity)

### Nonlinear Metrics (3)
- **HRV_SD1**: Poincaré plot SD1 (short-term variability)
- **HRV_SD2**: Poincaré plot SD2 (long-term variability)
- **HRV_SampEn**: Sample entropy (signal complexity/irregularity)

## 🏗️ Architecture Highlights

### BIDS Compliance
- Full adherence to Brain Imaging Data Structure standards
- Proper derivatives directory structure
- Comprehensive metadata preservation
- Standardized naming conventions

### Error Handling & Validation
- Input data validation with clear error messages
- Signal quality assessment and filtering
- Graceful failure modes with informative logging
- Edge case handling for insufficient data

### Performance Optimization
- Essential metrics selection for computational efficiency
- Efficient data loading and processing pipelines
- Memory-conscious operations for large datasets
- Configurable quality thresholds

### Future-Ready Design
- Placeholder for epoched analysis (30-second sliding windows)
- Extensible architecture for additional physiological signals
- Modular design for easy feature additions
- Comprehensive type hints for maintainability

## 🚀 Usage Examples

### Single Subject Processing
```bash
poetry run python scripts/preprocess_bvp.py --subject sub-f01p01 --session ses-01
```

### Batch Processing
```bash
poetry run python scripts/preprocess_bvp.py --batch --subject-pattern "sub-f01*"
```

### Custom Configuration
```bash
poetry run python scripts/preprocess_bvp.py --subject sub-f01p01 --session ses-01 --config custom_config.yaml
```

## 📈 Impact & Benefits

1. **Research Ready**: Complete BVP processing pipeline ready for family therapy research
2. **Standardized**: BIDS-compliant outputs for reproducible research
3. **Validated**: Comprehensive testing ensures reliability
4. **Documented**: Clear documentation and examples for easy adoption
5. **Extensible**: Architecture ready for additional physiological signals
6. **Performance**: Optimized for efficiency with essential metrics selection

## 🔄 Next Steps (Sprint 3)

1. **EDA Processing Pipeline**: Implement Electrodermal Activity processing following BVP patterns
2. **Advanced Analytics**: Implement 30-second epoched analysis for dynamic monitoring
3. **Integration Testing**: Cross-signal analysis and synchronization
4. **Performance Optimization**: Further optimization for large-scale batch processing
5. **Documentation**: Complete user guides and API documentation

## 🎉 Sprint 2 Status: ✅ COMPLETED

All 17 major tasks completed successfully, establishing a robust foundation for the TherasyncPipeline project. The BVP processing pipeline is production-ready and follows best practices for scientific computing and reproducible research.