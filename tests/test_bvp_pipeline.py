"""
Unit tests for BVP processing pipeline components.

This module provides comprehensive testing for all BVP processing components
including data loading, cleaning, metrics extraction, and BIDS output.
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import sys

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

import pandas as pd
import numpy as np

from src.core.config_loader import ConfigLoader
from src.physio.bvp_loader import BVPLoader
from src.physio.bvp_cleaner import BVPCleaner
from src.physio.bvp_metrics import BVPMetricsExtractor
from src.physio.bvp_bids_writer import BVPBIDSWriter


class TestBVPLoader(unittest.TestCase):
    """Test BVP data loading functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test data structure
        self.test_subject = "sub-f01p01"
        self.test_session = "ses-01"
        self.test_moment = "restingstate"
        
        # Create test directories
        physio_dir = (self.temp_path / "sourcedata" / self.test_subject / 
                     self.test_session / "physio")
        physio_dir.mkdir(parents=True)
        
        # Create test BVP files
        self._create_test_bvp_files(physio_dir)
        
        # Create test config
        self.test_config = {
            'paths': {'sourcedata': str(self.temp_path / "sourcedata")},
            'moments': [{'name': self.test_moment}]
        }
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_bvp_files(self, physio_dir: Path):
        """Create test BVP data files."""
        # Create test TSV data
        time_values = np.arange(0, 10, 1/64)  # 10 seconds at 64 Hz
        bvp_values = np.sin(2 * np.pi * 1.2 * time_values) + np.random.normal(0, 0.1, len(time_values))
        
        test_data = pd.DataFrame({
            'time': time_values,
            'bvp': bvp_values
        })
        
        # Save TSV file
        base_filename = f"{self.test_subject}_{self.test_session}_task-{self.test_moment}_recording-bvp"
        tsv_file = physio_dir / f"{base_filename}.tsv"
        test_data.to_csv(tsv_file, sep='\t', index=False)
        
        # Save JSON metadata
        json_file = physio_dir / f"{base_filename}.json"
        metadata = {
            "SamplingFrequency": 64.0,
            "StartTime": 0,
            "Columns": ["time", "bvp"],
            "Units": ["s", "AU"],
            "TaskName": self.test_moment,
            "RecordingType": "BVP",
            "FamilyID": "f01"
        }
        
        with open(json_file, 'w') as f:
            json.dump(metadata, f)
    
    @patch('physio.bvp_loader.ConfigLoader')
    def test_loader_initialization(self, mock_config):
        """Test BVP loader initialization."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.bvp': {},
            'moments': []
        }.get(key, default)
        
        loader = BVPLoader()
        self.assertIsInstance(loader, BVPLoader)
        mock_config.assert_called_once()
    
    def test_load_moment_data_success(self):
        """Test successful loading of moment data."""
        with patch('physio.bvp_loader.ConfigLoader') as mock_config:
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                'paths.sourcedata': str(self.temp_path / "sourcedata")
            }.get(key, default)
            
            loader = BVPLoader()
            data, metadata = loader.load_moment_data(
                self.test_subject, self.test_session, self.test_moment
            )
            
            self.assertIsInstance(data, pd.DataFrame)
            self.assertIn('time', data.columns)
            self.assertIn('bvp', data.columns)
            self.assertGreater(len(data), 0)
            
            self.assertIsInstance(metadata, dict)
            self.assertEqual(metadata['SamplingFrequency'], 64.0)
            self.assertEqual(metadata['TaskName'], self.test_moment)
    
    def test_load_moment_data_missing_file(self):
        """Test loading with missing files."""
        with patch('physio.bvp_loader.ConfigLoader') as mock_config:
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                'paths.sourcedata': str(self.temp_path / "sourcedata")
            }.get(key, default)
            
            loader = BVPLoader()
            with self.assertRaises(FileNotFoundError):
                loader.load_moment_data(
                    self.test_subject, self.test_session, "nonexistent"
                )
    
    def test_data_validation(self):
        """Test data validation functionality."""
        with patch('physio.bvp_loader.ConfigLoader') as mock_config:
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                'paths.sourcedata': str(self.temp_path / "sourcedata")
            }.get(key, default)
            
            loader = BVPLoader()
            
            # Test with valid data
            valid_data = pd.DataFrame({
                'time': [0, 1, 2],
                'bvp': [100, 110, 105]
            })
            valid_metadata = {'SamplingFrequency': 64}
            
            # Should not raise exception
            loader._validate_data_structure(valid_data, valid_metadata, Path("test.tsv"))
            
            # Test with invalid data (missing columns)
            invalid_data = pd.DataFrame({'time': [0, 1, 2]})
            
            with self.assertRaises(ValueError):
                loader._validate_data_structure(invalid_data, valid_metadata, Path("test.tsv"))


class TestBVPCleaner(unittest.TestCase):
    """Test BVP signal cleaning functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_config = {
            'physio': {
                'bvp': {
                    'sampling_rate': 64,
                    'processing': {
                        'method': 'elgendi',
                        'method_quality': 'templatematch',
                        'quality_threshold': 0.8
                    }
                }
            }
        }
        
        # Create test signal
        self.test_signal = self._create_test_signal()
    
    def _create_test_signal(self):
        """Create synthetic BVP signal for testing."""
        # Create 30 seconds of synthetic BVP signal at 64 Hz
        duration = 30
        sampling_rate = 64
        t = np.arange(0, duration, 1/sampling_rate)
        
        # Simulate BVP with heart rate around 70 BPM
        heart_rate = 70 / 60  # Hz
        bvp_signal = np.sin(2 * np.pi * heart_rate * t) + 0.1 * np.random.normal(0, 1, len(t))
        
        return bvp_signal
    
    @patch('physio.bvp_cleaner.ConfigLoader')
    @patch('physio.bvp_cleaner.nk')
    def test_cleaner_initialization(self, mock_nk, mock_config):
        """Test BVP cleaner initialization."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.bvp': self.test_config['physio']['bvp'],
            'physio.bvp.processing': self.test_config['physio']['bvp']['processing']
        }.get(key, default)
        
        cleaner = BVPCleaner()
        self.assertEqual(cleaner.method, 'elgendi')
        self.assertEqual(cleaner.method_quality, 'templatematch')
        self.assertEqual(cleaner.quality_threshold, 0.8)
    
    @patch('physio.bvp_cleaner.ConfigLoader')
    @patch('physio.bvp_cleaner.nk.ppg_process')
    def test_process_signal_success(self, mock_ppg_process, mock_config):
        """Test successful signal processing."""
        # Mock config
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.bvp': self.test_config['physio']['bvp'],
            'physio.bvp.processing': self.test_config['physio']['bvp']['processing'],
            'physio.bvp.sampling_rate': 64
        }.get(key, default)
        
        # Mock NeuroKit2 response
        mock_processed_signals = pd.DataFrame({
            'PPG_Clean': self.test_signal,
            'PPG_Rate': np.full(len(self.test_signal), 70)
        })
        mock_processing_info = {
            'PPG_Peaks': [64, 128, 192, 256],  # Mock peak indices
            'sampling_rate': 64
        }
        mock_ppg_process.return_value = (mock_processed_signals, mock_processing_info)
        
        cleaner = BVPCleaner()
        processed_signals, processing_info = cleaner.process_signal(
            self.test_signal, sampling_rate=64, moment="test"
        )
        
        self.assertIsInstance(processed_signals, pd.DataFrame)
        self.assertIn('PPG_Clean', processed_signals.columns)
        self.assertIsInstance(processing_info, dict)
        self.assertIn('PPG_Peaks', processing_info)
        mock_ppg_process.assert_called_once()
    
    @patch('physio.bvp_cleaner.ConfigLoader')
    def test_input_validation(self, mock_config):
        """Test input signal validation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.bvp': self.test_config['physio']['bvp']
        }.get(key, default)
        
        cleaner = BVPCleaner()
        
        # Test empty signal
        with self.assertRaises(ValueError):
            cleaner._validate_input_signal(np.array([]), 64, "test")
        
        # Test invalid sampling rate
        with self.assertRaises(ValueError):
            cleaner._validate_input_signal(self.test_signal, 0, "test")
        
        # Test all NaN signal
        with self.assertRaises(ValueError):
            cleaner._validate_input_signal(np.full(100, np.nan), 64, "test")


class TestBVPMetricsExtractor(unittest.TestCase):
    """Test BVP metrics extraction functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_config = {
            'physio': {
                'bvp': {
                    'metrics': {
                        'selected_metrics': {
                            'time_domain': ['HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD'],
                            'frequency_domain': ['HRV_LF', 'HRV_HF'],
                            'nonlinear': ['HRV_SD1', 'HRV_SD2']
                        }
                    }
                }
            }
        }
        
        # Create mock processed results
        self.mock_processed_results = self._create_mock_processed_results()
    
    def _create_mock_processed_results(self):
        """Create mock processed results for testing."""
        # Create mock processed signals
        processed_signals = pd.DataFrame({
            'PPG_Clean': np.random.normal(0, 1, 1920),  # 30 seconds at 64 Hz
            'PPG_Rate': np.full(1920, 70)
        })
        
        # Create mock processing info with peaks
        processing_info = {
            'PPG_Peaks': [64, 128, 192, 256, 320, 384, 448, 512, 576, 640],  # 10 peaks
            'sampling_rate': 64,
            'moment': 'test'
        }
        
        return {
            'restingstate': (processed_signals, processing_info),
            'therapy': (processed_signals, processing_info)
        }
    
    @patch('physio.bvp_metrics.ConfigLoader')
    def test_extractor_initialization(self, mock_config):
        """Test metrics extractor initialization."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.bvp': self.test_config['physio']['bvp'],
            'physio.bvp.metrics': self.test_config['physio']['bvp']['metrics'],
            'physio.bvp.metrics.selected_metrics': self.test_config['physio']['bvp']['metrics']['selected_metrics']
        }.get(key, default)
        
        extractor = BVPMetricsExtractor()
        self.assertEqual(len(extractor.time_domain_metrics), 3)
        self.assertEqual(len(extractor.frequency_domain_metrics), 2)
        self.assertEqual(len(extractor.nonlinear_metrics), 2)
    
    @patch('physio.bvp_metrics.ConfigLoader')
    @patch('physio.bvp_metrics.nk.hrv_time')
    @patch('physio.bvp_metrics.nk.hrv_frequency')
    @patch('physio.bvp_metrics.nk.hrv_nonlinear')
    def test_extract_hrv_metrics_success(self, mock_nonlinear, mock_frequency, 
                                       mock_time, mock_config):
        """Test successful HRV metrics extraction."""
        # Mock config
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.bvp.metrics': self.test_config['physio']['bvp']['metrics'],
            'physio.bvp.metrics.selected_metrics': self.test_config['physio']['bvp']['metrics']['selected_metrics']
        }.get(key, default)
        
        # Mock NeuroKit2 HRV functions
        mock_time.return_value = pd.DataFrame({
            'HRV_MeanNN': [800],
            'HRV_SDNN': [50],
            'HRV_RMSSD': [30]
        })
        
        mock_frequency.return_value = pd.DataFrame({
            'HRV_LF': [1200],
            'HRV_HF': [800]
        })
        
        mock_nonlinear.return_value = pd.DataFrame({
            'HRV_SD1': [21.2],
            'HRV_SD2': [70.7]
        })
        
        extractor = BVPMetricsExtractor()
        peaks = np.array([64, 128, 192, 256, 320])
        
        metrics = extractor._extract_hrv_metrics(peaks, 64, "test")
        
        self.assertIn('HRV_MeanNN', metrics)
        self.assertIn('HRV_LF', metrics)
        self.assertIn('HRV_SD1', metrics)
        self.assertEqual(metrics['HRV_MeanNN'], 800)
    
    @patch('physio.bvp_metrics.ConfigLoader')
    def test_peaks_validation(self, mock_config):
        """Test peak validation for HRV analysis."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.bvp.metrics': {}
        }.get(key, default)
        
        extractor = BVPMetricsExtractor()
        
        # Test insufficient peaks
        few_peaks = [64, 128]
        self.assertFalse(extractor._validate_peaks_for_hrv(few_peaks, "test"))
        
        # Test sufficient peaks
        enough_peaks = list(range(64, 640, 64))  # 10 peaks
        self.assertTrue(extractor._validate_peaks_for_hrv(enough_peaks, "test"))
    
    @patch('physio.bvp_metrics.ConfigLoader')
    def test_session_metrics_extraction(self, mock_config):
        """Test extraction of session-level metrics."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.bvp.metrics': self.test_config['physio']['bvp']['metrics'],
            'physio.bvp.metrics.selected_metrics': {}
        }.get(key, default)
        
        extractor = BVPMetricsExtractor()
        
        with patch.object(extractor, '_extract_hrv_metrics') as mock_extract:
            mock_extract.return_value = {'HRV_MeanNN': 800, 'HRV_SDNN': 50}
            
            session_metrics = extractor.extract_session_metrics(self.mock_processed_results)
            
            self.assertIn('restingstate', session_metrics)
            self.assertIn('therapy', session_metrics)
            self.assertEqual(mock_extract.call_count, 2)


class TestBVPBIDSWriter(unittest.TestCase):
    """Test BIDS output formatting functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.test_config = {
            'paths': {'derivatives': str(self.temp_path / "derivatives")},
            'bids': {}
        }
        
        # Create test data
        self.test_subject = "sub-f01p01"
        self.test_session = "ses-01"
        self.mock_processed_results = self._create_mock_data()
        self.mock_session_metrics = {
            'restingstate': {'HRV_MeanNN': 800.0, 'HRV_SDNN': 50.0},
            'therapy': {'HRV_MeanNN': 750.0, 'HRV_SDNN': 45.0}
        }
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_mock_data(self):
        """Create mock processed data for testing."""
        processed_signals = pd.DataFrame({
            'time': np.arange(0, 10, 1/64),
            'PPG_Clean': np.random.normal(0, 1, 640),
            'PPG_Rate': np.full(640, 70)
        })
        
        processing_info = {
            'PPG_Peaks': [64, 128, 192, 256],
            'sampling_rate': 64,
            'processing_method': 'elgendi',
            'quality_method': 'templatematch'
        }
        
        return {
            'restingstate': (processed_signals, processing_info)
        }
    
    @patch('physio.bvp_bids_writer.ConfigLoader')
    def test_writer_initialization(self, mock_config):
        """Test BIDS writer initialization."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'paths.derivatives': str(self.temp_path / "derivatives"),
            'bids': {}
        }.get(key, default)
        
        writer = BVPBIDSWriter()
        
        # Check if pipeline directory was created
        pipeline_dir = self.temp_path / "derivatives" / "therasync-bvp"
        self.assertTrue(pipeline_dir.exists())
        
        # Check if dataset description was created
        dataset_desc = pipeline_dir / "dataset_description.json"
        self.assertTrue(dataset_desc.exists())
    
    @patch('physio.bvp_bids_writer.ConfigLoader')
    def test_save_processed_data(self, mock_config):
        """Test saving processed data in BIDS format."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'paths.derivatives': str(self.temp_path / "derivatives"),
            'bids': {}
        }.get(key, default)
        
        writer = BVPBIDSWriter()
        
        created_files = writer.save_processed_data(
            self.test_subject,
            self.test_session,
            self.mock_processed_results,
            self.mock_session_metrics
        )
        
        # Check that files were created
        self.assertIn('processed_signals', created_files)
        self.assertIn('metrics', created_files)
        self.assertIn('metadata', created_files)
        self.assertIn('summary', created_files)
        
        # Check total file count
        total_files = sum(len(files) for files in created_files.values())
        self.assertGreater(total_files, 0)
        
        # Verify actual files exist
        for file_list in created_files.values():
            for file_path in file_list:
                self.assertTrue(Path(file_path).exists())
    
    @patch('physio.bvp_bids_writer.ConfigLoader')
    def test_bids_filename_format(self, mock_config):
        """Test BIDS-compliant filename formatting."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'paths.derivatives': str(self.temp_path / "derivatives"),
            'bids': {}
        }.get(key, default)
        
        writer = BVPBIDSWriter()
        
        created_files = writer.save_processed_data(
            self.test_subject,
            self.test_session,
            self.mock_processed_results,
            self.mock_session_metrics
        )
        
        # Check BIDS filename patterns
        for file_path in created_files['processed_signals']:
            filename = Path(file_path).name
            self.assertTrue(filename.startswith(f"{self.test_subject}_{self.test_session}"))
            self.assertIn('task-restingstate', filename)
            self.assertIn('recording-bvp', filename)
    
    def test_json_serialization(self):
        """Test JSON serialization of numpy types."""
        with patch('physio.bvp_bids_writer.ConfigLoader'):
            writer = BVPBIDSWriter()
            
            # Test serialization of numpy types
            test_obj = np.float64(123.45)
            result = writer._json_serializer(test_obj)
            self.assertIsInstance(result, float)
            
            test_array = np.array([1, 2, 3])
            result = writer._json_serializer(test_array)
            self.assertIsInstance(result, list)


class TestBVPPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete BVP pipeline."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create complete test data structure
        self.test_subject = "sub-f01p01"
        self.test_session = "ses-01"
        self.test_moment = "restingstate"
        
        self._setup_test_data_structure()
    
    def tearDown(self):
        """Clean up integration test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _setup_test_data_structure(self):
        """Set up complete test data structure."""
        # Create directories
        sourcedata_dir = self.temp_path / "sourcedata"
        derivatives_dir = self.temp_path / "derivatives"
        
        physio_dir = (sourcedata_dir / self.test_subject / 
                     self.test_session / "physio")
        physio_dir.mkdir(parents=True)
        
        # Create realistic test BVP data
        duration = 60  # 1 minute
        sampling_rate = 64
        time_values = np.arange(0, duration, 1/sampling_rate)
        
        # Simulate realistic BVP signal with heart rate ~70 BPM
        heart_rate = 70 / 60  # Hz
        bvp_signal = (
            100 * np.sin(2 * np.pi * heart_rate * time_values) +
            20 * np.sin(2 * np.pi * 2 * heart_rate * time_values) +
            10 * np.random.normal(0, 1, len(time_values))
        )
        
        test_data = pd.DataFrame({
            'time': time_values,
            'bvp': bvp_signal
        })
        
        # Save test files
        base_filename = f"{self.test_subject}_{self.test_session}_task-{self.test_moment}_recording-bvp"
        tsv_file = physio_dir / f"{base_filename}.tsv"
        test_data.to_csv(tsv_file, sep='\t', index=False)
        
        json_file = physio_dir / f"{base_filename}.json"
        metadata = {
            "SamplingFrequency": 64.0,
            "StartTime": 0,
            "Columns": ["time", "bvp"],
            "Units": ["s", "AU"],
            "TaskName": self.test_moment,
            "RecordingType": "BVP",
            "FamilyID": "f01"
        }
        
        with open(json_file, 'w') as f:
            json.dump(metadata, f)
        
        # Create test config
        self.test_config_path = self.temp_path / "config.yaml"
        config_content = f"""
paths:
  sourcedata: {sourcedata_dir}
  derivatives: {derivatives_dir}

moments:
  - name: {self.test_moment}

physio:
  bvp:
    sampling_rate: 64
    processing:
      method: elgendi
      method_quality: templatematch
      quality_threshold: 0.8
    metrics:
      selected_metrics:
        time_domain:
          - HRV_MeanNN
          - HRV_SDNN
        frequency_domain:
          - HRV_LF
          - HRV_HF
        nonlinear:
          - HRV_SD1
"""
        
        with open(self.test_config_path, 'w') as f:
            f.write(config_content)
    
    @patch('physio.bvp_cleaner.nk.ppg_process')
    @patch('physio.bvp_metrics.nk.hrv_time')
    @patch('physio.bvp_metrics.nk.hrv_frequency')
    @patch('physio.bvp_metrics.nk.hrv_nonlinear')
    def test_complete_pipeline(self, mock_nonlinear, mock_frequency, 
                              mock_time, mock_ppg_process):
        """Test the complete BVP pipeline end-to-end."""
        # Mock NeuroKit2 functions
        mock_processed_signals = pd.DataFrame({
            'PPG_Clean': np.random.normal(0, 1, 3840),  # 60 seconds at 64 Hz
            'PPG_Rate': np.full(3840, 70)
        })
        mock_processing_info = {
            'PPG_Peaks': list(range(64, 3840, 64)),  # ~60 peaks for 60 seconds
            'sampling_rate': 64
        }
        mock_ppg_process.return_value = (mock_processed_signals, mock_processing_info)
        
        mock_time.return_value = pd.DataFrame({
            'HRV_MeanNN': [857.1],
            'HRV_SDNN': [41.5]
        })
        
        mock_frequency.return_value = pd.DataFrame({
            'HRV_LF': [1127.4],
            'HRV_HF': [975.3]
        })
        
        mock_nonlinear.return_value = pd.DataFrame({
            'HRV_SD1': [29.4]
        })
        
        # Run complete pipeline
        loader = BVPLoader(self.test_config_path)
        cleaner = BVPCleaner(self.test_config_path)
        metrics_extractor = BVPMetricsExtractor(self.test_config_path)
        bids_writer = BVPBIDSWriter(self.test_config_path)
        
        # Step 1: Load data
        loaded_data = loader.load_subject_session_data(
            self.test_subject, self.test_session
        )
        self.assertIn(self.test_moment, loaded_data)
        
        # Step 2: Process signals
        processed_results = cleaner.process_moment_signals(loaded_data)  # type: ignore
        self.assertIn(self.test_moment, processed_results)
        
        # Step 3: Extract metrics
        session_metrics = metrics_extractor.extract_session_metrics(processed_results)
        self.assertIn(self.test_moment, session_metrics)
        self.assertIn('HRV_MeanNN', session_metrics[self.test_moment])
        
        # Step 4: Save BIDS output
        created_files = bids_writer.save_processed_data(
            self.test_subject, self.test_session,
            processed_results, session_metrics
        )
        
        # Verify output files were created
        total_files = sum(len(files) for files in created_files.values())
        self.assertGreater(total_files, 5)  # Expect multiple output files
        
        # Verify file contents
        derivatives_dir = self.temp_path / "derivatives" / "therasync-bvp"
        subject_dir = derivatives_dir / self.test_subject / self.test_session / "physio"
        
        # Check if metrics file exists and has correct content
        metrics_file = subject_dir / f"{self.test_subject}_{self.test_session}_desc-bvpmetrics_physio.tsv"
        self.assertTrue(metrics_file.exists())
        
        metrics_df = pd.read_csv(metrics_file, sep='\\t')
        self.assertIn('moment', metrics_df.columns)
        self.assertIn('HRV_MeanNN', metrics_df.columns)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)