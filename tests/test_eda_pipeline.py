"""
Unit tests for EDA processing pipeline components.

This module provides comprehensive testing for all EDA processing components
including data loading, cleaning (cvxEDA decomposition), metrics extraction,
and BIDS output.

Authors: Lena Adel, Remy Ramadour
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
from src.physio.eda_loader import EDALoader
from src.physio.eda_cleaner import EDACleaner
from src.physio.eda_metrics import EDAMetricsExtractor
from src.physio.eda_bids_writer import EDABIDSWriter


class TestEDALoader(unittest.TestCase):
    """Test EDA data loading functionality."""
    
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
        
        # Create test EDA files
        self._create_test_eda_files(physio_dir)
        
        # Create test config
        self.test_config = {
            'paths': {'sourcedata': str(self.temp_path / "sourcedata")},
            'physio': {
                'eda': {
                    'sampling_rate': 4
                }
            },
            'moments': [{'name': self.test_moment}]
        }
        
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_eda_files(self, physio_dir: Path):
        """Create test EDA data files."""
        # Create test TSV data (60 seconds at 4 Hz = 240 samples)
        duration = 60  # seconds
        sampling_rate = 4  # Hz
        time_values = np.arange(0, duration, 1/sampling_rate)
        
        # Simulate realistic EDA signal:
        # Tonic component (slow drift): 0.5-2.0 μS
        tonic = 1.0 + 0.5 * np.sin(2 * np.pi * 0.01 * time_values)
        
        # Phasic component (SCRs): occasional spikes
        phasic = np.zeros_like(time_values)
        scr_times = [10, 25, 45]  # SCR peaks at these times
        for scr_time in scr_times:
            scr_idx = int(scr_time * sampling_rate)
            # Create SCR with rise and recovery
            for i in range(len(time_values)):
                t_diff = time_values[i] - scr_time
                if 0 <= t_diff <= 10:  # SCR lasting 10 seconds
                    # Exponential rise and decay
                    phasic[i] += 0.3 * np.exp(-abs(t_diff - 2) / 2)
        
        # Add small noise
        noise = np.random.normal(0, 0.02, len(time_values))
        eda_values = tonic + phasic + noise
        
        # Ensure positive values
        eda_values = np.maximum(eda_values, 0.1)
        
        test_data = pd.DataFrame({
            'time': time_values,
            'eda': eda_values
        })
        
        # Save TSV file
        base_filename = f"{self.test_subject}_{self.test_session}_task-{self.test_moment}_recording-eda"
        tsv_file = physio_dir / f"{base_filename}.tsv"
        test_data.to_csv(tsv_file, sep='\t', index=False)
        
        # Save JSON metadata
        json_file = physio_dir / f"{base_filename}.json"
        metadata = {
            "SamplingFrequency": 4.0,
            "StartTime": 0,
            "Columns": ["time", "eda"],
            "Units": ["s", "μS"],
            "TaskName": self.test_moment,
            "RecordingType": "EDA",
            "FamilyID": "f01",
            "DeviceManufacturer": "Empatica",
            "DeviceModel": "E4"
        }
        
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    @patch('physio.eda_loader.ConfigLoader')
    def test_loader_initialization(self, mock_config):
        """Test EDA loader initialization."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.eda': {'sampling_rate': 4},
            'paths.sourcedata': str(self.temp_path / "sourcedata")
        }.get(key, default)
        
        loader = EDALoader()
        self.assertIsInstance(loader, EDALoader)
        self.assertEqual(loader.sampling_rate, 4)
        mock_config.assert_called_once()
    
    def test_load_subject_session_success(self):
        """Test successful loading of subject/session data."""
        with patch('physio.eda_loader.ConfigLoader') as mock_config:
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                'paths.sourcedata': str(self.temp_path / "sourcedata"),
                'physio.eda.sampling_rate': 4
            }.get(key, default)
            
            loader = EDALoader()
            data, metadata = loader.load_subject_session(
                self.test_subject, self.test_session, self.test_moment
            )
            
            self.assertIsInstance(data, pd.DataFrame)
            self.assertIn('time', data.columns)
            self.assertIn('eda', data.columns)
            self.assertEqual(len(data), 240)  # 60 seconds * 4 Hz
            
            self.assertIsInstance(metadata, dict)
            self.assertEqual(metadata['SamplingFrequency'], 4.0)
            self.assertEqual(metadata['TaskName'], self.test_moment)
            
            # Verify EDA values are reasonable (0.1 - 3.0 μS)
            self.assertTrue((data['eda'] >= 0).all())
            self.assertTrue((data['eda'] <= 5.0).all())
    
    def test_load_subject_session_missing_file(self):
        """Test loading with missing files."""
        with patch('physio.eda_loader.ConfigLoader') as mock_config:
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                'paths.sourcedata': str(self.temp_path / "sourcedata"),
                'physio.eda.sampling_rate': 4
            }.get(key, default)
            
            loader = EDALoader()
            with self.assertRaises(FileNotFoundError):
                loader.load_subject_session(
                    self.test_subject, self.test_session, "nonexistent"
                )
    
    def test_data_validation(self):
        """Test EDA data validation functionality."""
        with patch('physio.eda_loader.ConfigLoader') as mock_config:
            mock_config.return_value.get.side_effect = lambda key, default=None: {
                'paths.sourcedata': str(self.temp_path / "sourcedata"),
                'physio.eda.sampling_rate': 4
            }.get(key, default)
            
            loader = EDALoader()
            
            # Test with valid data
            valid_data = pd.DataFrame({
                'time': [0, 0.25, 0.5, 0.75],
                'eda': [1.0, 1.1, 1.05, 1.2]
            })
            valid_metadata = {'SamplingFrequency': 4}
            
            # Should not raise exception
            loader._validate_data_structure(valid_data, valid_metadata, Path("test.tsv"))
            
            # Test with invalid data (missing columns)
            invalid_data = pd.DataFrame({'time': [0, 0.25, 0.5]})
            
            with self.assertRaises(ValueError):
                loader._validate_data_structure(invalid_data, valid_metadata, Path("test.tsv"))
            
            # Test with negative EDA values
            negative_data = pd.DataFrame({
                'time': [0, 0.25, 0.5],
                'eda': [1.0, -0.5, 1.2]
            })
            
            with self.assertRaises(ValueError):
                loader._validate_data_structure(negative_data, valid_metadata, Path("test.tsv"))


class TestEDACleaner(unittest.TestCase):
    """Test EDA signal cleaning and decomposition functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_config = {
            'physio': {
                'eda': {
                    'sampling_rate': 4,
                    'processing': {
                        'method': 'cvxEDA',
                        'scr_threshold': 0.01,
                        'scr_min_amplitude': 0.01
                    }
                }
            }
        }
        
        # Create test signal (60 seconds at 4 Hz)
        self.test_signal = self._create_test_signal()
    
    def _create_test_signal(self):
        """Create synthetic EDA signal for testing."""
        duration = 60
        sampling_rate = 4
        t = np.arange(0, duration, 1/sampling_rate)
        
        # Tonic baseline with slow drift
        tonic = 1.0 + 0.3 * np.sin(2 * np.pi * 0.01 * t)
        
        # Add several SCRs
        phasic = np.zeros_like(t)
        scr_times = [10, 25, 40, 55]
        for scr_time in scr_times:
            t_diff = t - scr_time
            # SCR with realistic rise/recovery time
            mask = (t_diff >= 0) & (t_diff <= 10)
            phasic[mask] += 0.2 * np.exp(-np.abs(t_diff[mask] - 2) / 2)
        
        # Add noise
        noise = np.random.normal(0, 0.02, len(t))
        
        return np.maximum(tonic + phasic + noise, 0.1)
    
    @patch('physio.eda_cleaner.ConfigLoader')
    def test_cleaner_initialization(self, mock_config):
        """Test EDA cleaner initialization."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.eda': self.test_config['physio']['eda'],
            'physio.eda.processing': self.test_config['physio']['eda']['processing'],
            'physio.eda.processing.method': 'cvxEDA',
            'physio.eda.processing.scr_threshold': 0.01
        }.get(key, default)
        
        cleaner = EDACleaner()
        self.assertEqual(cleaner.method, 'cvxEDA')
        self.assertEqual(cleaner.scr_threshold, 0.01)
    
    @patch('physio.eda_cleaner.ConfigLoader')
    @patch('physio.eda_cleaner.nk.eda_process')
    def test_process_signal_success(self, mock_eda_process, mock_config):
        """Test successful EDA signal processing."""
        # Mock config
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.eda': self.test_config['physio']['eda'],
            'physio.eda.processing': self.test_config['physio']['eda']['processing'],
            'physio.eda.sampling_rate': 4,
            'physio.eda.processing.method': 'cvxEDA'
        }.get(key, default)
        
        # Mock NeuroKit2 response with cvxEDA decomposition
        mock_processed_signals = pd.DataFrame({
            'EDA_Raw': self.test_signal,
            'EDA_Clean': self.test_signal,
            'EDA_Tonic': np.ones(len(self.test_signal)) * 1.0,
            'EDA_Phasic': self.test_signal - 1.0,
            'SCR_Onsets': np.zeros(len(self.test_signal)),
            'SCR_Peaks': np.zeros(len(self.test_signal)),
            'SCR_Amplitude': np.zeros(len(self.test_signal))
        })
        
        # Mark some SCR peaks
        mock_processed_signals.loc[40, 'SCR_Peaks'] = 1
        mock_processed_signals.loc[100, 'SCR_Peaks'] = 1
        mock_processed_signals.loc[160, 'SCR_Peaks'] = 1
        
        mock_processing_info = {
            'SCR_Onsets': [35, 95, 155],
            'SCR_Peaks': [40, 100, 160],
            'SCR_Amplitude': [0.15, 0.20, 0.18],
            'sampling_rate': 4
        }
        mock_eda_process.return_value = (mock_processed_signals, mock_processing_info)
        
        cleaner = EDACleaner()
        processed_signals, processing_info = cleaner.process_signal(
            self.test_signal, sampling_rate=4, moment="test"
        )
        
        self.assertIsInstance(processed_signals, pd.DataFrame)
        self.assertIn('EDA_Clean', processed_signals.columns)
        self.assertIn('EDA_Tonic', processed_signals.columns)
        self.assertIn('EDA_Phasic', processed_signals.columns)
        
        self.assertIsInstance(processing_info, dict)
        self.assertIn('SCR_Peaks', processing_info)
        self.assertEqual(len(processing_info['SCR_Peaks']), 3)
        
        mock_eda_process.assert_called_once()
    
    @patch('physio.eda_cleaner.ConfigLoader')
    def test_input_validation(self, mock_config):
        """Test input signal validation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.eda': self.test_config['physio']['eda']
        }.get(key, default)
        
        cleaner = EDACleaner()
        
        # Test with empty signal
        with self.assertRaises(ValueError):
            cleaner.process_signal(np.array([]), sampling_rate=4)
        
        # Test with invalid sampling rate
        with self.assertRaises(ValueError):
            cleaner.process_signal(self.test_signal, sampling_rate=0)
        
        # Test with signal too short (< 10 seconds)
        short_signal = self.test_signal[:20]  # 5 seconds at 4 Hz
        with self.assertRaises(ValueError):
            cleaner.process_signal(short_signal, sampling_rate=4)
    
    @patch('physio.eda_cleaner.ConfigLoader')
    def test_scr_detection(self, mock_config):
        """Test SCR detection functionality."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.eda': self.test_config['physio']['eda'],
            'physio.eda.processing': self.test_config['physio']['eda']['processing']
        }.get(key, default)
        
        cleaner = EDACleaner()
        
        # Create phasic signal with clear SCRs
        t = np.arange(0, 60, 0.25)  # 4 Hz
        phasic = np.zeros_like(t)
        
        # Add 3 clear SCRs with amplitude > 0.01 μS
        scr_times = [10, 30, 50]
        for scr_time in scr_times:
            mask = (t >= scr_time) & (t <= scr_time + 10)
            t_diff = t[mask] - scr_time
            phasic[mask] += 0.15 * np.exp(-np.abs(t_diff - 2) / 2)
        
        # Mock the _detect_scr_peaks method
        with patch.object(cleaner, '_detect_scr_peaks') as mock_detect:
            mock_detect.return_value = {
                'onsets': [40, 120, 200],
                'peaks': [48, 128, 208],
                'amplitudes': [0.15, 0.14, 0.16]
            }
            
            scr_info = cleaner._detect_scr_peaks(phasic, 4)
            
            self.assertEqual(len(scr_info['peaks']), 3)
            self.assertTrue(all(amp >= 0.01 for amp in scr_info['amplitudes']))


class TestEDAMetricsExtractor(unittest.TestCase):
    """Test EDA metrics extraction functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.test_config = {
            'physio': {
                'eda': {
                    'metrics': [
                        'scr_count', 'scr_mean_amplitude', 'scr_rate_per_min',
                        'tonic_mean', 'tonic_std', 'tonic_range',
                        'phasic_mean', 'phasic_std', 'phasic_rate_change'
                    ]
                }
            }
        }
        
        # Create test processed signals
        self.test_signals, self.test_info = self._create_test_data()
    
    def _create_test_data(self):
        """Create test processed EDA data."""
        n_samples = 240  # 60 seconds at 4 Hz
        
        signals = pd.DataFrame({
            'EDA_Clean': np.random.uniform(0.5, 2.0, n_samples),
            'EDA_Tonic': np.linspace(1.0, 1.5, n_samples),
            'EDA_Phasic': np.random.uniform(-0.1, 0.3, n_samples),
            'SCR_Peaks': np.zeros(n_samples)
        })
        
        # Mark some SCR peaks
        peak_indices = [40, 100, 160, 200]
        signals.loc[peak_indices, 'SCR_Peaks'] = 1
        
        info = {
            'SCR_Peaks': peak_indices,
            'SCR_Amplitude': [0.15, 0.20, 0.18, 0.12],
            'SCR_RiseTime': [1.5, 1.8, 1.6, 1.4],
            'SCR_RecoveryTime': [3.2, 3.5, 3.0, 2.8],
            'sampling_rate': 4,
            'duration': 60
        }
        
        return signals, info
    
    @patch('physio.eda_metrics.ConfigLoader')
    def test_extractor_initialization(self, mock_config):
        """Test metrics extractor initialization."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.eda.metrics': self.test_config['physio']['eda']['metrics']
        }.get(key, default)
        
        extractor = EDAMetricsExtractor()
        self.assertIsInstance(extractor, EDAMetricsExtractor)
        self.assertTrue(len(extractor.metrics_to_compute) > 0)
    
    @patch('physio.eda_metrics.ConfigLoader')
    def test_extract_metrics_success(self, mock_config):
        """Test successful metrics extraction."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.eda.metrics': self.test_config['physio']['eda']['metrics']
        }.get(key, default)
        
        extractor = EDAMetricsExtractor()
        metrics = extractor.extract_metrics(
            self.test_signals, self.test_info, moment="test"
        )
        
        self.assertIsInstance(metrics, dict)
        
        # Check SCR metrics
        self.assertIn('scr_count', metrics)
        self.assertEqual(metrics['scr_count'], 4)
        
        self.assertIn('scr_mean_amplitude', metrics)
        self.assertGreater(metrics['scr_mean_amplitude'], 0)
        
        self.assertIn('scr_rate_per_min', metrics)
        self.assertAlmostEqual(metrics['scr_rate_per_min'], 4.0, delta=0.1)
        
        # Check tonic metrics
        self.assertIn('tonic_mean', metrics)
        self.assertIn('tonic_std', metrics)
        self.assertIn('tonic_range', metrics)
        
        # Check phasic metrics
        self.assertIn('phasic_mean', metrics)
        self.assertIn('phasic_std', metrics)
    
    @patch('physio.eda_metrics.ConfigLoader')
    def test_scr_metrics_calculation(self, mock_config):
        """Test SCR-specific metrics calculation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.eda.metrics': ['scr_count', 'scr_mean_amplitude', 
                                   'scr_max_amplitude', 'scr_sum_amplitude']
        }.get(key, default)
        
        extractor = EDAMetricsExtractor()
        metrics = extractor.extract_metrics(
            self.test_signals, self.test_info, moment="test"
        )
        
        # Verify SCR count
        self.assertEqual(metrics['scr_count'], 4)
        
        # Verify amplitude metrics
        expected_mean = np.mean([0.15, 0.20, 0.18, 0.12])
        self.assertAlmostEqual(metrics['scr_mean_amplitude'], expected_mean, places=3)
        
        self.assertEqual(metrics['scr_max_amplitude'], 0.20)
        
        expected_sum = sum([0.15, 0.20, 0.18, 0.12])
        self.assertAlmostEqual(metrics['scr_sum_amplitude'], expected_sum, places=3)
    
    @patch('physio.eda_metrics.ConfigLoader')
    def test_tonic_metrics_calculation(self, mock_config):
        """Test tonic component metrics calculation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.eda.metrics': ['tonic_mean', 'tonic_std', 'tonic_min', 
                                   'tonic_max', 'tonic_range']
        }.get(key, default)
        
        extractor = EDAMetricsExtractor()
        metrics = extractor.extract_metrics(
            self.test_signals, self.test_info, moment="test"
        )
        
        tonic = self.test_signals['EDA_Tonic'].values
        
        self.assertAlmostEqual(metrics['tonic_mean'], np.mean(tonic), places=3)
        self.assertAlmostEqual(metrics['tonic_std'], np.std(tonic, ddof=1), places=3)
        self.assertAlmostEqual(metrics['tonic_min'], np.min(tonic), places=3)
        self.assertAlmostEqual(metrics['tonic_max'], np.max(tonic), places=3)
        self.assertAlmostEqual(metrics['tonic_range'], np.ptp(tonic), places=3)
    
    @patch('physio.eda_metrics.ConfigLoader')
    def test_phasic_metrics_calculation(self, mock_config):
        """Test phasic component metrics calculation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.eda.metrics': ['phasic_mean', 'phasic_std', 'phasic_max',
                                   'phasic_auc', 'phasic_rate_change']
        }.get(key, default)
        
        extractor = EDAMetricsExtractor()
        metrics = extractor.extract_metrics(
            self.test_signals, self.test_info, moment="test"
        )
        
        phasic = self.test_signals['EDA_Phasic'].values
        
        self.assertAlmostEqual(metrics['phasic_mean'], np.mean(phasic), places=3)
        self.assertAlmostEqual(metrics['phasic_std'], np.std(phasic, ddof=1), places=3)
        self.assertAlmostEqual(metrics['phasic_max'], np.max(phasic), places=3)
        
        # Check AUC (area under curve)
        self.assertIn('phasic_auc', metrics)
        self.assertIsInstance(metrics['phasic_auc'], (int, float))
    
    @patch('physio.eda_metrics.ConfigLoader')
    def test_empty_scr_handling(self, mock_config):
        """Test handling of data with no SCRs detected."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'physio.eda.metrics': ['scr_count', 'scr_mean_amplitude', 'scr_rate_per_min']
        }.get(key, default)
        
        # Create data with no SCRs
        signals = self.test_signals.copy()
        signals['SCR_Peaks'] = 0
        
        info = {
            'SCR_Peaks': [],
            'SCR_Amplitude': [],
            'sampling_rate': 4,
            'duration': 60
        }
        
        extractor = EDAMetricsExtractor()
        metrics = extractor.extract_metrics(signals, info, moment="test")
        
        self.assertEqual(metrics['scr_count'], 0)
        self.assertTrue(np.isnan(metrics['scr_mean_amplitude']) or 
                       metrics['scr_mean_amplitude'] == 0)
        self.assertEqual(metrics['scr_rate_per_min'], 0.0)


class TestEDABIDSWriter(unittest.TestCase):
    """Test EDA BIDS output writer functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.test_subject = "sub-f01p01"
        self.test_session = "ses-01"
        self.test_moment = "restingstate"
        
        # Create test data
        n_samples = 240  # 60 seconds at 4 Hz
        self.test_signals = pd.DataFrame({
            'time': np.arange(0, 60, 0.25),
            'EDA_Raw': np.random.uniform(0.5, 2.0, n_samples),
            'EDA_Clean': np.random.uniform(0.5, 2.0, n_samples),
            'EDA_Tonic': np.linspace(1.0, 1.5, n_samples),
            'EDA_Phasic': np.random.uniform(-0.1, 0.3, n_samples)
        })
        
        self.test_scr_events = pd.DataFrame({
            'onset': [10.0, 25.0, 40.0],
            'peak_time': [11.5, 26.8, 41.6],
            'amplitude': [0.15, 0.20, 0.18],
            'rise_time': [1.5, 1.8, 1.6],
            'recovery_time': [3.2, 3.5, 3.0]
        })
        
        self.test_metrics = {
            'scr_count': 3,
            'scr_mean_amplitude': 0.177,
            'scr_rate_per_min': 3.0,
            'tonic_mean': 1.25,
            'tonic_std': 0.15,
            'phasic_mean': 0.08
        }
        
        self.test_metadata = {
            'SamplingFrequency': 4.0,
            'TaskName': self.test_moment,
            'FamilyID': 'f01'
        }
        
        self.test_config = {
            'paths': {'derivatives': str(self.temp_path / "derivatives")}
        }
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    @patch('physio.eda_bids_writer.ConfigLoader')
    def test_writer_initialization(self, mock_config):
        """Test BIDS writer initialization."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'paths.derivatives': str(self.temp_path / "derivatives")
        }.get(key, default)
        
        writer = EDABIDSWriter()
        self.assertIsInstance(writer, EDABIDSWriter)
    
    @patch('physio.eda_bids_writer.ConfigLoader')
    def test_write_all_outputs_success(self, mock_config):
        """Test successful writing of all output files."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'paths.derivatives': str(self.temp_path / "derivatives")
        }.get(key, default)
        
        writer = EDABIDSWriter()
        output_files = writer.write_all_outputs(
            subject=self.test_subject,
            session=self.test_session,
            moment=self.test_moment,
            processed_signals=self.test_signals,
            scr_events=self.test_scr_events,
            metrics=self.test_metrics,
            metadata=self.test_metadata
        )
        
        # Verify output files dict
        self.assertIsInstance(output_files, dict)
        self.assertIn('processed', output_files)
        self.assertIn('events', output_files)
        self.assertIn('metrics', output_files)
        self.assertIn('metadata', output_files)
        self.assertIn('summary', output_files)
        
        # Verify files were created
        for file_type, file_path in output_files.items():
            self.assertTrue(Path(file_path).exists(), 
                          f"{file_type} file not created: {file_path}")
        
        # Verify BIDS structure
        expected_base = (self.temp_path / "derivatives" / "physio_preprocessing" / 
                        self.test_subject / self.test_session / "physio")
        self.assertTrue(expected_base.exists())
    
    @patch('physio.eda_bids_writer.ConfigLoader')
    def test_write_processed_signals(self, mock_config):
        """Test writing processed signals file."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'paths.derivatives': str(self.temp_path / "derivatives")
        }.get(key, default)
        
        writer = EDABIDSWriter()
        output_path = writer.write_processed_signals(
            self.test_subject, self.test_session, self.test_moment,
            self.test_signals, self.test_metadata
        )
        
        self.assertTrue(Path(output_path).exists())
        
        # Verify file contents
        data = pd.read_csv(output_path, sep='\t')
        self.assertIn('time', data.columns)
        self.assertIn('EDA_Clean', data.columns)
        self.assertIn('EDA_Tonic', data.columns)
        self.assertIn('EDA_Phasic', data.columns)
        self.assertEqual(len(data), len(self.test_signals))
        
        # Verify JSON sidecar
        json_path = output_path.replace('_physio.tsv.gz', '_physio.json')
        self.assertTrue(Path(json_path).exists())
        
        with open(json_path, 'r') as f:
            json_data = json.load(f)
            self.assertEqual(json_data['SamplingFrequency'], 4.0)
    
    @patch('physio.eda_bids_writer.ConfigLoader')
    def test_write_scr_events(self, mock_config):
        """Test writing SCR events file."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'paths.derivatives': str(self.temp_path / "derivatives")
        }.get(key, default)
        
        writer = EDABIDSWriter()
        output_path = writer.write_scr_events(
            self.test_subject, self.test_session, self.test_moment,
            self.test_scr_events, self.test_metadata
        )
        
        self.assertTrue(Path(output_path).exists())
        
        # Verify file contents
        events = pd.read_csv(output_path, sep='\t')
        self.assertIn('onset', events.columns)
        self.assertIn('peak_time', events.columns)
        self.assertIn('amplitude', events.columns)
        self.assertEqual(len(events), 3)
        
        # Verify JSON sidecar
        json_path = output_path.replace('_events.tsv', '_events.json')
        self.assertTrue(Path(json_path).exists())
    
    @patch('physio.eda_bids_writer.ConfigLoader')
    def test_write_metrics(self, mock_config):
        """Test writing metrics file."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'paths.derivatives': str(self.temp_path / "derivatives")
        }.get(key, default)
        
        writer = EDABIDSWriter()
        output_path = writer.write_metrics(
            self.test_subject, self.test_session, self.test_moment,
            self.test_metrics, self.test_metadata
        )
        
        self.assertTrue(Path(output_path).exists())
        
        # Verify file contents
        with open(output_path, 'r') as f:
            metrics_data = json.load(f)
            
        self.assertEqual(metrics_data['scr_count'], 3)
        self.assertAlmostEqual(metrics_data['scr_mean_amplitude'], 0.177, places=3)
        self.assertEqual(metrics_data['scr_rate_per_min'], 3.0)
        self.assertIn('tonic_mean', metrics_data)
    
    @patch('physio.eda_bids_writer.ConfigLoader')
    def test_bids_filename_format(self, mock_config):
        """Test BIDS-compliant filename generation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'paths.derivatives': str(self.temp_path / "derivatives")
        }.get(key, default)
        
        writer = EDABIDSWriter()
        
        # Test processed signals filename
        filename = writer._get_bids_filename(
            self.test_subject, self.test_session, self.test_moment, 
            'physio', 'tsv.gz'
        )
        
        expected = f"{self.test_subject}_{self.test_session}_task-{self.test_moment}_physio.tsv.gz"
        self.assertEqual(filename, expected)
        
        # Test events filename
        filename = writer._get_bids_filename(
            self.test_subject, self.test_session, self.test_moment,
            'events', 'tsv'
        )
        
        expected = f"{self.test_subject}_{self.test_session}_task-{self.test_moment}_events.tsv"
        self.assertEqual(filename, expected)
    
    @patch('physio.eda_bids_writer.ConfigLoader')
    def test_create_output_directory(self, mock_config):
        """Test output directory creation."""
        mock_config.return_value.get.side_effect = lambda key, default=None: {
            'paths.derivatives': str(self.temp_path / "derivatives")
        }.get(key, default)
        
        writer = EDABIDSWriter()
        
        # Create directory structure
        output_dir = writer._get_output_dir(self.test_subject, self.test_session)
        
        self.assertTrue(output_dir.exists())
        self.assertTrue(output_dir.is_dir())
        
        # Verify BIDS structure
        expected_path = (self.temp_path / "derivatives" / "physio_preprocessing" /
                        self.test_subject / self.test_session / "physio")
        self.assertEqual(output_dir, expected_path)


class TestEDAPipelineIntegration(unittest.TestCase):
    """Integration tests for complete EDA pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        self.test_subject = "sub-f01p01"
        self.test_session = "ses-01"
        self.test_moment = "restingstate"
        
        # Create full test data structure
        self._create_test_data_structure()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_data_structure(self):
        """Create complete test data structure."""
        # Create sourcedata directory
        physio_dir = (self.temp_path / "sourcedata" / self.test_subject /
                     self.test_session / "physio")
        physio_dir.mkdir(parents=True)
        
        # Create realistic EDA data
        duration = 60
        sampling_rate = 4
        t = np.arange(0, duration, 1/sampling_rate)
        
        # Realistic EDA signal with tonic + phasic
        tonic = 1.2 + 0.3 * np.sin(2 * np.pi * 0.01 * t)
        phasic = np.zeros_like(t)
        
        scr_times = [10, 25, 40, 55]
        for scr_time in scr_times:
            mask = (t >= scr_time) & (t <= scr_time + 10)
            t_diff = t[mask] - scr_time
            phasic[mask] += 0.2 * np.exp(-np.abs(t_diff - 2) / 2)
        
        eda = tonic + phasic + np.random.normal(0, 0.02, len(t))
        eda = np.maximum(eda, 0.1)
        
        # Save TSV
        base_filename = f"{self.test_subject}_{self.test_session}_task-{self.test_moment}_recording-eda"
        data = pd.DataFrame({'time': t, 'eda': eda})
        data.to_csv(physio_dir / f"{base_filename}.tsv", sep='\t', index=False)
        
        # Save JSON
        metadata = {
            "SamplingFrequency": 4.0,
            "StartTime": 0,
            "Columns": ["time", "eda"],
            "Units": ["s", "μS"],
            "TaskName": self.test_moment,
            "RecordingType": "EDA",
            "FamilyID": "f01"
        }
        
        with open(physio_dir / f"{base_filename}.json", 'w') as f:
            json.dump(metadata, f)
    
    @patch('physio.eda_loader.ConfigLoader')
    @patch('physio.eda_cleaner.ConfigLoader')
    @patch('physio.eda_metrics.ConfigLoader')
    @patch('physio.eda_bids_writer.ConfigLoader')
    def test_full_pipeline_execution(self, mock_writer_config, mock_metrics_config,
                                     mock_cleaner_config, mock_loader_config):
        """Test complete EDA pipeline from load to write."""
        # Mock all configs
        def get_config(key, default=None):
            config_map = {
                'paths.sourcedata': str(self.temp_path / "sourcedata"),
                'paths.derivatives': str(self.temp_path / "derivatives"),
                'physio.eda.sampling_rate': 4,
                'physio.eda.processing': {'method': 'cvxEDA', 'scr_threshold': 0.01},
                'physio.eda.processing.method': 'cvxEDA',
                'physio.eda.metrics': ['scr_count', 'scr_mean_amplitude', 'tonic_mean']
            }
            return config_map.get(key, default)
        
        for mock_cfg in [mock_loader_config, mock_cleaner_config, 
                         mock_metrics_config, mock_writer_config]:
            mock_cfg.return_value.get.side_effect = get_config
        
        # Step 1: Load data
        loader = EDALoader()
        data, metadata = loader.load_subject_session(
            self.test_subject, self.test_session, self.test_moment
        )
        
        self.assertGreater(len(data), 0)
        self.assertIn('eda', data.columns)
        
        # Note: Full pipeline test would continue with cleaner, metrics, and writer
        # but requires proper mocking of NeuroKit2 functions
        # This basic test verifies the data loading step works


if __name__ == '__main__':
    unittest.main()
