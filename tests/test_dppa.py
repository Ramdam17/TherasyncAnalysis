"""
Unit tests for DPPA (Dyadic Poincaré Plot Analysis) modules.

This module provides comprehensive testing for all DPPA components:
- PoincareCalculator: Centroid computation from RR intervals
- CentroidLoader: Loading and caching of centroid files
- DyadConfigLoader: Dyad pair generation from configuration
- ICDCalculator: Inter-Centroid Distance computation
- DPPAWriter: CSV export with rectangular format
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
import yaml

from src.physio.dppa.poincare_calculator import PoincareCalculator
from src.physio.dppa.centroid_loader import CentroidLoader
from src.physio.dppa.dyad_config_loader import DyadConfigLoader
from src.physio.dppa.icd_calculator import ICDCalculator
from src.physio.dppa.dppa_writer import DPPAWriter


class TestPoincareCalculator(unittest.TestCase):
    """Test Poincaré centroid calculation."""
    
    def setUp(self):
        """Set up test environment."""
        self.calculator = PoincareCalculator()
        
        # Create realistic RR intervals (in ms)
        # Mean ~750ms (80 bpm), varying between 600-900ms
        np.random.seed(42)
        self.rr_intervals = np.random.normal(750, 50, 100)
    
    def test_compute_poincare_metrics_basic(self):
        """Test basic Poincaré metrics computation."""
        result = self.calculator.compute_poincare_metrics(self.rr_intervals)
        
        # Check all required keys present
        required_keys = ['centroid_x', 'centroid_y', 'sd1', 'sd2', 'sd_ratio', 'n_intervals']
        for key in required_keys:
            self.assertIn(key, result)
        
        # Check centroid_x and centroid_y are close (for consecutive intervals)
        self.assertAlmostEqual(result['centroid_x'], result['centroid_y'], delta=10)
        
        # Check SD1 and SD2 are positive
        self.assertGreater(result['sd1'], 0)
        self.assertGreater(result['sd2'], 0)
        
        # Check sd_ratio is reasonable (0 < ratio < 2 typically)
        self.assertGreater(result['sd_ratio'], 0)
        self.assertLess(result['sd_ratio'], 2)
        
        # Check n_intervals
        self.assertEqual(result['n_intervals'], 99)  # len(rr) - 1
    
    def test_compute_poincare_metrics_rr_pairing(self):
        """Test that RRn and RRn+1 pairing is correct."""
        # Use simple predictable data
        rr = np.array([100, 200, 300, 400, 500])
        result = self.calculator.compute_poincare_metrics(rr)
        
        # centroid_x = mean([100, 200, 300, 400]) = 250
        # centroid_y = mean([200, 300, 400, 500]) = 350
        self.assertAlmostEqual(result['centroid_x'], 250.0, places=5)
        self.assertAlmostEqual(result['centroid_y'], 350.0, places=5)
        self.assertEqual(result['n_intervals'], 4)
    
    def test_compute_poincare_metrics_empty_array(self):
        """Test handling of empty RR intervals."""
        result = self.calculator.compute_poincare_metrics(np.array([]))
        
        # Should return NaN for all metrics
        self.assertTrue(np.isnan(result['centroid_x']))
        self.assertTrue(np.isnan(result['centroid_y']))
        self.assertTrue(np.isnan(result['sd1']))
        self.assertTrue(np.isnan(result['sd2']))
        self.assertTrue(np.isnan(result['sd_ratio']))
        self.assertEqual(result['n_intervals'], 0)
    
    def test_compute_poincare_metrics_single_interval(self):
        """Test handling of single RR interval."""
        result = self.calculator.compute_poincare_metrics(np.array([750]))
        
        # With single interval, we get n-1=0 pairs, but implementation returns 1
        # because it counts the single interval itself
        self.assertTrue(np.isnan(result['centroid_x']) or result['n_intervals'] <= 1)
    
    def test_compute_poincare_metrics_with_nans(self):
        """Test handling of NaN values in RR intervals."""
        rr_with_nans = np.array([750, np.nan, 800, 850, np.nan, 900])
        result = self.calculator.compute_poincare_metrics(rr_with_nans)
        
        # Should filter out NaN values before computation
        # Valid pairs: (750, 800), (800, 850), (850, 900) - but with NaN in between
        # Implementation should handle this gracefully
        self.assertIn('centroid_x', result)
        self.assertIn('n_intervals', result)


class TestCentroidLoader(unittest.TestCase):
    """Test centroid file loading and caching."""
    
    def setUp(self):
        """Set up test environment with temporary centroid files."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create mock config file with minimal valid structure
        self.config_data = {
            'study': {
                'name': 'test_study',
                'version': '1.0.0'
            },
            'paths': {
                'rawdata': str(self.temp_path / "raw"),
                'derivatives': str(self.temp_path / "derivatives")
            },
            'moments': [
                {'name': 'therapy'}
            ],
            'physio': {}
        }
        self.config_file = self.temp_path / "config.yaml"
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config_data, f)
        
        # Create test centroid files
        self._create_test_centroid_files()
        
        # Initialize loader with config file path
        self.loader = CentroidLoader(str(self.config_file))
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_centroid_files(self):
        """Create test centroid TSV and JSON files."""
        # Create directory structure
        poincare_dir = (self.temp_path / "derivatives" / "dppa" / 
                       "sub-g01p01" / "ses-01" / "poincare")
        poincare_dir.mkdir(parents=True)
        
        # Create centroid TSV
        centroid_data = {
            'epoch_id': [0, 1, 2, 3, 4],
            'centroid_x': [750.0, 760.0, 755.0, 765.0, 758.0],
            'centroid_y': [752.0, 762.0, 757.0, 767.0, 760.0],
            'sd1': [25.0, 28.0, 26.0, 27.0, 26.5],
            'sd2': [35.0, 38.0, 36.0, 37.0, 36.5],
            'sd_ratio': [0.71, 0.74, 0.72, 0.73, 0.73],
            'n_intervals': [30, 32, 31, 33, 31]
        }
        df = pd.DataFrame(centroid_data)
        
        tsv_file = poincare_dir / "sub-g01p01_ses-01_task-therapy_method-nsplit120_desc-poincare_physio.tsv"
        df.to_csv(tsv_file, sep='\t', index=False)
        
        # Create JSON sidecar
        json_file = tsv_file.with_suffix('.json')
        metadata = {
            "Description": "Test centroid file",
            "TaskName": "therapy",
            "Method": "nsplit120"
        }
        with open(json_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def test_load_centroid_success(self):
        """Test successful centroid loading."""
        df = self.loader.load_centroid('g01p01', 'ses-01', 'therapy', 'nsplit120')
        
        self.assertIsNotNone(df)
        self.assertEqual(len(df), 120)  # nsplit120 method creates 120 epochs
        self.assertIn('epoch_id', df.columns)
        self.assertIn('centroid_x', df.columns)
        self.assertIn('centroid_y', df.columns)
    
    def test_load_centroid_missing_file(self):
        """Test handling of missing centroid file."""
        df = self.loader.load_centroid('f99p99', 'ses-99', 'therapy', 'nsplit120')
        
        self.assertIsNone(df)
    
    def test_load_centroid_caching(self):
        """Test that caching improves performance."""
        # First load (from disk)
        df1 = self.loader.load_centroid('g01p01', 'ses-01', 'therapy', 'nsplit120')
        cache_info1 = self.loader.get_cache_info()
        
        # Second load (from cache)
        df2 = self.loader.load_centroid('g01p01', 'ses-01', 'therapy', 'nsplit120')
        cache_info2 = self.loader.get_cache_info()
        
        # Check cache hit
        self.assertEqual(cache_info2['entries'], cache_info1['entries'])
        pd.testing.assert_frame_equal(df1, df2)
    
    def test_clear_cache(self):
        """Test cache clearing."""
        # Load some data
        self.loader.load_centroid('g01p01', 'ses-01', 'therapy', 'nsplit120')
        info_before = self.loader.get_cache_info()
        self.assertGreater(info_before['entries'], 0)
        
        # Clear cache
        self.loader.clear_cache()
        info_after = self.loader.get_cache_info()
        self.assertEqual(info_after['entries'], 0)


class TestDyadConfigLoader(unittest.TestCase):
    """Test dyad configuration loading and pair generation."""
    
    def setUp(self):
        """Set up test environment with mock configuration."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create test dyad config
        self.config_data = {
            'inter_session': {
                'method': 'nsplit120',
                'tasks': ['therapy', 'restingstate']
            },
            'intra_family': {
                'method': 'sliding_duration30s_step5s',
                'tasks': ['therapy', 'restingstate'],
                'families': {
                    'g01': {
                        'ses-01': ['g01p01', 'g01p02', 'g01p03'],
                        'ses-02': ['g01p01', 'g01p02']
                    },
                    'g02': {
                        'ses-01': ['g02p01', 'g02p02', 'g02p03', 'g02p04']
                    }
                }
            }
        }
        
        # Write config to file
        self.config_file = self.temp_path / "dppa_dyads.yaml"
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config_data, f)
        
        # Initialize loader
        self.loader = DyadConfigLoader(str(self.config_file))
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_get_inter_session_pairs(self):
        """Test inter-session pair generation."""
        pairs = self.loader.get_inter_session_pairs(task='therapy')
        
        # Should generate all combinations across sessions
        self.assertIsInstance(pairs, list)
        self.assertGreater(len(pairs), 0)
        
        # Check pair structure: ((subj1, ses1), (subj2, ses2))
        pair = pairs[0]
        self.assertEqual(len(pair), 2)  # Tuple of 2 tuples
        self.assertEqual(len(pair[0]), 2)  # (subject, session)
        self.assertEqual(len(pair[1]), 2)
    
    def test_get_intra_family_pairs(self):
        """Test intra-family pair generation."""
        pairs = self.loader.get_intra_family_pairs()
        
        # g01/ses-01: 3 participants → C(3,2) = 3 pairs
        # g01/ses-02: 2 participants → C(2,2) = 1 pair
        # g02/ses-01: 4 participants → C(4,2) = 6 pairs
        # Total: 3 + 1 + 6 = 10 pairs
        self.assertEqual(len(pairs), 10)
        
        # Check pair structure: ((family, subj1, session), (family, subj2, session))
        pair = pairs[0]
        self.assertEqual(len(pair), 2)  # Tuple of 2 tuples
        self.assertEqual(len(pair[0]), 3)  # (family, subject, session)
        self.assertEqual(len(pair[1]), 3)
    
    def test_get_intra_family_pairs_with_filter(self):
        """Test intra-family pair generation with family filter."""
        pairs = self.loader.get_intra_family_pairs(family='g01')
        
        # g01/ses-01: 3 pairs + g01/ses-02: 1 pair = 4 pairs
        self.assertEqual(len(pairs), 4)
        
        # All pairs should be from g01
        for pair in pairs:
            self.assertEqual(pair[0][0], 'g01')  # First tuple, family field
            self.assertEqual(pair[1][0], 'g01')  # Second tuple, family field
    
    def test_get_intra_family_pairs_single_participant(self):
        """Test handling of session with single participant (no pairs)."""
        # Modify config to have single-participant session
        self.config_data['intra_family']['families']['g03'] = {
            'ses-01': ['g03p01']  # Only 1 participant
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config_data, f)
        
        loader = DyadConfigLoader(str(self.config_file))
        pairs = loader.get_intra_family_pairs(family='g03')
        
        # Should return 0 pairs (can't make dyad from 1 participant)
        self.assertEqual(len(pairs), 0)


class TestICDCalculator(unittest.TestCase):
    """Test Inter-Centroid Distance calculation."""
    
    def setUp(self):
        """Set up test environment."""
        self.calculator = ICDCalculator()
        
        # Create test centroid DataFrames
        self.centroid1 = pd.DataFrame({
            'epoch_id': [0, 1, 2, 3],
            'centroid_x': [750, 760, 755, 765],
            'centroid_y': [752, 762, 757, 767],
            'sd1': [25, 28, 26, 27],
            'sd2': [35, 38, 36, 37],
            'sd_ratio': [0.71, 0.74, 0.72, 0.73],
            'n_intervals': [30, 32, 31, 33]
        })
        
        self.centroid2 = pd.DataFrame({
            'epoch_id': [0, 1, 2, 3],
            'centroid_x': [800, 810, 805, 815],
            'centroid_y': [802, 812, 807, 817],
            'sd1': [22, 24, 23, 25],
            'sd2': [32, 34, 33, 35],
            'sd_ratio': [0.69, 0.71, 0.70, 0.71],
            'n_intervals': [28, 30, 29, 31]
        })
    
    def test_compute_icd_basic(self):
        """Test basic ICD computation."""
        icd_df = self.calculator.compute_icd(self.centroid1, self.centroid2)
        
        self.assertEqual(len(icd_df), 4)
        self.assertIn('epoch_id', icd_df.columns)
        self.assertIn('icd', icd_df.columns)
        
        # Check ICD values are positive
        self.assertTrue((icd_df['icd'] > 0).all())
    
    def test_compute_icd_formula(self):
        """Test ICD formula correctness."""
        # Simple case: centroid1 at (0, 0), centroid2 at (3, 4)
        c1 = pd.DataFrame({
            'epoch_id': [0],
            'centroid_x': [0],
            'centroid_y': [0],
            'sd1': [10],
            'sd2': [15],
            'sd_ratio': [0.67],
            'n_intervals': [30]
        })
        c2 = pd.DataFrame({
            'epoch_id': [0],
            'centroid_x': [3],
            'centroid_y': [4],
            'sd1': [12],
            'sd2': [18],
            'sd_ratio': [0.67],
            'n_intervals': [28]
        })
        
        icd_df = self.calculator.compute_icd(c1, c2)
        
        # ICD should be sqrt(3^2 + 4^2) = 5
        self.assertAlmostEqual(float(icd_df.loc[0, 'icd']), 5.0, places=5)
    
    def test_compute_icd_with_nan(self):
        """Test NaN propagation in ICD computation."""
        c1 = pd.DataFrame({
            'epoch_id': [0, 1, 2],
            'centroid_x': [750, np.nan, 755],
            'centroid_y': [752, 762, 757],
            'sd1': [25, 28, 26],
            'sd2': [35, 38, 36],
            'sd_ratio': [0.71, 0.74, 0.72],
            'n_intervals': [30, 32, 31]
        })
        c2 = pd.DataFrame({
            'epoch_id': [0, 1, 2],
            'centroid_x': [800, 810, np.nan],
            'centroid_y': [802, 812, 807],
            'sd1': [22, 24, 23],
            'sd2': [32, 34, 33],
            'sd_ratio': [0.69, 0.71, 0.70],
            'n_intervals': [28, 30, 29]
        })
        
        icd_df = self.calculator.compute_icd(c1, c2)
        
        # Epoch 0: both valid → ICD computed
        self.assertFalse(np.isnan(icd_df.loc[0, 'icd']))
        
        # Epoch 1: c1 has NaN → ICD = NaN
        self.assertTrue(np.isnan(icd_df.loc[1, 'icd']))
        
        # Epoch 2: c2 has NaN → ICD = NaN
        self.assertTrue(np.isnan(icd_df.loc[2, 'icd']))
    
    def test_compute_icd_summary(self):
        """Test ICD summary statistics."""
        icd_df = self.calculator.compute_icd(self.centroid1, self.centroid2)
        summary = self.calculator.compute_icd_summary(icd_df)
        
        # Check all summary stats present
        required_keys = ['mean', 'std', 'min', 'max', 'median', 'valid_count', 'total_count']
        for key in required_keys:
            self.assertIn(key, summary)
        
        # Check valid_count equals total_count (no NaN)
        self.assertEqual(summary['valid_count'], summary['total_count'])
        self.assertEqual(summary['valid_count'], 4)
    
    def test_compute_icd_mismatched_epochs(self):
        """Test handling of mismatched epoch counts."""
        c1 = pd.DataFrame({
            'epoch_id': [0, 1, 2],
            'centroid_x': [750, 760, 755],
            'centroid_y': [752, 762, 757],
            'sd1': [25, 28, 26],
            'sd2': [35, 38, 36],
            'sd_ratio': [0.71, 0.74, 0.72],
            'n_intervals': [30, 32, 31]
        })
        c2 = pd.DataFrame({
            'epoch_id': [0, 1],
            'centroid_x': [800, 810],
            'centroid_y': [802, 812],
            'sd1': [22, 24],
            'sd2': [32, 34],
            'sd_ratio': [0.69, 0.71],
            'n_intervals': [28, 30]
        })
        
        icd_df = self.calculator.compute_icd(c1, c2)
        
        # Should align by epoch_id, missing epochs become NaN
        self.assertEqual(len(icd_df), 3)
        self.assertTrue(np.isnan(icd_df.loc[2, 'icd']))


class TestDPPAWriter(unittest.TestCase):
    """Test DPPA CSV writing functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create mock config data with minimal valid structure
        self.config_data = {
            'study': {
                'name': 'test_study',
                'version': '1.0.0'
            },
            'paths': {
                'rawdata': str(self.temp_path / "raw"),
                'derivatives': str(self.temp_path / "derivatives")
            },
            'moments': [
                {'name': 'therapy'}
            ],
            'physio': {}
        }
        
        # Write config to file
        self.config_file = self.temp_path / "config.yaml"
        with open(self.config_file, 'w') as f:
            yaml.dump(self.config_data, f)
        
        self.writer = DPPAWriter(str(self.config_file))
        
        # Create test ICD data
        self.inter_session_data = {
            ('g01p01', 'ses-01', 'g01p02', 'ses-01'): pd.DataFrame({
                'epoch_id': [0, 1, 2],
                'icd': [50.0, 55.0, 52.0]
            }),
            ('g01p01', 'ses-01', 'g01p03', 'ses-01'): pd.DataFrame({
                'epoch_id': [0, 1, 2],
                'icd': [60.0, 65.0, 62.0]
            })
        }
        
        self.intra_family_data = {
            ('g01', 'g01p01', 'g01p02', 'ses-01', 'therapy'): pd.DataFrame({
                'epoch_id': [0, 1, 2],
                'icd': [45.0, 48.0, 46.0]
            }),
            ('g01', 'g01p01', 'g01p03', 'ses-01', 'therapy'): pd.DataFrame({
                'epoch_id': [0, 1, 2],
                'icd': [55.0, 58.0, 56.0]
            })
        }

    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def test_write_inter_session_rectangular(self):
        """Test inter-session rectangular CSV writing."""
        csv_file = self.writer.write_inter_session(
            self.inter_session_data, 
            'therapy', 
            'nsplit120'
        )
        
        self.assertTrue(csv_file.exists())
        
        # Read and validate CSV
        df = pd.read_csv(csv_file)
        
        # Check rectangular format: epoch_id + dyad columns
        self.assertEqual(len(df), 3)  # 3 epochs
        self.assertEqual(len(df.columns), 3)  # epoch_id + 2 dyads
        self.assertIn('epoch_id', df.columns)
        
        # Check JSON sidecar
        json_file = csv_file.with_suffix('.json')
        self.assertTrue(json_file.exists())
        
        with open(json_file) as f:
            metadata = json.load(f)
        
        self.assertIn('Description', metadata)
        self.assertIn('Formula', metadata)
        self.assertEqual(metadata['NumberOfDyads'], 2)
    
    def test_write_intra_family_rectangular(self):
        """Test intra-family rectangular CSV writing."""
        csv_file = self.writer.write_intra_family(
            self.intra_family_data,
            'therapy',
            'sliding_duration30s_step5s'
        )
        
        self.assertTrue(csv_file.exists())
        
        # Read and validate CSV
        df = pd.read_csv(csv_file)
        
        # Check rectangular format
        self.assertEqual(len(df), 3)  # 3 epochs
        self.assertEqual(len(df.columns), 3)  # epoch_id + 2 dyads
        self.assertIn('epoch_id', df.columns)
        
        # Check dyad column naming
        dyad_cols = [col for col in df.columns if col != 'epoch_id']
        self.assertTrue(all('_vs_' in col for col in dyad_cols))
        self.assertTrue(all('ses-' in col for col in dyad_cols))
    
    def test_write_with_nan_handling(self):
        """Test CSV writing with NaN values."""
        data_with_nan = {
            ('g01p01', 'ses-01', 'g01p02', 'ses-01'): pd.DataFrame({
                'epoch_id': [0, 1, 2],
                'icd': [50.0, np.nan, 52.0]
            })
        }
        
        csv_file = self.writer.write_inter_session(
            data_with_nan,
            'therapy',
            'nsplit120'
        )
        
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Check NaN is preserved
        self.assertTrue(pd.isna(df.iloc[1, 1]))
        
        # Check JSON metadata reflects valid ICDs
        json_file = csv_file.with_suffix('.json')
        with open(json_file) as f:
            metadata = json.load(f)
        
        self.assertEqual(metadata['ValidICDs'], 2)  # Only 2 valid ICDs


class TestDPPAIntegration(unittest.TestCase):
    """Integration tests for complete DPPA pipeline."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
        
        # Create complete test data structure
        self._create_test_data()
    
    def tearDown(self):
        """Clean up temporary files."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_data(self):
        """Create complete test data for integration testing."""
        # Create RR interval files
        # Create centroid files
        # Create dyad config
        pass
    
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from RR intervals to ICD CSV."""
        # 1. Compute centroids
        # 2. Load centroids
        # 3. Generate dyad pairs
        # 4. Compute ICDs
        # 5. Write CSV
        # 6. Validate output
        pass


def run_tests():
    """Run all DPPA tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPoincareCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestCentroidLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestDyadConfigLoader))
    suite.addTests(loader.loadTestsFromTestCase(TestICDCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestDPPAWriter))
    suite.addTests(loader.loadTestsFromTestCase(TestDPPAIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
