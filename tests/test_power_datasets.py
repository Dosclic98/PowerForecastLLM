import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import benchmark_etth1 as b
from power_datasets import HOUSEHOLD_FEATURES, hourly_aggregate


class PowerDatasetTests(unittest.TestCase):
    def electricity(self, dates, values):
        return pd.DataFrame({'date': dates.strftime('%Y-%m-%d %H:%M:%S'), 'MT_001': values})

    def test_electricity_interval_alignment_and_chunk_boundaries(self):
        dates = pd.date_range('2012-01-01 00:15', periods=8, freq='15min')
        frame = self.electricity(dates, [4, 8, 12, 16, 20, 24, 28, 32])
        result, count = hourly_aggregate([frame.iloc[:3], frame.iloc[3:]], 'electricity')
        self.assertEqual(count, 8)
        self.assertEqual(result.index[0], pd.Timestamp('2012-01-01 00:00'))
        np.testing.assert_allclose(result.MT_001, [10, 26])

    def test_electricity_masks_leading_zeros_but_keeps_later_zeros(self):
        dates = pd.date_range('2012-01-01 00:15', periods=12, freq='15min')
        result, _ = hourly_aggregate([self.electricity(dates, [0]*4 + [4]*4 + [0]*4)], 'electricity')
        self.assertTrue(np.isnan(result.iloc[0, 0]))
        np.testing.assert_allclose(result.iloc[1:, 0], [4, 0])

    def test_dst_day_is_masked(self):
        dates = pd.date_range('2012-03-24 00:15', periods=96*3, freq='15min')
        result, _ = hourly_aggregate([self.electricity(dates, np.ones(len(dates)))], 'electricity')
        self.assertTrue(result.loc['2012-03-25'].isna().all().all())
        self.assertTrue(result.loc['2012-03-24'].notna().all().all())
        self.assertTrue(result.loc['2012-03-26'].notna().all().all())

    def test_household_units_and_missing_coverage(self):
        dates = pd.date_range('2007-01-01', periods=120, freq='min')
        frame = pd.DataFrame({name: np.ones(120) for name in HOUSEHOLD_FEATURES})
        frame['Date'] = dates.strftime('%d/%m/%Y')
        frame['Time'] = dates.strftime('%H:%M:%S')
        frame.loc[75, 'Global_active_power'] = np.nan
        result, _ = hourly_aggregate([frame.iloc[:33], frame.iloc[33:]], 'household')
        self.assertEqual(result.iloc[0].Global_active_power, 1.)
        self.assertEqual(result.iloc[0].Sub_metering_1, 60.)
        self.assertTrue(np.isnan(result.iloc[1].Global_active_power))
        self.assertEqual(result.iloc[1].Voltage, 1.)

    def test_window_filter_does_not_compress_time_or_require_future_covariates(self):
        values = np.ones((20, 2))
        values[5, 0] = np.nan
        values[12, 1] = np.nan
        origins = b.complete_origins(values, np.arange(4, 17), 4, 4)
        np.testing.assert_array_equal(origins, [10, 11, 12])
        x, y = b.make_windows(values, origins, 4, 4)
        self.assertTrue(np.isfinite(x).all() and np.isfinite(y).all())

    def test_mase_uses_actual_lag_pairs_across_gaps(self):
        training = np.array([1., 2., np.nan, 10., 12.])
        result = b.accuracy(np.zeros((1, 2)), np.ones((1, 2)), training, 1)
        self.assertAlmostEqual(result['mase'], 1 / 1.5)

    def test_custom_splits_propagate_to_cv_and_target_loader(self):
        args = b.parse_args(['--cv-folds', '5', '--split-ends', '16000', '20000',
                             '--target', 'load', '--feature-columns', 'load', 'voltage'])
        folds = b.cross_validation_folds(args)
        self.assertEqual(folds[0]['fit_score_start'], 16000 // 6)
        self.assertEqual(folds[-1]['fit_valid_end'], 16000)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'data.csv'
            pd.DataFrame({'date': pd.date_range('2000-01-01', periods=10, freq='h'),
                          'load': [1.]*9 + [np.nan], 'voltage': [230.]*10}).to_csv(path, index=False)
            _, values = b.load_data(path, features=['load', 'voltage'], test_end=10, allow_missing=True)
            self.assertEqual(values.shape, (10, 2))
            with self.assertRaises(ValueError):
                b.load_data(path, features=['load'], test_end=10)


if __name__ == '__main__':
    unittest.main()
