import contextlib
import io
import unittest

import numpy as np

import benchmark_etth1 as b


class SeasonalNaiveTests(unittest.TestCase):
    def test_copies_target_history_in_order_for_each_horizon(self):
        values = np.column_stack((np.arange(800), -np.arange(800)))
        for horizon in (24, 168):
            origins = np.array([512, 600])
            batch, _ = b.make_windows(values, origins, 512, horizon)
            expected = np.stack([np.arange(t - horizon, t) for t in origins])
            np.testing.assert_array_equal(b.seasonal_naive(batch, horizon), expected)

    def test_rejects_insufficient_history(self):
        with self.assertRaises(ValueError):
            b.seasonal_naive(np.zeros((2, 24, 1)), 168)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            b.parse_args(['--models', 'seasonal-naive', '--context', '24', '--horizons', '168'])

    def test_weekly_alias_is_removed(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            b.parse_args(['--models', 'seasonal-naive-weekly'])
