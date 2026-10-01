import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import benchmark_etth1 as engine
import benchmark_campaign as campaign
from campaign_reporting import summarize_campaign


class RepetitionTests(unittest.TestCase):
    def test_repeat_budget_and_invalid_count(self):
        args, _, settings = campaign.parse_args(['--repeats', '10'])
        budget = campaign.plan(args, settings)
        self.assertEqual(budget['total_lstm_fits'], 480)
        self.assertEqual(budget['model_evaluations'], 1200)
        for value in ('0', '-1'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                engine.parse_args(['--repeats', value])

    def test_fresh_setup_and_fits_with_exact_work_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.random.default_rng(7).normal(size=(240, 2)).astype(np.float32)
            values[60, 1] = np.nan
            source = root / 'input.csv'
            pd.DataFrame({'date': pd.date_range('2000-01-01', periods=240, freq='h'),
                          'target': values[:, 0], 'aux': values[:, 1]}).to_csv(source, index=False)
            output = root / 'run'
            steps = []
            actual_step = torch.optim.Adam.step

            def counted_step(optimizer, *args, **kwargs):
                steps.append(1)
                return actual_step(optimizer, *args, **kwargs)

            def fake_foundation(*_args):
                return lambda x, h: np.repeat(x[:, -1, :1], h, axis=1)

            flags = ['--data', str(source), '--target', 'target', '--feature-columns', 'target', 'aux',
                     '--allow-missing', '--split-ends', '192', '240', '--context', '8', '--season', '1',
                     '--horizons', '4', '--early-stopping-hours', '32', '--epochs', '2', '--hidden-size', '4',
                     '--train-stride', '8', '--validation-stride', '4', '--stride', '4', '--train-batch-size', '8',
                     '--models', 'lstm', 'chronos2', 'seasonal-naive', '--max-origins', '3', '--threads', '1',
                     '--repeats', '2', '--device', 'cpu', '--output', str(output)]
            with patch.object(engine, 'foundation_adapter', side_effect=fake_foundation) as setup, \
                    patch.object(engine, 'fit_lstm_candidate', wraps=engine.fit_lstm_candidate) as fit, \
                    patch.object(torch.optim.Adam, 'step', counted_step), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(engine.main(flags), 0)
            self.assertEqual(setup.call_count, 2)
            self.assertEqual(fit.call_count, 4)
            rows = pd.read_csv(output / 'metrics.csv')
            self.assertEqual(len(rows), 6)
            lstms = rows[rows.model == 'lstm']
            self.assertEqual(lstms.total_training_batches.sum(), len(steps))
            for row in lstms.itertuples():
                self.assertEqual(row.initial_train_windows, 18)  # One incomplete window removed.
                self.assertEqual(row.refit_train_windows, 22)
                self.assertEqual(row.initial_training_batches, math.ceil(18 / 8) * row.initial_epochs_run)
                self.assertEqual(row.refit_training_batches, math.ceil(22 / 8) * row.refit_epochs)
                self.assertEqual(row.total_training_batches, row.initial_training_batches + row.refit_training_batches)
                self.assertEqual(row.total_fit_batches, row.total_training_batches + row.total_early_stopping_batches)
                self.assertEqual(row.total_training_window_presentations, 18 * row.initial_epochs_run + 22 * row.refit_epochs)
                report = json.loads((output / row.artifact_directory / 'lstm_h4_seed42_selection.json').read_text())
                self.assertEqual(report['workload']['total_training_batches'], row.total_training_batches)
            for row in rows.itertuples():
                label = f'{row.model}_h4' + ('_seed42' if row.model == 'lstm' else '')
                self.assertTrue((output / row.artifact_directory / f'{label}_predictions.csv').exists())
            for metric in ('mae', 'rmse', 'refit_epochs', 'total_training_batches'):
                self.assertEqual(lstms.iloc[0][metric], lstms.iloc[1][metric])
            window_summary = pd.read_csv(output / 'summary/window_summary.csv')
            self.assertEqual(len(window_summary), 6)
            self.assertTrue(window_summary.mae_window_count.eq(3).all())
            summary = pd.read_csv(output / 'summary/repeat_summary.csv')
            timing = summary[(summary.model == 'lstm') & (summary.metric == 'training_seconds')].iloc[0]
            self.assertEqual(timing.repeat_count, 2)
            self.assertAlmostEqual(timing['mean'], lstms.training_seconds.mean())
            self.assertAlmostEqual(timing['std'], lstms.training_seconds.std(ddof=1))
            (root / 'campaign.json').write_text(json.dumps({
                'settings': {'models': ['lstm', 'chronos2', 'seasonal-naive'], 'horizons': [4], 'repeats': 2},
                'jobs': [{'dataset': 'etth1', 'series': 'etth1', 'output': 'run'}]}))
            self.assertTrue(summarize_campaign(root))
            combined = pd.read_csv(root / 'per_series_metrics.csv')
            self.assertEqual(len(combined), 6)
            self.assertEqual(len(pd.read_csv(root / 'coverage.csv')), 6)
            report = pd.read_csv(root / 'repeat_summary.csv')
            timing = report[(report.model == 'lstm') & (report.metric == 'training_seconds')].iloc[0]
            self.assertEqual(timing.repeat_count, 2)
            self.assertAlmostEqual(timing['std'], lstms.training_seconds.std(ddof=1))

    def test_campaign_keeps_customer_window_and_repeat_variation_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = []
            for series, means in [('A', [1., 3.]), ('B', [5., 9.])]:
                folder = root / series
                folder.mkdir()
                jobs.append({'dataset': 'electricity', 'series': series, 'output': series})
                rows = []
                for repeat, mean in enumerate(means, 1):
                    row = dict(model='seasonal-naive', horizon=24, seed=np.nan, repeat=repeat, origins=2,
                               full_batches_timed=2, mean_batch_latency_ms=1., std_batch_latency_ms=.1,
                               training_seconds=mean, setup_seconds_including_training=mean)
                    for metric in ('mae', 'rmse', 'mase'):
                        row[f'{metric}_window_mean'] = mean
                        row[f'{metric}_window_std'] = .5
                    rows.append(row)
                pd.DataFrame(rows).to_csv(folder / 'metrics.csv', index=False)
            (root / 'campaign.json').write_text(json.dumps({
                'settings': {'models': ['seasonal-naive'], 'horizons': [24], 'repeats': 2}, 'jobs': jobs}))
            self.assertTrue(summarize_campaign(root))
            summary = pd.read_csv(root / 'dataset_summary.csv')
            mae = summary[summary.metric == 'mae'].sort_values('repeat')
            np.testing.assert_allclose(mae['mean'], [3., 6.])
            self.assertTrue(mae.observation_count.eq(4).all())
            repeat_summary = pd.read_csv(root / 'dataset_repeat_summary.csv')
            item = repeat_summary[(repeat_summary.metric == 'mae') & (repeat_summary.statistic == 'mean')].iloc[0]
            self.assertAlmostEqual(item['mean'], 4.5)
            self.assertAlmostEqual(item['std'], np.std([3., 6.], ddof=1))
            costs = pd.read_csv(root / 'resource_summary.csv')
            self.assertTrue(costs.series_count.eq(2).all())
            # Missing B's second repetition excludes B from BOTH repeat comparisons.
            second = pd.read_csv(root / 'B/metrics.csv')
            second.iloc[:1].to_csv(root / 'B/metrics.csv', index=False)
            self.assertFalse(summarize_campaign(root))
            summary = pd.read_csv(root / 'dataset_summary.csv')
            mae = summary[summary.metric == 'mae'].sort_values('repeat')
            np.testing.assert_allclose(mae['mean'], [1., 3.])
            self.assertTrue(mae.eligible_series.eq(1).all())
            per_series = pd.read_csv(root / 'repeat_summary.csv')
            item = per_series[(per_series.series == 'B') & (per_series.metric == 'training_seconds')].iloc[0]
            self.assertEqual(item.repeat_count, 1)
            self.assertEqual(item.requested_repeats, 2)
            self.assertTrue(pd.isna(item['std']))
