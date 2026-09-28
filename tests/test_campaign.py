import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

import benchmark_campaign as campaign
from campaign_datasets import Series, split_ends
from campaign_reporting import combine_moments, summarize_campaign


class CampaignTests(unittest.TestCase):
    def test_shared_defaults_and_budget(self):
        args, _, settings = campaign.parse_args([])
        plan = campaign.plan(args, settings)
        self.assertEqual(settings.horizons, [24, 168])
        self.assertEqual(settings.input_mode, 'univariate')
        self.assertEqual(settings.cv_folds, 1)
        self.assertEqual(len(plan['electricity_clients']), 10)
        self.assertEqual(plan['datasets']['electricity'], 10)
        self.assertEqual(plan['model_evaluations'], 120)
        self.assertEqual(plan['total_lstm_fits'], 48)
        self.assertEqual(split_ends(10001), [7992, 10001])
        with patch.object(campaign, 'prepare_series', side_effect=AssertionError('plan must not prepare')):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(campaign.main(['--plan']), 0)

    def test_managed_arguments_cannot_be_overridden_or_abbreviated(self):
        for arguments in (['--split-ends', '1000', '2000', '3000'], ['--split', '1000', '2000', '3000']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                campaign.parse_args(arguments)

    def test_pooled_variance_matches_individual_observations(self):
        groups = [np.array([1., 4., 9.]), np.array([20.]), np.array([2., 3.])]
        counts = [len(x) for x in groups]
        means = [x.mean() for x in groups]
        stds = [x.std(ddof=1) if len(x) > 1 else np.nan for x in groups]
        count, mean, std = combine_moments(counts, means, stds)
        all_values = np.concatenate(groups)
        self.assertEqual(count, len(all_values))
        self.assertAlmostEqual(mean, all_values.mean())
        self.assertAlmostEqual(std, all_values.std(ddof=1))
        self.assertTrue(np.isnan(combine_moments([1], [7], [np.nan])[2]))
        self.assertEqual(combine_moments([2], [np.nan], [np.nan])[0], 0)

    def test_three_dataset_campaign_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dates = pd.date_range('2000-01-01', periods=4800, freq='h', name='date')
            values = np.sin(np.arange(len(dates)) / 19) + np.arange(len(dates)) / 2000
            pd.DataFrame({'date': dates, 'OT': values}).to_csv(root / 'ETTh1.csv', index=False)

            def prepared(dataset, *_args):
                target = 'MT_001' if dataset == 'electricity' else 'Global_active_power'
                return pd.DataFrame({target: values}, index=dates), {'dataset': dataset}

            output = root / 'campaign'
            with patch('campaign_datasets.prepare', side_effect=prepared), contextlib.redirect_stdout(io.StringIO()):
                result = campaign.main(['--data-dir', str(root), '--output', str(output),
                                        '--models', 'seasonal-naive', '--clients', 'MT_001', '--max-origins', '3'])
            self.assertEqual(result, 0)
            manifest = json.loads((output / 'campaign.json').read_text())
            self.assertEqual(manifest['status'], 'complete')
            self.assertEqual(len(manifest['jobs']), 3)
            for job in manifest['jobs']:
                self.assertEqual(job['split_ends'], [3840, 4800])
                self.assertEqual(job['status'], 'complete')
                for horizon in (24, 168):
                    predictions = pd.read_csv(output / job['output'] / f'seasonal-naive_h{horizon}_predictions.csv')
                    np.testing.assert_allclose(predictions.prediction[:horizon], values[3840-horizon:3840], rtol=1e-6)
            rows = pd.read_csv(output / 'per_series_metrics.csv')
            self.assertEqual(len(rows), 6)
            summary = pd.read_csv(output / 'dataset_summary.csv')
            self.assertEqual(len(summary), 24)
            self.assertTrue(summary.series_std.isna().all())
            for row in rows.itertuples():
                for metric in ('mae', 'rmse', 'mase'):
                    item = summary[(summary.dataset == row.dataset) & (summary.horizon == row.horizon) &
                                   (summary.metric == metric)].iloc[0]
                    self.assertEqual(item.observation_count, 3)
                    self.assertAlmostEqual(item['mean'], getattr(row, f'{metric}_window_mean'))
                    self.assertAlmostEqual(item['std'], getattr(row, f'{metric}_window_std'))
            # A missing requested model excludes the series for both methods, visibly.
            manifest['settings']['models'].append('lstm')
            (output / 'campaign.json').write_text(json.dumps(manifest))
            self.assertFalse(summarize_campaign(output))
            incomplete = pd.read_csv(output / 'dataset_summary.csv')
            self.assertTrue(incomplete.eligible_series.eq(0).all())
            self.assertTrue(incomplete['mean'].isna().all())

    def test_failed_job_does_not_stop_later_datasets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'campaign'
            jobs = [Series(dataset, dataset, root / f'{dataset}.csv', ['target'], [3840, 4800])
                    for dataset in ('etth1', 'electricity', 'household')]
            results = [OSError('simulated launch failure'),
                       type('Result', (), {'returncode': 1})(),
                       type('Result', (), {'returncode': 0})()]
            with patch.object(campaign, 'prepare_series', return_value=(jobs, {})), \
                    patch.object(campaign.subprocess, 'run', side_effect=results) as run, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = campaign.main(['--output', str(output), '--models', 'seasonal-naive'])
            self.assertEqual(result, 1)
            self.assertEqual(run.call_count, 3)
            manifest = json.loads((output / 'campaign.json').read_text())
            self.assertEqual(manifest['status'], 'partial_failure')
            self.assertEqual([job['status'] for job in manifest['jobs']], ['failed', 'failed', 'complete'])
            self.assertFalse(pd.read_csv(output / 'coverage.csv').completed.any())

    def test_all_failed_campaign_still_has_coverage_and_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'campaign.json').write_text(json.dumps({
                'settings': {'models': ['seasonal-naive'], 'horizons': [24]},
                'jobs': [{'dataset': 'etth1', 'series': 'etth1', 'output': 'missing'}]}))
            self.assertFalse(summarize_campaign(root))
            self.assertFalse(pd.read_csv(root / 'coverage.csv').completed.any())
            self.assertTrue(pd.read_csv(root / 'dataset_summary.csv').empty)
