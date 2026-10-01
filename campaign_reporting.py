"""Comparable campaign summaries with explicit window and series variability."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from repetition_reporting import REPEAT_METRICS, repetition_summary


def combine_moments(counts, means, stds):
    """Pool sample moments, including between-group variation (never average stds)."""
    counts, means, stds = [np.asarray(x, dtype=float) for x in (counts, means, stds)]
    valid = (counts > 0) & np.isfinite(means) & ((counts == 1) | np.isfinite(stds))
    counts, means, stds = counts[valid], means[valid], stds[valid]
    n = int(counts.sum())
    if not n:
        return 0, np.nan, np.nan
    mean = float(np.dot(counts, means) / n)
    within = np.where(counts > 1, (counts - 1) * np.nan_to_num(stds)**2, 0)
    variance_sum = (within + counts * (means - mean)**2).sum()
    return n, mean, float(np.sqrt(variance_sum / (n - 1))) if n > 1 else np.nan


def summarize_campaign(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'campaign.json').read_text())
    settings = manifest['settings']
    models, horizons = settings['models'], settings['horizons']
    repeats = settings.get('repeats', 1)
    frames, coverage = [], []
    for job in manifest['jobs']:
        metrics_path = directory / job['output'] / 'metrics.csv'
        frame = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        if not frame.empty:
            if 'repeat' not in frame:
                frame['repeat'] = 1
            if frame.duplicated(['model', 'horizon', 'repeat']).any():
                raise ValueError(f'Duplicate repetition results in {metrics_path}.')
            frame = frame.assign(dataset=job['dataset'], series=job['series'], run_directory=job['output'])
            frames.append(frame)
        for horizon in horizons:
            for model in models:
                for repeat in range(1, repeats + 1):
                    found = (not frame.empty and ((frame.model == model) & (frame.horizon == horizon) &
                                                  (frame['repeat'] == repeat)).any())
                    coverage.append({'dataset': job['dataset'], 'series': job['series'],
                                     'model': model, 'horizon': horizon, 'repeat': repeat, 'completed': bool(found)})
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(directory / 'coverage.csv', index=False)
    summary_columns = ['dataset', 'model', 'horizon', 'repeat', 'metric', 'observation_type',
                       'observation_count', 'mean', 'std', 'series_mean', 'series_std',
                       'metric_series', 'completed_series', 'eligible_series', 'requested_series']
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=['dataset', 'series', 'model', 'horizon', 'seed', 'repeat', 'run_directory'])
    rows.to_csv(directory / 'per_series_metrics.csv', index=False)
    summaries, costs = [], []
    if not rows.empty:
        for (dataset, horizon, repeat), availability in coverage.groupby(['dataset', 'horizon', 'repeat']):
            # One cohort across ALL models and repetitions for a fair repeat comparison.
            all_repeats = coverage[(coverage.dataset == dataset) & (coverage.horizon == horizon)]
            eligible = all_repeats.groupby('series').completed.all()
            names = eligible[eligible].index
            cohort = rows[(rows.dataset == dataset) & (rows.horizon == horizon) &
                          (rows['repeat'] == repeat) & rows.series.isin(names)]
            requested = availability.series.nunique()
            for model in models:
                group = cohort[cohort.model == model]
                completed = int(availability.loc[availability.model == model, 'completed'].sum())
                for metric in ('mae', 'rmse', 'mase', 'batch_latency_ms'):
                    latency = metric == 'batch_latency_ms'
                    mean_col = 'mean_batch_latency_ms' if latency else f'{metric}_window_mean'
                    std_col = 'std_batch_latency_ms' if latency else f'{metric}_window_std'
                    count_col = 'full_batches_timed' if latency else 'origins'
                    means = pd.to_numeric(group[mean_col], errors='coerce')
                    stds = pd.to_numeric(group[std_col], errors='coerce')
                    count, mean, std = combine_moments(group[count_col], means, stds)
                    summaries.append(dict(dataset=dataset, model=model, horizon=horizon, repeat=repeat, metric=metric,
                        observation_type='full inference batch' if latency else 'forecast window',
                        observation_count=count, mean=mean, std=std,
                        series_mean=means.mean(), series_std=means.std(ddof=1),
                        metric_series=int(means.notna().sum()), completed_series=completed,
                        eligible_series=len(names), requested_series=requested))
        # Costs include every successful run, even outside the comparison cohort.
        for (dataset, model, horizon, repeat), group in rows.groupby(['dataset', 'model', 'horizon', 'repeat']):
            for metric in ('training_seconds', 'setup_seconds_including_training', 'inference_seconds',
                           'forecasts_per_second'):
                if metric in group:
                    values = pd.to_numeric(group[metric], errors='coerce')
                    costs.append(dict(dataset=dataset, model=model, horizon=horizon, repeat=repeat, metric=metric,
                                      series_count=int(values.count()), mean=values.mean(), std=values.std(ddof=1)))
    pd.DataFrame(summaries, columns=summary_columns).to_csv(directory / 'dataset_summary.csv', index=False)
    pd.DataFrame(costs, columns=['dataset', 'model', 'horizon', 'repeat', 'metric', 'series_count', 'mean', 'std']).to_csv(
        directory / 'resource_summary.csv', index=False)
    repetition_summary(rows, ['dataset', 'series', 'model', 'horizon', 'seed'], REPEAT_METRICS, repeats).to_csv(
        directory / 'repeat_summary.csv', index=False)
    dataset_rows = pd.DataFrame(summaries, columns=summary_columns).rename(columns={'metric': 'measure'})
    repetition_summary(dataset_rows, ['dataset', 'model', 'horizon', 'measure'], ['mean', 'series_mean'], repeats).rename(
        columns={'metric': 'statistic', 'measure': 'metric'}).to_csv(directory / 'dataset_repeat_summary.csv', index=False)
    (directory / 'SUMMARY_README.txt').write_text(
        'dataset_summary.csv contains one result per repetition, comparing the same series across all requested models/repetitions at each horizon.\n'
        'Series missing any requested model/repetition are excluded for that horizon; coverage.csv lists every expected result.\n'
        'mean/std describe individual forecast-window errors (or full inference-batch latency), pooled across eligible series.\n'
        'series_mean/series_std describe the distribution of per-series means, giving customers equal weight.\n'
        'metric_series excludes undefined metrics, e.g. MASE with zero training scale. Counts are explicit.\n'
        'All std use ddof=1; fewer than two observations give an empty std, not zero.\n'
        'Overlapping or serially correlated windows mean std is descriptive, not a confidence interval.\n'
        'Mean window RMSE differs from pooled RMSE, retained in per_series_metrics.csv.\n'
        'MAE/RMSE retain each target\'s units; datasets are never pooled. MASE uses training-only lag --season.\n'
        'resource_summary.csv gives mean/std across successful series WITHIN each repetition, including runs outside the matched cohort.\n'
        'Latency excludes setup/training and uses full batches only. Training cost includes CV and final fitting.\n'
        'repeat_summary.csv gives mean/std across repeated setups/fits per series; fixed seed, with completed/requested counts.\n'
        'dataset_repeat_summary.csv gives mean/std across repetitions of the matched-cohort dataset means.\n'
        'Window std, customer std, and repetition std measure different variation and are never pooled together.\n'
        'LSTM workload includes all executed optimization/early-stopping batches, even epochs after the best checkpoint.\n'
        'Diagnostic evaluation after fitting is excluded from fit batch counts. Initial/refit window counts overlap and are not summed as unique windows.\n'
        'Setup is repeated in-process; downloads, imports and filesystem caches can make the first repetition slower.\n'
        'Raw forecasts, per-origin metrics, run metadata and CV artifacts live in each job directory.\n')
    return coverage.completed.all()
