"""Comparable campaign summaries with explicit window and series variability."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


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
    frames, coverage = [], []
    for job in manifest['jobs']:
        metrics_path = directory / job['output'] / 'metrics.csv'
        frame = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
        if not frame.empty:
            frame = frame.assign(dataset=job['dataset'], series=job['series'], run_directory=job['output'])
            frames.append(frame)
        for horizon in horizons:
            for model in models:
                found = (not frame.empty and ((frame.model == model) & (frame.horizon == horizon)).any())
                coverage.append({'dataset': job['dataset'], 'series': job['series'],
                                 'model': model, 'horizon': horizon, 'completed': bool(found)})
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(directory / 'coverage.csv', index=False)
    summary_columns = ['dataset', 'model', 'horizon', 'metric', 'observation_type',
                       'observation_count', 'mean', 'std', 'series_mean', 'series_std',
                       'metric_series', 'completed_series', 'eligible_series', 'requested_series']
    rows = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=['dataset', 'series', 'model', 'horizon', 'seed', 'run_directory'])
    rows.to_csv(directory / 'per_series_metrics.csv', index=False)
    summaries, costs = [], []
    if not rows.empty:
        for (dataset, horizon), availability in coverage.groupby(['dataset', 'horizon']):
            eligible = availability.groupby('series').completed.all()
            names = eligible[eligible].index
            cohort = rows[(rows.dataset == dataset) & (rows.horizon == horizon) & rows.series.isin(names)]
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
                    summaries.append(dict(dataset=dataset, model=model, horizon=horizon, metric=metric,
                        observation_type='full inference batch' if latency else 'forecast window',
                        observation_count=count, mean=mean, std=std,
                        series_mean=means.mean(), series_std=means.std(ddof=1),
                        metric_series=int(means.notna().sum()), completed_series=completed,
                        eligible_series=len(names), requested_series=requested))
        # Costs include every successful run, even outside the comparison cohort.
        for (dataset, model, horizon), group in rows.groupby(['dataset', 'model', 'horizon']):
            for metric in ('training_seconds', 'setup_seconds_including_training', 'inference_seconds',
                           'forecasts_per_second'):
                if metric in group:
                    values = pd.to_numeric(group[metric], errors='coerce')
                    costs.append(dict(dataset=dataset, model=model, horizon=horizon, metric=metric,
                                      series_count=int(values.count()), mean=values.mean(), std=values.std(ddof=1)))
    pd.DataFrame(summaries, columns=summary_columns).to_csv(directory / 'dataset_summary.csv', index=False)
    pd.DataFrame(costs, columns=['dataset', 'model', 'horizon', 'metric', 'series_count', 'mean', 'std']).to_csv(
        directory / 'resource_summary.csv', index=False)
    (directory / 'SUMMARY_README.txt').write_text(
        'dataset_summary.csv compares the same series across all requested models at each horizon.\n'
        'Series missing any requested model are excluded for that horizon; coverage.csv lists every expected result.\n'
        'mean/std describe individual forecast-window errors (or full inference-batch latency), pooled across eligible series.\n'
        'series_mean/series_std describe the distribution of per-series means, giving customers equal weight.\n'
        'metric_series excludes undefined metrics, e.g. MASE with zero training scale. Counts are explicit.\n'
        'All std use ddof=1; fewer than two observations give an empty std, not zero.\n'
        'Overlapping or serially correlated windows mean std is descriptive, not a confidence interval.\n'
        'Mean window RMSE differs from pooled RMSE, retained in per_series_metrics.csv.\n'
        'MAE/RMSE retain each target\'s units; datasets are never pooled. MASE uses training-only lag --season.\n'
        'resource_summary.csv gives mean/std across successful series, including runs outside the matched cohort.\n'
        'Latency excludes setup/training and uses full batches only. Training cost includes CV and final fitting.\n'
        'One LSTM seed is used; these are not across-seed standard deviations.\n'
        'Raw forecasts, per-origin metrics, run metadata and CV artifacts live in each job directory.\n')
    return coverage.completed.all()
