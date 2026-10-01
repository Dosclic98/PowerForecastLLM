#!/usr/bin/env python3
"""Run UCI electricity/household evaluations using the shared ETTh1 model adapters.

Example:
  python benchmark_power.py --dataset household --download --models seasonal-naive
  python benchmark_power.py --dataset electricity --download --clients MT_001 MT_002 --models lstm --device cuda
Remaining arguments are forwarded to benchmark_etth1.py. Defaults: horizons 24/168,
horizon-length seasonal-naive, LSTM and foundation models; 80/20 development/test split and one chronological early-stopping holdout.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

import benchmark_etth1 as engine
from power_datasets import SOURCES, HOUSEHOLD_FEATURES, prepare
from campaign_datasets import DEFAULT_CLIENTS, split_ends


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', required=True, choices=list(SOURCES))
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--data-dir', type=Path, default=Path('data/power'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--clients', nargs='+', help='Electricity client IDs. Default: fixed subset of 10 early-active clients, evaluated separately.')
    parser.add_argument('--prepare-only', action='store_true')
    args, forwarded = parser.parse_known_args(argv)
    reserved = {'--data', '--target', '--feature-columns', '--split-ends', '--dataset-name',
                '--allow-missing', '--summarize'}
    if any(token.split('=')[0] in reserved for token in forwarded):
        parser.error('Dataset input, target, splits and missing-data policy are managed by this runner.')
    if args.clients and args.dataset != 'electricity':
        parser.error('--clients applies only to electricity.')
    if '--models' not in forwarded and not any(x.startswith('--models=') for x in forwarded):
        forwarded += ['--models', 'seasonal-naive', 'lstm', *engine.CHECKPOINTS]
    if '--horizons' not in forwarded and not any(x.startswith('--horizons=') for x in forwarded):
        forwarded += ['--horizons', '24', '168']
    # Validate model options before a potentially large download.
    model_args = engine.parse_args(forwarded)
    hourly, manifest = prepare(args.dataset, args.data_dir, args.download)
    series = list(dict.fromkeys(args.clients or DEFAULT_CLIENTS)) if args.dataset == 'electricity' else ['household']
    if args.dataset == 'electricity' and any(name not in hourly for name in series):
        parser.error('Unknown electricity client; use IDs such as MT_001.')
    if args.prepare_only:
        print(json.dumps(manifest, indent=2))
        return 0
    # Keep dates common across clients, and round boundaries down to whole days.
    ends = split_ends(len(hourly))
    if model_args.lstm_plan:
        planned = engine.parse_args(['--split-ends', *map(str, ends), *forwarded])
        per_series = (len(engine.lstm_grid(planned)) * planned.cv_folds + 1) * len(planned.horizons) * planned.repeats
        print(json.dumps({'dataset': args.dataset, 'series': series, 'series_count': len(series),
                          'lstm_fits_per_series': per_series, 'total_lstm_fits': per_series * len(series),
                          'folds': engine.cross_validation_folds(planned),
                          'grid': engine.lstm_grid(planned)}, indent=2))
        return 0
    output = args.output or Path('results') / (args.dataset + datetime.now(timezone.utc).strftime('_%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    engine.write_json(output / 'dataset.json', {**manifest, 'series': series, 'split_ends': ends,
                                              'forwarded_arguments': forwarded})
    metrics, failures = [], []
    for processed, name in enumerate(series, start=1):
        frame = hourly[[name]] if args.dataset == 'electricity' else hourly[HOUSEHOLD_FEATURES]
        source_csv = output / f'{name}_input.csv'
        frame.to_csv(source_csv)
        command_args = ['--data', str(source_csv), '--target', frame.columns[0],
                        '--feature-columns', *frame.columns, '--dataset-name', args.dataset,
                        '--split-ends', *map(str, ends), '--allow-missing',
                        '--output', str(output / name), *forwarded]
        result = subprocess.run([sys.executable, str(Path(engine.__file__).resolve()), *command_args])
        if result.returncode:
            failures.append(name)
        metrics_path = output / name / 'metrics.csv'
        if metrics_path.exists():
            metrics.append(pd.read_csv(metrics_path).assign(series=name, dataset=args.dataset))
        # Save incremental results for long panel runs.
        if metrics:
            combined = pd.concat(metrics, ignore_index=True)
            combined.to_csv(output / 'per_series_metrics.csv', index=False)
            from repetition_reporting import REPEAT_METRICS, repetition_summary
            repetition_summary(combined, ['dataset', 'series', 'model', 'horizon', 'seed'],
                               REPEAT_METRICS, model_args.repeats).to_csv(output / 'repeat_summary.csv', index=False)
            keys = ['dataset', 'model', 'horizon', 'seed', 'repeat']
            summary = combined.groupby(keys, dropna=False).agg(
                mae=('mae', 'mean'), rmse=('rmse', 'mean'), mase=('mase', 'mean'),
                completed_series=('series', 'nunique'), mase_series=('mase', 'count')).reset_index()
            summary['requested_series'] = len(series)
            summary.to_csv(output / 'macro_metrics.csv', index=False)
        engine.write_json(output / 'status.json', {'failed_series': failures, 'processed_series': processed,
                                                   'status': ('partial_failure' if failures else 'complete')
                                                       if processed == len(series) else 'running',
                                                   'requested_series': len(series)})
    print(f'Results: {output.resolve()}')
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
