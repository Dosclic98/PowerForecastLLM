#!/usr/bin/env python3
"""Run a common forecasting experiment on ETTh1, electricity and household.

All model/training options from benchmark_etth1.py are accepted and applied to
all datasets. Campaign defaults: univariate, horizons 24/168, context 512,
stride 24, 80/20 development/test split, one early-stopping holdout, ten electricity clients.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import benchmark_etth1 as engine
from campaign_datasets import DATASETS, DEFAULT_CLIENTS, prepare_series
from campaign_reporting import summarize_campaign


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Model options: --models, --horizons, --context, --stride, --device, --epochs,\n'
               '--cv-folds, --lstm-*, --max-origins, etc. See benchmark_etth1.py --help.\n'
               'Examples:\n'
               '  python benchmark_campaign.py --download --device cuda\n'
               '  python benchmark_campaign.py --plan\n'
               '  python benchmark_campaign.py --models seasonal-naive --clients MT_001 --max-origins 3')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--etth1-data', type=Path, help='Default: DATA_DIR/ETTh1.csv.')
    parser.add_argument('--clients', nargs='+', help='Electricity IDs; default fixed subset of 10 early-active customers.')
    parser.add_argument('--download', action='store_true')
    parser.add_argument('--output', type=Path, help='New campaign directory; never overwrite an existing run.')
    parser.add_argument('--plan', action='store_true', help='Print settings and fit budget without downloads or execution.')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare data and save the validated job manifest.')
    parser.add_argument('--summarize', type=Path, help='Rebuild campaign summaries from saved job results.')
    args, forwarded = parser.parse_known_args(argv)
    reserved = {'--data', '--target', '--feature-columns', '--split-ends', '--dataset-name',
                '--allow-missing', '--lstm-plan'}
    if any(token.split('=')[0] in reserved for token in forwarded):
        parser.error('The campaign manages dataset inputs and splits; use --plan for its budget.')
    if args.clients and 'electricity' not in args.datasets:
        parser.error('--clients requires electricity in --datasets.')
    if args.summarize and (args.plan or args.prepare_only or args.output):
        parser.error('--summarize cannot be combined with --plan, --prepare-only or --output.')
    args.datasets = list(dict.fromkeys(args.datasets))
    specified = {token.split('=')[0] for token in forwarded if token.startswith('--')}
    for option, defaults in (('--horizons', ['24', '168']), ('--input-mode', ['univariate'])):
        if option not in specified:
            forwarded.extend([option, *defaults])
    settings = engine.parse_args(forwarded)
    args.etth1_data = args.etth1_data or args.data_dir / 'ETTh1.csv'
    return args, forwarded, settings


def plan(args, settings):
    counts = {dataset: (len(set(args.clients)) if args.clients else len(DEFAULT_CLIENTS)) if dataset == 'electricity' else 1
              for dataset in args.datasets}
    fits = (len(engine.lstm_grid(settings)) * settings.cv_folds + 1) if 'lstm' in settings.models else 0
    return {'datasets': counts, 'models': settings.models, 'horizons': settings.horizons,
            'input_mode': settings.input_mode, 'context': settings.context, 'stride': settings.stride,
            'split': '80% development / 20% test; development end rounded down to a whole day',
            'validation_protocol': 'single_holdout' if settings.cv_folds == 1 else 'expanding_window_cv',
            'validation_fits_per_configuration': settings.cv_folds,
            'lstm_configurations': engine.lstm_grid(settings) if 'lstm' in settings.models else [],
            'electricity_clients': list(dict.fromkeys(args.clients or DEFAULT_CLIENTS)) if 'electricity' in args.datasets else [],
            'refit': ('full development period; epochs = best holdout epoch' if settings.cv_folds == 1 else
                      'full development period; epochs = ceiling of median best CV epoch'),
            'mase_lag': settings.season, 'seed': settings.seeds[0],
            'model_evaluations': sum(counts.values()) * len(settings.models) * len(settings.horizons),
            'lstm_fits_per_series_per_horizon': fits,
            'total_lstm_fits': fits * sum(counts.values()) * len(settings.horizons),
            'smoke_test': settings.max_origins is not None}


def main(argv=None):
    args, forwarded, settings = parse_args(argv)
    if args.summarize:
        complete = summarize_campaign(args.summarize)
        print(f'Summaries: {args.summarize.resolve()}')
        return 0 if complete else 1
    budget = plan(args, settings)
    print(json.dumps(budget, indent=2), flush=True)
    if args.plan:
        return 0
    if 'toto2' in settings.models and sys.version_info < (3, 12):
        raise ValueError('Toto 2 requires Python 3.12+; use the benchmark environment.')
    output = (args.output or Path('results') / datetime.now(timezone.utc).strftime('campaign_%Y%m%dT%H%M%S_%fZ')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {'status': 'preparing', 'protocol_version': 3, 'plan': budget,
                'settings': vars(settings), 'forwarded_arguments': forwarded, 'jobs': [],
                'code_sha256': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ('benchmark_campaign.py', 'campaign_datasets.py',
                                             'campaign_reporting.py', 'benchmark_etth1.py', 'power_datasets.py')}}
    engine.write_json(output / 'campaign.json', manifest)
    try:
        series, sources = prepare_series(args.datasets, args.data_dir, args.etth1_data,
                                         args.clients, args.download, settings.input_mode, output)
        manifest['sources'] = sources
        for item in series:
            relative = str(Path('runs') / item.dataset / item.name)
            command_args = [*forwarded, *item.arguments(), '--output', str(output / relative)]
            # Validate every dataset's real boundaries before starting any expensive fit.
            engine.parse_args(command_args)
            manifest['jobs'].append({**asdict(item), 'series': item.name, 'output': relative,
                                     'arguments': command_args, 'status': 'pending'})
    except (Exception, SystemExit) as exc:
        manifest.update(status='preparation_failed', error=f'{type(exc).__name__}: {exc}')
        engine.write_json(output / 'campaign.json', manifest)
        raise
    manifest['status'] = 'prepared' if args.prepare_only else 'running'
    engine.write_json(output / 'campaign.json', manifest)
    if args.prepare_only:
        print(f'Prepared campaign: {output}')
        return 0
    failed = False
    for index, job in enumerate(manifest['jobs'], 1):
        print(f"Campaign job {index}/{len(series)}: {job['dataset']}/{job['series']}", flush=True)
        job['status'] = 'running'
        engine.write_json(output / 'campaign.json', manifest)
        command = [sys.executable, str(Path(engine.__file__).resolve()), *job['arguments']]
        try:
            result = subprocess.run(command, check=False)
            job.update(returncode=result.returncode, status='complete' if result.returncode == 0 else 'failed')
            failed |= result.returncode != 0
        except OSError as exc:
            job.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            failed = True
        engine.write_json(output / 'campaign.json', manifest)
    complete = summarize_campaign(output)
    manifest['status'] = 'complete' if complete and not failed else 'partial_failure'
    engine.write_json(output / 'campaign.json', manifest)
    print(f'Campaign results: {output}')
    return 0 if manifest['status'] == 'complete' else 1


if __name__ == '__main__':
    sys.exit(main())
