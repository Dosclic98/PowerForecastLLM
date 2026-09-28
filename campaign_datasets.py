"""Dataset preparation and the common chronological campaign split."""
from dataclasses import dataclass
import hashlib
from pathlib import Path

import pandas as pd

import benchmark_etth1 as engine
from power_datasets import HOUSEHOLD_FEATURES, prepare

DATASETS = ('etth1', 'electricity', 'household')

# Fixed, reproducible subset with observations in the initial 2011 training
# period. Selected for early availability, without inspecting test targets/scores.
DEFAULT_CLIENTS = [f'MT_{i:03d}' for i in (124, 131, 132, 156, 158, 159, 161, 162, 163, 166)]




@dataclass
class Series:
    dataset: str
    name: str
    path: Path
    features: list[str]
    split_ends: list[int]

    def arguments(self):
        return ['--data', str(self.path), '--target', self.features[0],
                '--feature-columns', *self.features, '--split-ends', *map(str, self.split_ends),
                '--dataset-name', self.dataset, '--allow-missing']


def split_ends(length):
    """80% development / 20% test; round the development end down to a whole day."""
    return [int(length * .8) // 24 * 24, length]


def prepare_series(datasets, data_dir, etth1_data, clients, download, input_mode, output):
    series, manifests = [], {}
    inputs = output / 'inputs'
    inputs.mkdir()
    for dataset in datasets:
        if dataset == 'etth1':
            # Use the existing validated download/loader, then the entire hourly record.
            features = ['OT'] if input_mode == 'univariate' else engine.FEATURES
            engine.load_data(etth1_data, download, features=features, test_end=1)
            frame = pd.read_csv(etth1_data)
            path = inputs / 'etth1.csv'
            frame[['date', *features]].to_csv(path, index=False)
            series.append(Series(dataset, dataset, path, features, split_ends(len(frame))))
            manifests[dataset] = {'source': str(etth1_data), 'url': engine.DATA_URL,
                                  'source_sha256': hashlib.sha256(etth1_data.read_bytes()).hexdigest(),
                                  'hourly_rows': len(frame), 'target_units': 'oil temperature (degrees C)'}
            continue
        hourly, manifest = prepare(dataset, data_dir / 'power', download)
        manifests[dataset] = manifest
        names = list(dict.fromkeys(clients or DEFAULT_CLIENTS)) if dataset == 'electricity' else ['household']
        if dataset == 'electricity' and any(name not in hourly for name in names):
            raise ValueError('Unknown electricity client; use IDs such as MT_001.')
        for name in names:
            features = [name] if dataset == 'electricity' else HOUSEHOLD_FEATURES
            if input_mode == 'univariate':
                features = features[:1]
            path = inputs / f'{dataset}_{name}.csv'
            hourly[features].to_csv(path)
            series.append(Series(dataset, name, path, features, split_ends(len(hourly))))
    return series, manifests
