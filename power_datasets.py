"""UCI power data preparation. Output is hourly, with gaps preserved as NaNs."""
from pathlib import Path
import hashlib
import json
import shutil
import urllib.request
import zipfile

import numpy as np
import pandas as pd

SOURCES = {
    'electricity': {
        'url': 'https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip',
        'member': 'LD2011_2014.txt', 'doi': '10.24432/C58C86',
    },
    'household': {
        'url': 'https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip',
        'member': 'household_power_consumption.txt', 'doi': '10.24432/C58K54',
    },
}
HOUSEHOLD_FEATURES = ['Global_active_power', 'Global_reactive_power', 'Voltage', 'Global_intensity',
                      'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']


def hourly_aggregate(chunks, dataset):
    """Require every source sample in each hour; never fill missing values."""
    sums, counts = [], []
    previous = None
    expected_step = np.timedelta64(15 if dataset == 'electricity' else 1, 'm')
    source_rows = 0
    for chunk in chunks:
        if dataset == 'electricity':
            dates = pd.to_datetime(chunk.iloc[:, 0], format='%Y-%m-%d %H:%M:%S')
            numeric = chunk.iloc[:, 1:].apply(pd.to_numeric, errors='raise')
            # Electricity labels denote ends of 15-minute intervals. Assign the
            # four intervals ending 00:15..01:00 to the hour starting at 00:00.
            hours = (dates - np.timedelta64(15, 'm')).dt.floor('h')
        else:
            dates = pd.to_datetime(chunk['Date'] + ' ' + chunk['Time'], format='%d/%m/%Y %H:%M:%S')
            numeric = chunk[HOUSEHOLD_FEATURES].apply(pd.to_numeric, errors='raise')
            hours = dates.dt.floor('h')
        if dates.isna().any() or not dates.diff().iloc[1:].eq(expected_step).all():
            raise ValueError('Source timestamps must be ordered, unique and on the expected regular grid.')
        if previous is not None and dates.iloc[0] - previous != expected_step:
            raise ValueError('Non-contiguous source chunks.')
        previous = dates.iloc[-1]
        source_rows += len(chunk)
        numeric.index = pd.DatetimeIndex(hours)
        sums.append(numeric.groupby(level=0).sum())
        counts.append(numeric.groupby(level=0).count())
    total = pd.concat(sums).groupby(level=0).sum()
    count = pd.concat(counts).groupby(level=0).sum()
    expected_count = 4 if dataset == 'electricity' else 60
    hourly = (total / expected_count).where(count == expected_count)
    if dataset == 'household':
        # Minute submeter readings are Wh; sum them to hourly Wh, not average.
        hourly[HOUSEHOLD_FEATURES[4:]] *= 60
    hourly = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq='h'))
    if dataset == 'electricity':
        # Never interpret pre-activation placeholders as observed zero demand.
        active = (hourly.notna() & hourly.ne(0)).cummax()
        hourly = hourly.where(active)
        # The source encodes DST using artificial zeros/combined measurements.
        # Conservatively mask both whole transition days, preserving the clock grid.
        idx = hourly.index
        dst_day = idx.month.isin([3, 10]) & (idx.dayofweek == 6) & (idx.day >= 25)
        hourly.loc[dst_day] = np.nan
    hourly.index.name = 'date'
    return hourly, source_rows


def prepare(dataset, directory, download=False):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f'{dataset}.zip'
    source = SOURCES[dataset]
    if not archive.exists():
        if not download:
            raise FileNotFoundError(f'{archive} missing; use --download.')
        temporary = archive.with_suffix('.zip.part')
        print(f'Downloading {dataset} from UCI...', flush=True)
        try:
            with urllib.request.urlopen(source['url'], timeout=60) as response, temporary.open('wb') as target:
                shutil.copyfileobj(response, target)
            with zipfile.ZipFile(temporary) as z:
                if source['member'] not in z.namelist():
                    raise ValueError('Unexpected UCI archive contents.')
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    output = directory / f'{dataset}_hourly.csv'
    manifest_path = directory / f'{dataset}_preprocessing.json'
    with archive.open('rb') as handle:
        source_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if output.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('source_sha256') == source_hash and manifest.get('preprocessor_sha256') == script_hash:
            return pd.read_csv(output, index_col='date', parse_dates=['date']), manifest
    print(f'Preparing hourly {dataset}...', flush=True)
    with zipfile.ZipFile(archive) as z, z.open(source['member']) as handle:
        chunks = pd.read_csv(handle, sep=';', decimal=',' if dataset == 'electricity' else '.',
                             na_values=['?'], chunksize=100_000)
        hourly, source_rows = hourly_aggregate(chunks, dataset)
    hourly.to_csv(output)
    manifest = {**source, 'dataset': dataset, 'license': 'CC BY 4.0',
                'source_sha256': source_hash, 'preprocessor_sha256': script_hash,
                'source_rows': source_rows, 'hourly_rows': len(hourly),
                'start': str(hourly.index[0]), 'end': str(hourly.index[-1]),
                'missing_hours_per_column': hourly.isna().sum().to_dict(),
                'coverage_threshold': '100%: four quarter-hours or sixty minutes per hour',
                'imputation': 'none; incomplete context or target windows excluded',
                'electricity_policy': 'mask leading zeros and whole March/October DST transition days',
                'units': 'mean power kW; household submeter channels are summed Wh',
                'hour_labels': 'interval start; electricity raw interval-end labels shifted back 15 minutes'}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    return hourly, manifest
