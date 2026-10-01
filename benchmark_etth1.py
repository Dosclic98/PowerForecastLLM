#!/usr/bin/env python3
"""Compare a trained LSTM with frozen time-series foundation models on ETTh1.

The full comparison requires Python 3.12+ because Toto 2 requires it.
Create a new environment, including when an older .venv-benchmark already exists:
    python -m pip install uv
    python -m uv venv --python 3.12 --seed .venv-benchmark
    source .venv-benchmark/bin/activate
    python -m pip install numpy pandas torch chronos-forecasting toto-models
    python -m pip install 'timesfm[torch] @ git+https://github.com/google-research/timesfm.git'

uv downloads Python 3.12 if necessary. LSTM and seasonal-naive can still run on
Python 3.10; explicitly select them with --models lstm seasonal-naive.

Examples:
    python benchmark_etth1.py --download --models seasonal-naive
    python benchmark_etth1.py --download --models lstm --seeds 42 --epochs 5
    python benchmark_etth1.py --download --horizons 24 168 --device cuda
    python benchmark_etth1.py --summarize results/etth1_RUN_DIRECTORY

summary/ contains per-origin accuracy and mean/std across origins for each run.
Each repetition has its own row; --repeats repeats fresh setups/fits with a fixed seed. Standard deviations use
ddof=1 and are undefined for a single observation. Overlapping forecast windows
are correlated: these descriptive standard deviations are not confidence intervals.
Mean window RMSE differs from the pooled RMSE retained in metrics.csv.
metrics.csv also contains *_window_mean and *_window_std for MAE, RMSE and MASE.
summary/repeat_summary.csv reports run-level timing/accuracy mean/std across
repetitions. LSTM counts record optimization batches, stopping batches, and valid
training windows in the initial fit and final refit. Setup caches are not flushed.

Defaults: seven historical input variables, oil-temperature (OT) evaluation,
512 hours of context, horizon 24, daily test origins, one reproducibility seed (42).
The default LSTM is fixed: hidden size 128 per layer, two layers, learning rate
0.001, inter-layer dropout 0.3, zero weight decay, output mode last,
and the requested --context. By default, fit on development data excluding its
last 720 hours, using that chronological holdout for early stopping. Then initialize
a fresh model and train on ALL development data for the best holdout epoch count.
The final 20% is used only for testing. MASE uses the full development period.
--early-stopping-hours (--cv-stop-hours alias) sets the holdout length.
--cv-folds defaults to 1 (single holdout); values >=2 explicitly enable the older
expanding-window CV protocol. Parameter grids require that optional CV mode.
--epochs and --patience control the initial early-stopping fit. --lstm-plan prints
the configuration, validation boundaries and fit count without training.
*_epoch_selection/ holds the first fit's diagnostics; *_selection.json records
its best epoch and final refit metadata. Optional CV writes *_cv_folds.csv and
*_tuning.csv. training_seconds includes all fitting; selected_training_seconds
covers only the final full-development refit. Inference excludes training.
Foundation models use their median (0.5 quantile) forecasts. Both MAE
and RMSE score the SAME predictions, in original target units.
Foundation models jointly forecast the seven channels; only OT is scored.
The LSTM consumes seven channels and predicts OT only. No future load values
are supplied. Foundation-model latency includes forecasting all channels.
Seasonal-naive remains an OT-only baseline. Use --input-mode univariate to
reproduce the previous target-only experiment. Results record the input mode.

The standalone timeline uses 30-day months: 16 development / 4 test; remaining
rows are unused. --split-ends accepts two endpoints: DEVELOPMENT TEST. The campaign
runner instead uses 80/20 of the full record. Pretraining overlap is not audited.

Timing measures the complete adapter call: CPU NumPy input -> CPU NumPy forecast,
including preprocessing and transfers, after warmup, with CUDA synchronization.
Loading, training, metrics and CSV writes are excluded. Batch size 1 measures
request latency; larger batches measure throughput (partial final batches are
excluded from latency percentiles). Every model uses identical test origins.

Sources / APIs checked September 2026:
https://github.com/zhouhaoyi/ETDataset
https://github.com/zhouhaoyi/Informer2020/blob/main/data/data_loader.py
https://huggingface.co/amazon/chronos-2
https://github.com/google-research/timesfm
https://huggingface.co/Datadog/Toto-2.0-313m
TimesFM 3 weights are licensed for non-commercial, non-production use.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import random
import sys
import time
import urllib.request
from datetime import datetime, timezone
from itertools import product
from pathlib import Path


DATA_URL = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
CHECKPOINTS = {
    "chronos2": "amazon/chronos-2",
    "timesfm3": "google/timesfm-3.0-pytorch",
    "toto2": "Datadog/Toto-2.0-313m",
}
VALID_END, TEST_END = 16 * 30 * 24, 20 * 30 * 24
FEATURES = ["OT", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    parser.add_argument("--data", type=Path, default=Path("data/ETTh1.csv"))
    parser.add_argument("--target", default="OT")
    parser.add_argument("--feature-columns", nargs="+", help="Prepared CSV inputs, target must be first.")
    parser.add_argument("--split-ends", nargs=2, type=int, default=[VALID_END, TEST_END],
                        metavar=("DEVELOPMENT", "TEST"), help="Exclusive development/test endpoints (80/20).")
    parser.add_argument("--allow-missing", action="store_true", help="Exclude windows with missing history/targets.")
    parser.add_argument("--dataset-name", default="etth1")
    parser.add_argument("--download", action="store_true", help="Download ETTh1 if --data is missing.")
    parser.add_argument("--input-mode", choices=["multivariate", "univariate"], default="multivariate")
    parser.add_argument("--output", type=Path, help="New output directory (must not already exist).")
    parser.add_argument("--summarize", type=Path,
                        help="Summarize saved metrics/predictions without rerunning models.")
    parser.add_argument("--models", nargs="+", choices=["seasonal-naive", "lstm", *CHECKPOINTS],
                        default=["seasonal-naive", "lstm", *CHECKPOINTS])
    parser.add_argument("--horizons", nargs="+", type=int, default=[24])
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--stride", type=int, default=24, help="Hours between validation/test origins.")
    parser.add_argument("--season", type=int, default=24, help="MASE scaling period; seasonal-naive uses the forecast horizon.")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size for every model.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-origins", type=int, help="Smoke-test limit: first N test origins only.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU thread count.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="One seed reused for every fresh training repetition (default: 42).")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Fresh setup/training repetitions per model/horizon, using the same seed (default: 1).")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-tune-lstm", action="store_true",
                        help="Use --hidden-size and --learning-rate with no weight decay; skip the search.")
    parser.add_argument("--lstm-hidden-sizes", nargs="+", type=int,
                        help="Optional search sizes; default: only --hidden-size (128).")
    parser.add_argument("--lstm-learning-rates", nargs="+", type=float,
                        help="Optional search rates; default: only --learning-rate (0.001).")
    parser.add_argument("--lstm-weight-decays", nargs="+", type=float, default=[0.0],
                        help="L2 regularization strengths; default: 0.")
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=1)
    parser.add_argument("--lstm-plan", action="store_true", help="Print search configurations and exit without training.")
    parser.add_argument("--lstm-contexts", nargs="+", type=int,
                        help="Optional context search; default: only --context.")
    parser.add_argument("--lstm-layers", nargs="+", type=int)
    parser.add_argument("--lstm-dropouts", nargs="+", type=float, default=[0.3],
                        help="Dropout between LSTM layers; inactive with one layer. No output-head dropout.")
    parser.add_argument("--lstm-output-modes", nargs="+", choices=["direct", "last"], default=["last"],
                        help="last centers the target on its last observed value and predicts a residual.")
    parser.add_argument("--lstm-batch-sizes", nargs="+", type=int)
    parser.add_argument("--cv-folds", type=int, default=1,
                        help="Default 1: one early-stopping holdout; >=2 opts into chronological CV.")
    parser.add_argument("--early-stopping-hours", "--cv-stop-hours", dest="cv_stop_hours", type=int, default=30 * 24,
                        help="Early-stopping tail in development data (default: 720 hours).")
    parser.set_defaults(validation_blocks=2)
    parser.add_argument("--validation-stride", type=int,
                        help="Validation origin spacing; default max(--stride, horizon), avoiding overlap.")
    args = parser.parse_args(argv)
    args.valid_end, args.test_end = args.split_ends
    if not 0 < args.valid_end < args.test_end:
        parser.error("Development/test endpoints must be strictly increasing positive indices.")
    if args.cv_folds < 1:
        parser.error("Use 1 for a single holdout or >=2 for chronological CV.")
    args.train_end = (args.valid_end - args.cv_stop_hours if args.cv_folds == 1
                      else args.valid_end // (args.cv_folds + 1))
    args.feature_columns = args.feature_columns or FEATURES
    if args.feature_columns[0] != args.target:
        parser.error("The first feature column must be the target.")
    if args.input_mode == "univariate":
        args.feature_columns = [args.target]
    positive = ["repeats", "context", "stride", "season", "batch_size", "warmup", "threads", "epochs",
                "patience", "hidden_size", "layers", "train_batch_size", "train_stride", "cv_folds", "cv_stop_hours"]
    if any(getattr(args, key) <= 0 for key in positive) or any(h <= 0 for h in args.horizons):
        parser.error("Lengths, counts, horizons and strides must be positive.")
    if not 0 < args.learning_rate < float("inf"):
        parser.error("--learning-rate must be finite and positive.")
    if args.lstm_hidden_sizes and any(size <= 0 for size in args.lstm_hidden_sizes):
        parser.error("--lstm-hidden-sizes must be positive.")
    if args.lstm_learning_rates and any(not 0 < rate < float("inf") for rate in args.lstm_learning_rates):
        parser.error("--lstm-learning-rates must be finite and positive.")
    if any(not 0 <= decay < float("inf") for decay in args.lstm_weight_decays):
        parser.error("--lstm-weight-decays must be finite and nonnegative.")
    if args.max_origins is not None and args.max_origins <= 0:
        parser.error("--max-origins must be positive.")
    if args.context < args.season or args.context + max(args.horizons) > args.valid_end:
        parser.error("Context must cover the season and leave room for training targets.")
    if any(name in args.models for name in ("seasonal-naive", "lstm")) and args.context < max(args.horizons):
        parser.error("Seasonal-naive (including LSTM validation) needs --context at least the largest horizon.")
    if max(args.horizons) > args.test_end - args.valid_end:
        parser.error("Horizon exceeds the validation/test period.")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**32).")
    if args.validation_stride is not None and args.validation_stride <= 0:
        parser.error("--validation-stride must be positive.")
    for grid in [args.lstm_contexts, args.lstm_layers, args.lstm_batch_sizes]:
        if grid and any(v <= 0 for v in grid):
            parser.error("Context, layer and batch-size grids must be positive.")
    if args.lstm_contexts and any(c < args.season or c > args.context for c in args.lstm_contexts):
        parser.error("LSTM contexts must lie between --season and --context.")
    if any(not 0 <= d < 1 for d in args.lstm_dropouts):
        parser.error("Dropout must be in [0, 1).")
    if "lstm" in args.models:
        if args.cv_folds > 1 and args.valid_end // (args.cv_folds + 1) < max(args.horizons):
            parser.error("Each CV scoring block must fit a complete forecast horizon.")
        if args.cv_stop_hours < max(args.horizons):
            parser.error("--cv-stop-hours must fit a complete forecast horizon.")
        optimization_end = args.train_end if args.cv_folds == 1 else args.train_end - args.cv_stop_hours
        if optimization_end < args.context + max(args.horizons):
            parser.error("The initial CV block must fit context, horizon and --cv-stop-hours; "
                         "use fewer folds, a shorter stopping tail or more development data.")
    if len(set(args.seeds)) != 1:
        parser.error("Use one --seeds value; --repeats repeats fresh fitting with that fixed seed.")
    if "lstm" in args.models and args.cv_folds == 1 and len(lstm_grid(args)) != 1:
        parser.error("Single-holdout training requires one fixed LSTM configuration; "
                     "use --cv-folds >=2 to opt into a parameter search.")
    args.models = list(dict.fromkeys(args.models))
    args.horizons = list(dict.fromkeys(args.horizons))
    args.seeds = list(dict.fromkeys(args.seeds))
    return args


def load_data(path, download=False, input_mode="multivariate", *, features=None, test_end=TEST_END, allow_missing=False):
    import numpy as np
    import pandas as pd

    if not path.exists():
        if not download:
            raise FileNotFoundError(f"Missing {path}. Supply --download or --data PATH.")
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            content = response.read()
        path.write_bytes(content)
    frame = pd.read_csv(path)
    features = features or (FEATURES if input_mode == "multivariate" else ["OT"])
    if not {"date", *features}.issubset(frame.columns):
        raise ValueError(f"ETTh1 CSV must contain date and {features} columns.")
    dates = pd.to_datetime(frame["date"], errors="raise")
    # Explicit units avoid pandas/NumPy comparisons involving generic timedeltas.
    timestamps = dates.to_numpy(dtype="datetime64[ns]")
    if dates.isna().any() or not np.all(np.diff(timestamps) == np.timedelta64(1, "h")):
        raise ValueError("Timestamps must be unique, ordered and exactly hourly; no automatic imputation.")
    values = frame[features].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if len(values) < test_end or np.isinf(values).any() or (not allow_missing and np.isnan(values).any()):
        raise ValueError(f"Need {test_end} hourly rows; NaNs require --allow-missing and infinities are forbidden.")
    return dates.iloc[:test_end].reset_index(drop=True), values[:test_end]


def complete_origins(values, origins, context, horizon):
    """Filter using full shared input context and future target only, without imputing."""
    import numpy as np

    origins = np.asarray(origins, dtype=int)
    history_bad = np.r_[0, np.cumsum(~np.isfinite(values).all(axis=1))]
    target_bad = np.r_[0, np.cumsum(~np.isfinite(values[:, 0]))]
    valid = ((history_bad[origins] - history_bad[origins - context] == 0) &
             (target_bad[origins + horizon] - target_bad[origins] == 0))
    return origins[valid]


def make_windows(values, origins, context, horizon):
    import numpy as np

    origins = np.asarray(origins, dtype=int)
    if not len(origins) or origins.min() < context or origins.max() + horizon > len(values):
        raise ValueError("Forecast origins do not have sufficient context/targets.")
    x = np.stack([values[t - context:t] for t in origins])
    target = values[:, 0] if values.ndim == 2 else values
    y = np.stack([target[t:t + horizon] for t in origins])
    return x, y


def seasonal_naive(batch, horizon):
    """Copy the last horizon target values in chronological order."""
    if horizon <= 0 or batch.shape[1] < horizon:
        raise ValueError("Seasonal-naive needs at least horizon observations in its context.")
    return batch[:, -horizon:, 0]


def window_statistics(measures):
    """Mean and sample standard deviation of one metric value per origin."""
    import numpy as np

    result = {}
    for name, values in measures.items():
        values = np.asarray(values, dtype=np.float64)
        valid = len(values) > 0 and np.isfinite(values).all()
        result[f"{name}_window_mean"] = float(values.mean()) if valid else None
        result[f"{name}_window_std"] = float(values.std(ddof=1)) if valid and len(values) > 1 else None
    return result


def accuracy(actual, predicted, training, season):
    import numpy as np

    if predicted.shape != actual.shape or not np.isfinite(predicted).all():
        raise ValueError(f"Invalid predictions: expected finite {actual.shape}, got {predicted.shape}.")
    error = predicted.astype(np.float64) - actual.astype(np.float64)
    differences = np.abs(training[season:].astype(np.float64) - training[:-season])
    differences = differences[np.isfinite(differences)]
    scale = float(differences.mean()) if len(differences) else 0.0
    mae = float(np.abs(error).mean())
    window_mae = np.abs(error).mean(axis=1)
    stats = window_statistics({
        "mae": window_mae,
        "rmse": np.sqrt(np.mean(error**2, axis=1)),
        "mase": window_mae / scale if scale > 0 else np.full(len(error), np.nan),
    })
    return {"mae": mae, "rmse": float(np.sqrt(np.mean(error**2))),
            "mase": mae / scale if scale > 0 else None, **stats}


def synchronize(device):
    if device == "cuda":
        import torch
        torch.cuda.synchronize()


def benchmark(predict, inputs, horizon, device, batch_size, warmup):
    import numpy as np

    first = inputs[:batch_size]
    for _ in range(warmup):
        predict(first, horizon)
    synchronize(device)
    predictions, durations, sizes = [], [], []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start:start + batch_size]
        synchronize(device)
        begin = time.perf_counter()
        output = predict(batch, horizon)
        synchronize(device)
        elapsed = time.perf_counter() - begin
        output = np.asarray(output)
        if output.shape != (len(batch), horizon) or not np.isfinite(output).all():
            raise ValueError(f"Adapter returned invalid shape/values: {output.shape}.")
        predictions.append(output)
        durations.append(elapsed)
        sizes.append(len(batch))
    full = [d for d, size in zip(durations, sizes) if size == batch_size]
    timing = {"median_batch_latency_ms": float(np.median(full) * 1000) if full else None,
              "mean_batch_latency_ms": float(np.mean(full) * 1000) if full else None,
              "std_batch_latency_ms": float(np.std(full, ddof=1) * 1000) if len(full) > 1 else None,
              "p95_batch_latency_ms": float(np.percentile(full, 95) * 1000) if full else None,
              "full_batches_timed": len(full), "inference_seconds": sum(durations),
              "forecasts_per_second": len(inputs) / sum(durations)}
    return np.concatenate(predictions), timing


def validation_origins(horizon, args):
    """Disjoint target blocks; historical inputs may cross earlier boundaries."""
    import numpy as np

    train_end = getattr(args, "fit_train_end", args.train_end)
    valid_end = getattr(args, "fit_valid_end", args.valid_end)
    score_start = getattr(args, "fit_score_start", None)
    edges = ([train_end, score_start, valid_end] if score_start is not None else
             np.linspace(train_end, valid_end, args.validation_blocks + 1, dtype=int))
    stride = args.validation_stride or max(args.stride, horizon)
    return [np.arange(a, b - horizon + 1, stride) for a, b in zip(edges[:-1], edges[1:])]


def lstm_grid(args):
    dimensions = {
        "hidden_size": args.lstm_hidden_sizes or [args.hidden_size],
        "learning_rate": args.lstm_learning_rates or [args.learning_rate],
        "weight_decay": args.lstm_weight_decays,
        "context": args.lstm_contexts or [args.context],
        "layers": args.lstm_layers or [args.layers],
        "dropout": args.lstm_dropouts,
        "output_mode": args.lstm_output_modes,
        "train_batch_size": args.lstm_batch_sizes or [args.train_batch_size],
    }
    if args.no_tune_lstm:
        return [dict(hidden_size=args.hidden_size, learning_rate=args.learning_rate, weight_decay=0.0,
                     context=args.context, layers=args.layers, dropout=args.lstm_dropouts[0],
                     output_mode=args.lstm_output_modes[0], train_batch_size=args.train_batch_size)]
    return [dict(zip(dimensions, values)) for values in
            product(*(dict.fromkeys(v) for v in dimensions.values()))]


def fit_lstm_candidate(values, horizon, args, device, seed, output):
    import numpy as np
    import pandas as pd
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if values.ndim != 2:
        raise ValueError("LSTM inputs must have shape (time, features), with the target first.")
    train_end = getattr(args, "fit_train_end", args.train_end)
    valid_end = getattr(args, "fit_valid_end", args.valid_end)
    # Each fold learns its own scaling using only its optimization prefix.
    mean = np.nanmean(values[:train_end], axis=0)
    std = np.maximum(np.nanstd(values[:train_end], axis=0), 1e-8)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("Training prefix has a feature with no observed values.")
    target_mean, target_std = float(mean[0]), float(std[0])
    scaled = (values[:valid_end] - mean) / std
    shared_context = getattr(args, "shared_context", args.context)
    train_origins = complete_origins(scaled, np.arange(shared_context, train_end - horizon + 1, args.train_stride),
                                     shared_context, horizon)
    refit_full = getattr(args, "refit_full", False)
    scheduled_blocks = [] if refit_full else validation_origins(horizon, args)
    blocks = [complete_origins(scaled, origins, shared_context, horizon) for origins in scheduled_blocks]
    if not len(train_origins) or any(not len(block) for block in blocks):
        raise ValueError(f"Insufficient complete windows in fold {getattr(args, 'fold', 'final')}: "
                         f"training={len(train_origins)}, stopping/scoring={[len(block) for block in blocks]}. "
                         "Inspect missing-data coverage or adjust context/stopping-tail lengths.")
    tx, ty = make_windows(scaled, train_origins, args.context, horizon)
    train = DataLoader(TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty)),
                       batch_size=args.train_batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(seed))
    valid = None
    if not refit_full:
        vx, vy = make_windows(scaled, blocks[0], args.context, horizon)
        valid = DataLoader(TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy)),
                           batch_size=args.train_batch_size)

    class ForecastLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(values.shape[1], args.hidden_size, args.layers, batch_first=True,
                                   dropout=args.dropout if args.layers > 1 else 0.0)
            self.head = nn.Linear(args.hidden_size, horizon)

        def forward(self, x):
            offset = x[:, -1:, 0].clone() if args.output_mode == "last" else 0.0
            if args.output_mode == "last":
                x = x.clone()
                x[:, :, 0] = x[:, :, 0] - offset
            _, (hidden, _) = self.encoder(x)
            return self.head(hidden[-1]) + offset

    model = ForecastLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss()
    best_loss, best_state, stale, best_epoch = float("inf"), None, 0, 0
    training_batches = early_stopping_batches = training_window_presentations = 0
    history = []
    synchronize(device)
    begin = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = 0.0
        for x, y in train:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            training_batches += 1
            training_window_presentations += len(x)
            train_total += loss.item() * len(x)
        if not np.isfinite(train_total):
            raise ValueError("LSTM training loss became non-finite.")
        if refit_full:
            history.append({"epoch": epoch, "train_mae_scaled": train_total / len(train.dataset),
                            "train_mae": train_total / len(train.dataset) * target_std})
            best_epoch = epoch
            print(f"  seed={seed} refit epoch={epoch}/{args.epochs}", flush=True)
            continue
        model.eval()
        valid_total = 0.0
        with torch.inference_mode():
            for x, y in valid:
                early_stopping_batches += 1
                valid_total += loss_fn(model(x.to(device)), y.to(device)).item() * len(x)
        valid_loss = valid_total / len(valid.dataset)
        if not np.isfinite(valid_loss):
            raise ValueError("LSTM validation loss became non-finite.")
        history.append({"epoch": epoch, "train_mae_scaled": train_total / len(train.dataset),
                        "validation_mae_scaled": valid_loss,
                        "train_mae": train_total / len(train.dataset) * target_std,
                        "early_stopping_mae": valid_loss * target_std})
        print(f"  seed={seed} epoch={epoch} early_stopping_MAE={valid_loss * target_std:.5f}", flush=True)
        if valid_loss < best_loss:
            best_loss, stale, best_epoch = valid_loss, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    synchronize(device)
    train_seconds = time.perf_counter() - begin
    if refit_full:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    # Evaluate the frozen checkpoint on separate selection blocks, once per trial.
    def score(origins):
        sx, sy = make_windows(scaled, origins, args.context, horizon)
        loader = DataLoader(TensorDataset(torch.from_numpy(sx), torch.from_numpy(sy)),
                            batch_size=args.train_batch_size)
        baseline_x, _ = make_windows(scaled, origins, horizon, horizon)
        naive_errors = np.abs(seasonal_naive(baseline_x, horizon) - sy).mean(axis=1) * target_std
        errors, persistence_errors = [], []
        with torch.inference_mode():
            for x, y in loader:
                pred = model(x.to(device)).cpu()
                errors.extend((pred - y).abs().mean(dim=1).numpy() * target_std)
                persistence_errors.extend((x[:, -1:, 0] - y).abs().mean(dim=1).numpy() * target_std)
        return {"mae": float(np.mean(errors)), "seasonal_naive_mae": float(np.mean(naive_errors)),
                "persistence_mae": float(np.mean(persistence_errors)), "origins": len(origins),
                "first_origin": int(origins[0]), "last_target_exclusive": int(origins[-1] + horizon)}

    block_scores = [dict(block=i, role="early_stopping" if i == 0 else "selection",
                         scheduled_origins=len(scheduled_blocks[i]),
                         excluded_origins=len(scheduled_blocks[i]) - len(origins), **score(origins))
                    for i, origins in enumerate(blocks)]
    selection_scores = block_scores[1:] or block_scores
    selection_mae = float(np.mean([b["mae"] for b in selection_scores])) if selection_scores else None
    stopping_mae = None if refit_full else best_loss * target_std
    selected_train_mae = score(train_origins)["mae"]
    workload = {"train_windows": len(train_origins), "train_batches_per_epoch": len(train),
                "training_batches": training_batches, "early_stopping_batches": early_stopping_batches,
                "fit_batches": training_batches + early_stopping_batches,
                "training_window_presentations": training_window_presentations,
                "early_stopping_windows": len(blocks[0]) if blocks else 0,
                "epochs_run": len(history)}
    pd.DataFrame(block_scores, columns=["block", "role", "scheduled_origins", "excluded_origins",
                                       "mae", "seasonal_naive_mae", "persistence_mae", "origins",
                                       "first_origin", "last_target_exclusive"]).to_csv(output / f"lstm_h{horizon}_seed{seed}_validation.csv", index=False)
    torch.save({**workload, "state_dict": best_state, "mean": mean.tolist(), "std": std.tolist(), "seed": seed,
                "input_size": values.shape[1], "features": args.feature_columns[:values.shape[1]], "target": args.target,
                "context": args.context, "horizon": horizon, "hidden_size": args.hidden_size,
                "layers": args.layers, "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay, "best_epoch": best_epoch,
                "dropout": args.dropout, "dropout_placement": "between_lstm_layers",
                "effective_dropout": model.encoder.dropout, "output_mode": args.output_mode,
                "validation_blocks": len(blocks), "refit_full": refit_full,
                "train_end_exclusive": train_end, "validation_end_exclusive": valid_end,
                "early_stopping_mae": stopping_mae,
                "validation_mae": selection_mae}, output / f"lstm_h{horizon}_seed{seed}.pt")
    pd.DataFrame(history).to_csv(output / f"lstm_h{horizon}_seed{seed}_training.csv", index=False)

    def predict(x, h):
        with torch.inference_mode():
            tensor = torch.from_numpy((x[:, -args.context:] - mean) / std).to(device)
            return model(tensor).cpu().numpy() * target_std + target_mean

    return predict, {**workload, "training_seconds": train_seconds, "best_epoch": best_epoch,
                     "dropout_placement": "between_lstm_layers", "effective_dropout": model.encoder.dropout,
                     "validation_mae": selection_mae,
                     "early_stopping_mae": stopping_mae,
                     "selected_train_mae": selected_train_mae,
                     "validation_seasonal_naive_mae": float(np.mean([b["seasonal_naive_mae"] for b in selection_scores])) if selection_scores else None,
                     "validation_persistence_mae": float(np.mean([b["persistence_mae"] for b in selection_scores])) if selection_scores else None,
                     "validation_block_mae_std": float(np.std([b["mae"] for b in selection_scores], ddof=1))
                         if len(selection_scores) > 1 else None,
                     "train_end_exclusive": train_end, "validation_end_exclusive": valid_end,
                     "train_windows": len(train_origins), "epochs_run": len(history),
                     "refit_full": refit_full,
                     "hit_epoch_limit": not refit_full and len(history) == args.epochs,
                     "parameters": sum(p.numel() for p in model.parameters())}


def cross_validation_folds(args):
    """Outer scoring follows a disjoint inner stopping tail and growing training prefix."""
    import numpy as np

    if args.cv_folds == 1:
        return [{"fold": 1, "fit_train_end": args.valid_end - args.cv_stop_hours,
                 "fit_valid_end": args.valid_end, "validation_blocks": 1}]
    edges = np.linspace(0, args.valid_end, args.cv_folds + 2, dtype=int)[1:]
    return [{"fold": i + 1, "fit_train_end": int(start - args.cv_stop_hours),
             "fit_score_start": int(start), "fit_valid_end": int(end)}
            for i, (start, end) in enumerate(zip(edges[:-1], edges[1:]))]


def training_workload(selection_fits, final_fit):
    """Aggregate optimization/early-stopping work, without calling reused windows unique."""
    records = [*selection_fits, final_fit]
    counts = {}
    for key in ("training_batches", "early_stopping_batches", "fit_batches", "training_window_presentations"):
        counts[f"total_{key}"] = sum(fit[key] for fit in records)
    counts.update(selection_training_batches=sum(fit["training_batches"] for fit in selection_fits),
                  refit_training_batches=final_fit["training_batches"],
                  refit_train_windows=final_fit["train_windows"])
    return counts


def train_lstm_holdout(values, horizon, args, device, seed, output):
    """Select an epoch on one trailing holdout, then refit all development data."""
    configs = lstm_grid(args)
    if len(configs) != 1:
        raise ValueError("Single-holdout training requires one fixed configuration.")
    config = configs[0]
    development = values[:args.valid_end].copy()
    label = f"lstm_h{horizon}_seed{seed}"
    split = cross_validation_folds(args)[0]
    selection_args = argparse.Namespace(**{**vars(args), "shared_context": args.context, **config, **split})
    selection_dir = output / f"{label}_epoch_selection"
    selection_dir.mkdir(parents=True, exist_ok=False)
    synchronize(device)
    begin = time.perf_counter()
    print("  LSTM: selecting epochs on one chronological holdout", flush=True)
    selection_predict, epoch_info = fit_lstm_candidate(development, horizon, selection_args, device, seed, selection_dir)
    refit_epochs = int(epoch_info["best_epoch"])
    del selection_predict
    gc.collect()
    final_args = argparse.Namespace(**{**vars(args), "shared_context": args.context, **config,
                                      "fit_train_end": args.valid_end, "fit_valid_end": args.valid_end,
                                      "refit_full": True, "epochs": refit_epochs})
    print(f"  LSTM: refitting all development data for {refit_epochs} epochs", flush=True)
    predict, final_info = fit_lstm_candidate(development, horizon, final_args, device, seed, output)
    synchronize(device)
    elapsed = time.perf_counter() - begin
    workload = training_workload([epoch_info], final_info)
    workload.update(initial_train_windows=epoch_info["train_windows"],
                    initial_training_batches=epoch_info["training_batches"],
                    initial_epochs_run=epoch_info["epochs_run"])
    write_json(output / f"{label}_selection.json", {
        "workload": workload,
        "validation_protocol": "single chronological early-stopping holdout; full-development refit",
        "selection_metric": "early_stopping_mae", "seed": seed, "horizon": horizon,
        "configuration": config, "epoch_selection_split": split, "epoch_selection": epoch_info,
        "refit_epochs": refit_epochs, "refit_epoch_rule": "best epoch on the single early-stopping holdout",
        "final_fit": final_info, "test_used_for_selection": False, "test_start": args.valid_end,
        "checkpoint": f"{label}.pt"})
    return predict, {**config, **final_info, **workload, "refit_epochs": refit_epochs,
                     "validation_protocol": "single_holdout", "cv_folds": 0,
                     "epoch_selection_mae": epoch_info["early_stopping_mae"],
                     "epoch_selection_train_end": split["fit_train_end"],
                     "epoch_selection_training_seconds": epoch_info["training_seconds"],
                     "selected_training_seconds": final_info["training_seconds"],
                     "training_seconds": epoch_info["training_seconds"] + final_info["training_seconds"],
                     "tuning_seconds": elapsed, "tuning_trials": 0,
                     "selection_metric": "early_stopping_mae"}


def train_lstm(values, horizon, args, device, seed, output):
    """Rank configurations by mean outer-fold MAE, then fit one final model."""
    import numpy as np
    import pandas as pd

    if args.cv_folds == 1:
        return train_lstm_holdout(values, horizon, args, device, seed, output)
    development_values = values[:args.valid_end].copy()
    candidates = lstm_grid(args)
    folds = cross_validation_folds(args)
    label = f"lstm_h{horizon}_seed{seed}"
    trials, fold_results = [], []
    best_info, best_config = None, None
    synchronize(device)
    begin = time.perf_counter()
    for trial, config in enumerate(candidates, start=1):
        scores = []
        print(f"  LSTM CV trial {trial}/{len(candidates)}: {config}", flush=True)
        for fold in folds:
            fold_args = argparse.Namespace(**{**vars(args), "shared_context": args.context, **config, **fold})
            fold_dir = output / f"{label}_tuning" / f"trial_{trial:02d}" / f"fold_{fold['fold']:02d}"
            fold_dir.mkdir(parents=True, exist_ok=False)
            print(f"    Fold {fold['fold']}/{len(folds)}", flush=True)
            predict, info = fit_lstm_candidate(development_values[:fold["fit_valid_end"]],
                                               horizon, fold_args, device, seed, fold_dir)
            scores.append(info)
            fold_results.append({"trial": trial, **config, **fold, **info})
            pd.DataFrame(fold_results).to_csv(output / f"{label}_cv_folds.csv", index=False)
            predict = None
            gc.collect()
        trial_info = {"trial": trial, **config, "cv_folds": len(folds),
                      "cv_mae": float(np.mean([s["validation_mae"] for s in scores])),
                      "cv_mae_std": float(np.std([s["validation_mae"] for s in scores], ddof=1)),
                      "cv_seasonal_naive_mae": float(np.mean([s["validation_seasonal_naive_mae"] for s in scores])),
                      "cv_persistence_mae": float(np.mean([s["validation_persistence_mae"] for s in scores])),
                      "refit_epochs": int(np.ceil(np.median([s["best_epoch"] for s in scores]))),
                      "training_seconds": sum(s["training_seconds"] for s in scores)}
        trials.append(trial_info)
        pd.DataFrame(trials).to_csv(output / f"{label}_tuning.csv", index=False)
        # Strict comparison resolves ties by the predefined grid order.
        if best_info is None or trial_info["cv_mae"] < best_info["cv_mae"]:
            best_info, best_config = trial_info.copy(), config.copy()

    # Refit all development observations for a duration selected only from CV.
    # No validation tail is withheld and test observations never enter fitting.
    final_args = argparse.Namespace(**{**vars(args), "shared_context": args.context, **best_config,
                                      "fit_train_end": args.valid_end, "fit_valid_end": args.valid_end,
                                      "refit_full": True, "epochs": best_info["refit_epochs"]})
    print(f"  Selected trial {best_info['trial']}: CV_MAE={best_info['cv_mae']:.5f}; final fit", flush=True)
    predict, final_info = fit_lstm_candidate(development_values, horizon, final_args, device, seed, output)
    synchronize(device)
    tuning_seconds = time.perf_counter() - begin
    workload = training_workload(fold_results, final_info)
    selection = {"workload": workload, "selection_metric": "cv_mae", "seed": seed, "horizon": horizon,
                 "trial_count": len(trials), "selected": best_info, "folds": folds,
                 "validation_protocol": "optional expanding training over development; inner early stopping; mean outer-fold MAE",
                 "refit_epoch_rule": "ceiling of median best epoch across winning configuration CV folds",
                 "validation_stride": args.validation_stride or max(args.stride, horizon),
                 "final_fit": final_info, "tuning_seconds": tuning_seconds,
                 "test_used_for_selection": False, "test_start": args.valid_end,
                 "checkpoint": f"{label}.pt"}
    write_json(output / f"{label}_selection.json", selection)
    info = {**best_info, **final_info, **workload,
            "selected_training_seconds": final_info["training_seconds"],
            "training_seconds": sum(t["training_seconds"] for t in trials) + final_info["training_seconds"],
            "tuning_seconds": tuning_seconds, "tuning_trials": len(trials),
            "selection_metric": "cv_mae"}
    return predict, info


def foundation_adapter(name, device, batch_size):
    import numpy as np
    import torch

    checkpoint = CHECKPOINTS[name]
    if name == "chronos2":
        from chronos import Chronos2Pipeline
        model = Chronos2Pipeline.from_pretrained(checkpoint, device_map=device)

        def predict(x, h):
            with torch.inference_mode():
                quantiles, _ = model.predict_quantiles(
                    inputs=torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1))), prediction_length=h,
                    quantile_levels=[0.5], batch_size=batch_size * x.shape[2], cross_learning=False)
            return np.stack([q[0, :, 0].cpu().numpy() for q in quantiles])

    elif name == "timesfm3":
        from timesfm3 import ModelConfig, TimesFM3Evaluator
        model = TimesFM3Evaluator(ModelConfig(checkpoint_path=checkpoint,
                                             per_core_batch_size=batch_size, device=device))

        def predict(x, h):
            with torch.inference_mode():
                results = list(model.predict_batch(list(x.transpose(0, 2, 1)), horizon=h, return_quantiles=True,
                                                   use_symmetric_averaging=False))
            return np.stack([np.asarray(r.quantiles).reshape(x.shape[2], h, 9)[0, :, 4] for r in results])

    elif name == "toto2":
        from toto2 import Toto2Model
        model = Toto2Model.from_pretrained(checkpoint).to(device).eval()

        def predict(x, h):
            with torch.inference_mode():
                target = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 2, 1))).to(device)
                # Toto patches the context directly; its horizon rounding does not
                # pad history. Left padding preserves the forecast origin and all
                # observed history, while the mask excludes artificial values.
                padding = (-target.shape[-1]) % model.config.patch_size
                target_mask = torch.ones_like(target, dtype=torch.bool)
                if padding:
                    target = torch.nn.functional.pad(target, (padding, 0), value=0.0)
                    target_mask = torch.nn.functional.pad(target_mask, (padding, 0), value=False)
                quantiles = model.forecast(
                    {"target": target, "target_mask": target_mask,
                     "series_ids": torch.zeros(target.shape[:2], device=device, dtype=torch.long)},
                    horizon=h, decode_block_size=768, has_missing_values=bool(padding))
                return quantiles[4, :, 0, :].cpu().numpy()
    else:
        raise ValueError(f"Unknown foundation model: {name}")
    return predict


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + "\n")


def summarize_run(directory):
    """Add window statistics to metrics.csv and summarize saved forecasts."""
    import numpy as np
    import pandas as pd

    from repetition_reporting import REPEAT_METRICS, repetition_summary

    runs = pd.read_csv(directory / "metrics.csv")
    if "repeat" not in runs:
        runs["repeat"] = 1
    windows = []
    for row_index, run in runs.iterrows():
        horizon = int(run.horizon)
        label = f"{run.model}_h{horizon}"
        if pd.notna(run.seed):
            label += f"_seed{int(run.seed)}"
        artifact_directory = run.get("artifact_directory", ".")
        predictions = pd.read_csv(directory / artifact_directory / f"{label}_predictions.csv")
        if not np.isfinite(predictions[["actual", "prediction"]].to_numpy()).all():
            raise ValueError(f"Non-finite forecasts in {label}.")
        error = predictions.prediction.astype(float) - predictions.actual.astype(float)
        predictions = predictions.assign(abs_error=error.abs(), squared_error=error**2)
        grouped = predictions.groupby("origin", sort=True)
        if len(grouped) != int(run.origins) or not grouped.size().eq(horizon).all():
            raise ValueError(f"Incomplete forecast windows in {label}.")
        per_origin = grouped.agg(mae=("abs_error", "mean"), mse=("squared_error", "mean"))
        per_origin["rmse"] = np.sqrt(per_origin.pop("mse"))
        # Reuse the exact training-derived MASE scale from the original run.
        if pd.isna(run.mase):
            per_origin["mase"] = np.nan
        elif run.mae > 0:
            per_origin["mase"] = per_origin.mae * (run.mase / run.mae)
        elif (per_origin.mae == 0).all() and run.mase == 0:
            per_origin["mase"] = 0.0
        else:
            raise ValueError(f"Inconsistent saved MASE in {label}.")
        stats = window_statistics({key: per_origin[key].to_numpy() for key in ["mae", "rmse", "mase"]})
        for key, value in stats.items():
            runs.at[row_index, key] = value if value is not None else np.nan
        per_origin = per_origin.reset_index().assign(model=run.model, seed=run.seed, horizon=horizon, repeat=int(run["repeat"]))
        windows.append(per_origin)
    windows = pd.concat(windows, ignore_index=True)
    measures = ["mae", "rmse", "mase"]
    window_summary = windows.groupby(["model", "horizon", "seed", "repeat"], dropna=False)[measures].agg(
        ["count", "mean", "std"])
    window_summary.columns = [f"{metric}_window_{stat}" for metric, stat in window_summary.columns]
    output = directory / "summary"
    output.mkdir(exist_ok=True)
    windows.to_csv(output / "per_origin_metrics.csv", index=False)
    window_summary.to_csv(output / "window_summary.csv")
    runs.to_csv(directory / "metrics.csv", index=False)
    metadata_path = directory / "run.json"
    requested = json.loads(metadata_path.read_text())["arguments"].get("repeats", 1) if metadata_path.exists() else int(runs["repeat"].max())
    repetition_summary(runs, ["model", "horizon", "seed"], REPEAT_METRICS, requested).to_csv(
        output / "repeat_summary.csv", index=False)
    (output / "README.txt").write_text(
        "per_origin_metrics.csv: each metric is computed across the forecast horizon at one origin.\n"
        "window_summary.csv: mean and sample std across origins, separately for each model/horizon/seed/repetition.\n"
        "Mean window RMSE is not pooled RMSE; metrics.csv retains pooled RMSE.\n"
        "metrics.csv includes *_window_mean and *_window_std for MAE, RMSE and MASE.\n"
        "repeat_summary.csv: mean/sample std of run-level metrics across fresh setups/fits with a fixed seed.\n"
        "Workload counts distinguish optimization batches and early-stopping batches; post-fit diagnostics are excluded.\n"
        "Window std is available for every model with at least two forecast origins.\n"
        "New runs also record mean/std batch latency in metrics.csv. Old latency std cannot be reconstructed.\n"
        "All std use ddof=1. Windows may overlap and be serially correlated; std are descriptive, not confidence intervals.\n"
    )
    print(f"Summaries: {output.resolve()}")


def main(argv=None):
    args = parse_args(argv)
    if args.summarize is not None:
        summarize_run(args.summarize)
        return 0
    if args.lstm_plan:
        grid = lstm_grid(args)
        print(json.dumps({"configurations_per_horizon": len(grid),
                          "repeats": args.repeats,
                          "validation_fits": len(grid) * len(args.horizons) * args.cv_folds * args.repeats,
                          "final_fits": len(args.horizons) * args.repeats,
                          "total_fits": (len(grid) * args.cv_folds + 1) * len(args.horizons) * args.repeats,
                          "maximum_epochs_per_fit": args.epochs,
                          "folds": cross_validation_folds(args), "grid": grid}, indent=2))
        return 0
    if "toto2" in args.models and sys.version_info < (3, 12):
        raise SystemExit(
            "Toto 2 requires Python 3.12+. This interpreter is "
            f"{platform.python_version()}. See --help to create a Python 3.12 environment, "
            "or use --models lstm seasonal-naive on Python 3.10."
        )
    import numpy as np
    import pandas as pd

    needs_torch = any(name != "seasonal-naive" for name in args.models)
    hardware = {"platform": platform.platform(), "python": sys.version}
    device = "cpu"
    if needs_torch:
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("Install PyTorch for neural models; see installation instructions in --help.") from exc
        device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
        if device == "auto":
            device = "cpu"
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable.")
        torch.set_num_threads(args.threads)
        hardware.update(torch=torch.__version__, cuda=torch.version.cuda,
                        accelerator=torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
                        torch_threads=torch.get_num_threads())
    elif args.device == "cuda":
        raise ValueError("Seasonal-naive is NumPy/CPU only. Use --device cpu or auto.")
    dates, values = load_data(args.data, args.download, args.input_mode,
                              features=args.feature_columns, test_end=args.test_end, allow_missing=args.allow_missing)
    output = args.output or Path("results") / datetime.now(timezone.utc).strftime("etth1_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    metadata = {"arguments": vars(args), "device": device, "hardware": hardware,
                "input_mode": args.input_mode, "features": args.feature_columns[:values.shape[1]], "target": args.target,
                "foundation_output": "Joint channel forecasts; only first channel scored; no future covariates.",
                "packages": packages, "checkpoints": CHECKPOINTS, "data_url": DATA_URL if args.dataset_name == "etth1" else None,
                "dataset": args.dataset_name,
                "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "split_end_indices_exclusive": [args.valid_end, args.test_end],
                "mase_scale_end_exclusive": args.valid_end,
                "split_dates": {"development": [str(dates[0]), str(dates[args.valid_end - 1])],
                                "test": [str(dates[args.valid_end]), str(dates[args.test_end - 1])]},
                "pretraining_overlap": "Not audited; frozen inference does not establish unseen data.",
                "timing": "Warm adapter latency, CPU NumPy input/output, synchronized CUDA; excludes loading/training.",
                "repetition_policy": "fresh setup/fit, fixed seed and test windows; sequential in one process per series",
                "setup_timing": "includes model loading and LSTM fitting; caches/imports may be warm after first repetition; no cache flushing",
                "smoke_test": args.max_origins is not None, "status": "running"}
    write_json(output / "run.json", metadata)
    rows, errors = [], []
    for horizon in args.horizons:
        scheduled = np.arange(args.valid_end, args.test_end - horizon + 1, args.stride)
        origins = complete_origins(values, scheduled, args.context, horizon)
        write_json(output / f"coverage_h{horizon}.json", {
            "scheduled_origins": len(scheduled), "complete_origins": len(origins),
            "excluded_origins": len(scheduled) - len(origins), "context": args.context,
            "policy": "exclude any missing input or target; shared origins for every model"})
        if not len(origins):
            errors.append({"run": f"all_models_h{horizon}", "error": "No complete test windows; see coverage report."})
            write_json(output / "errors.json", errors)
            continue
        if args.max_origins is not None:
            origins = origins[:args.max_origins]
        x, y = make_windows(values, origins, args.context, horizon)
        for name in args.models:
            for repeat in range(1, args.repeats + 1):
                seed = args.seeds[0] if name == "lstm" else None
                artifact_directory = f"repeat_{repeat:02d}" if args.repeats > 1 else "."
                run_output = output / artifact_directory
                run_output.mkdir(exist_ok=True)
                label = f"{name}_h{horizon}" + (f"_seed{seed}" if seed is not None else "")
                print(f"Running {label} repetition {repeat}/{args.repeats}: {len(origins)} origins", flush=True)
                predict = None
                try:
                    model_device = "cpu" if name == "seasonal-naive" else device
                    synchronize(model_device)
                    begin = time.perf_counter()
                    info = {"training_seconds": 0.0}
                    if name == "seasonal-naive":
                        predict = seasonal_naive
                    elif name == "lstm":
                        predict, info = train_lstm(values, horizon, args, device, seed, run_output)
                    else:
                        predict = foundation_adapter(name, device, args.batch_size)
                    synchronize(model_device)
                    setup_seconds = time.perf_counter() - begin
                    pred, timing = benchmark(predict, x, horizon, model_device, args.batch_size, args.warmup)
                    metrics = accuracy(y, pred, values[:args.valid_end, 0], args.season)
                    row = {"repeat": repeat, "artifact_directory": artifact_directory, "model": name, "seed": seed, "horizon": horizon, "context": args.context,
                           "origins": len(origins), "batch_size": args.batch_size, "device": model_device,
                           "input_mode": "univariate" if name == "seasonal-naive" else args.input_mode,
                           "input_features": 1 if name == "seasonal-naive" else values.shape[1],
                           "setup_seconds_including_training": setup_seconds, **info, **metrics, **timing}
                    indices = origins[:, None] + np.arange(horizon)
                    pd.DataFrame({"origin": np.repeat(dates.iloc[origins].to_numpy(), horizon),
                                  "timestamp": dates.to_numpy()[indices].ravel(),
                                  "lead_hour": np.tile(np.arange(1, horizon + 1), len(origins)),
                                  "actual": y.ravel(), "prediction": pred.ravel()}).to_csv(
                                      run_output / f"{label}_predictions.csv", index=False)
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
                    print(f"  MAE={metrics['mae']:.5f} RMSE={metrics['rmse']:.5f}", flush=True)
                except Exception as exc:
                    error = {"run": label, "repeat": repeat, "error": f"{type(exc).__name__}: {exc}"}
                    errors.append(error)
                    write_json(output / "errors.json", errors)
                    print(f"  FAILED: {error['error']}", file=sys.stderr, flush=True)
                finally:
                    predict = None
                    gc.collect()
                    if needs_torch and device == "cuda":
                        torch.cuda.empty_cache()
    metadata.update(status="partial_failure" if errors else "complete", failed_runs=len(errors),
                    completed_runs=len(rows))
    write_json(output / "run.json", metadata)
    if rows:
        summarize_run(output)
    print(f"Results: {output.resolve()}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
