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
Each trained LSTM has its own row; seeds are never pooled. Standard deviations use
ddof=1 and are undefined for a single observation. Overlapping forecast windows
are correlated: these descriptive standard deviations are not confidence intervals.
Mean window RMSE differs from the pooled RMSE retained in metrics.csv.
metrics.csv also contains *_window_mean and *_window_std for MAE, RMSE and MASE.

Defaults: seven historical input variables, oil-temperature (OT) evaluation,
512 hours of context, horizon 24, daily
rolling origins, one LSTM seed (42). For each horizon, eight LSTM configurations
are trained: hidden size 64/128, learning rate 0.001/(0.001/3), and L2 weight
decay 0/0.0001. Each uses MAE loss and validation early stopping. The checkpoint
with lowest validation MAE is evaluated on test data, without retraining.
Override the grid with --lstm-hidden-sizes, --lstm-learning-rates, and
--lstm-weight-decays, or use --no-tune-lstm to run a single configuration.
*_tuning.csv and *_selection.json record the search and selected configuration.
training_seconds includes all candidate training; selected_training_seconds
is for the winning trial only. Inference timing excludes the entire search.
Foundation models use their median (0.5 quantile) forecasts. Both MAE
and RMSE score the SAME predictions, in original target units.
Foundation models jointly forecast the seven channels; only OT is scored.
The LSTM consumes seven channels and predicts OT only. No future load values
are supplied. Foundation-model latency includes forecasting all channels.
Seasonal-naive remains an OT-only baseline. Use --input-mode univariate to
reproduce the previous target-only experiment. Results record the input mode.

The standard Informer split uses 30-day months: 12 train / 4 validation / 4 test.
Remaining rows are unused. This is a conventional benchmark split, NOT proof
that foundation models did not encounter these data during pretraining.

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
import shutil
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
TRAIN_END, VALID_END, TEST_END = 12 * 30 * 24, 16 * 30 * 24, 20 * 30 * 24
FEATURES = ["OT", "HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, default=Path("data/ETTh1.csv"))
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
    parser.add_argument("--season", type=int, default=24, help="Seasonal baseline and MASE period.")
    parser.add_argument("--batch-size", type=int, default=1, help="Inference batch size for every model.")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-origins", type=int, help="Smoke-test limit: first N test origins only.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU thread count.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Fixed LSTM training seed; default gives one selected model per horizon.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--no-tune-lstm", action="store_true",
                        help="Use --hidden-size and --learning-rate with no weight decay; skip the search.")
    parser.add_argument("--lstm-hidden-sizes", nargs="+", type=int,
                        help="Search sizes; default: --hidden-size and twice that size.")
    parser.add_argument("--lstm-learning-rates", nargs="+", type=float,
                        help="Search rates; default: --learning-rate and one third of that rate.")
    parser.add_argument("--lstm-weight-decays", nargs="+", type=float, default=[0.0, 1e-4],
                        help="L2 regularization strengths searched; default: 0 and 0.0001.")
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--train-stride", type=int, default=1)
    args = parser.parse_args(argv)
    positive = ["context", "stride", "season", "batch_size", "warmup", "threads", "epochs",
                "patience", "hidden_size", "layers", "train_batch_size", "train_stride"]
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
    if args.context < args.season or args.context + max(args.horizons) > TRAIN_END:
        parser.error("Context must cover the season and leave room for training targets.")
    if max(args.horizons) > TEST_END - VALID_END:
        parser.error("Horizon exceeds the validation/test period.")
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error("Seeds must be in [0, 2**32).")
    args.models = list(dict.fromkeys(args.models))
    args.horizons = list(dict.fromkeys(args.horizons))
    args.seeds = list(dict.fromkeys(args.seeds))
    return args


def load_data(path, download=False, input_mode="multivariate"):
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
    features = FEATURES if input_mode == "multivariate" else ["OT"]
    if not {"date", *features}.issubset(frame.columns):
        raise ValueError(f"ETTh1 CSV must contain date and {features} columns.")
    dates = pd.to_datetime(frame["date"], errors="raise")
    # Explicit units avoid pandas/NumPy comparisons involving generic timedeltas.
    timestamps = dates.to_numpy(dtype="datetime64[ns]")
    if dates.isna().any() or not np.all(np.diff(timestamps) == np.timedelta64(1, "h")):
        raise ValueError("Timestamps must be unique, ordered and exactly hourly; no automatic imputation.")
    values = frame[features].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    if len(values) < TEST_END or not np.isfinite(values).all():
        raise ValueError(f"Need at least {TEST_END} finite hourly observations for all input variables.")
    return dates.iloc[:TEST_END].reset_index(drop=True), values[:TEST_END]


def make_windows(values, origins, context, horizon):
    import numpy as np

    origins = np.asarray(origins, dtype=int)
    if not len(origins) or origins.min() < context or origins.max() + horizon > len(values):
        raise ValueError("Forecast origins do not have sufficient context/targets.")
    x = np.stack([values[t - context:t] for t in origins])
    target = values[:, 0] if values.ndim == 2 else values
    y = np.stack([target[t:t + horizon] for t in origins])
    return x, y


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
    scale = float(np.abs(np.diff(training.astype(np.float64), n=1)).mean()) if season == 1 else float(
        np.abs(training[season:].astype(np.float64) - training[:-season]).mean())
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
        raise ValueError("LSTM inputs must have shape (time, features), with OT first.")
    mean = values[:TRAIN_END].mean(axis=0)
    std = np.maximum(values[:TRAIN_END].std(axis=0), 1e-8)
    target_mean, target_std = float(mean[0]), float(std[0])
    scaled = (values[:VALID_END] - mean) / std
    train_origins = np.arange(args.context, TRAIN_END - horizon + 1, args.train_stride)
    valid_origins = np.arange(TRAIN_END, VALID_END - horizon + 1, args.stride)
    tx, ty = make_windows(scaled, train_origins, args.context, horizon)
    vx, vy = make_windows(scaled, valid_origins, args.context, horizon)
    train = DataLoader(TensorDataset(torch.from_numpy(tx), torch.from_numpy(ty)),
                       batch_size=args.train_batch_size, shuffle=True,
                       generator=torch.Generator().manual_seed(seed))
    valid = DataLoader(TensorDataset(torch.from_numpy(vx), torch.from_numpy(vy)),
                       batch_size=args.train_batch_size)

    class ForecastLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.LSTM(values.shape[1], args.hidden_size, args.layers, batch_first=True)
            self.head = nn.Linear(args.hidden_size, horizon)

        def forward(self, x):
            _, (hidden, _) = self.encoder(x)
            return self.head(hidden[-1])

    model = ForecastLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss()
    best_loss, best_state, stale, best_epoch = float("inf"), None, 0, 0
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
            train_total += loss.item() * len(x)
        model.eval()
        valid_total = 0.0
        with torch.inference_mode():
            for x, y in valid:
                valid_total += loss_fn(model(x.to(device)), y.to(device)).item() * len(x)
        valid_loss = valid_total / len(valid.dataset)
        if not np.isfinite(valid_loss):
            raise ValueError("LSTM validation loss became non-finite.")
        history.append({"epoch": epoch, "train_mae_scaled": train_total / len(train.dataset),
                        "validation_mae_scaled": valid_loss})
        print(f"  seed={seed} epoch={epoch} validation_MAE={valid_loss * target_std:.5f}", flush=True)
        if valid_loss < best_loss:
            best_loss, stale, best_epoch = valid_loss, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    synchronize(device)
    train_seconds = time.perf_counter() - begin
    model.load_state_dict(best_state)
    model.eval()
    torch.save({"state_dict": best_state, "mean": mean.tolist(), "std": std.tolist(), "seed": seed,
                "input_size": values.shape[1], "features": FEATURES[:values.shape[1]], "target": "OT",
                "context": args.context, "horizon": horizon, "hidden_size": args.hidden_size,
                "layers": args.layers, "learning_rate": args.learning_rate,
                "weight_decay": args.weight_decay, "best_epoch": best_epoch,
                "validation_mae": best_loss * target_std}, output / f"lstm_h{horizon}_seed{seed}.pt")
    pd.DataFrame(history).to_csv(output / f"lstm_h{horizon}_seed{seed}_training.csv", index=False)

    def predict(x, h):
        with torch.inference_mode():
            tensor = torch.from_numpy((x - mean) / std).to(device)
            return model(tensor).cpu().numpy() * target_std + target_mean

    return predict, {"training_seconds": train_seconds, "best_epoch": best_epoch,
                     "validation_mae": best_loss * target_std,
                     "parameters": sum(p.numel() for p in model.parameters())}


def train_lstm(values, horizon, args, device, seed, output):
    """Select a checkpoint by validation MAE; test values never enter the search."""
    import pandas as pd

    # One seed for all candidates. The selected checkpoint is used as-is, without
    # retraining on validation observations or selecting using test-set results.
    development_values = values[:VALID_END].copy()
    sizes = args.lstm_hidden_sizes or [args.hidden_size, 2 * args.hidden_size]
    rates = args.lstm_learning_rates or [args.learning_rate, args.learning_rate / 3]
    decays = args.lstm_weight_decays
    if args.no_tune_lstm:
        sizes, rates, decays = [args.hidden_size], [args.learning_rate], [0.0]
    candidates = list(product(dict.fromkeys(sizes), dict.fromkeys(rates), dict.fromkeys(decays)))
    label = f"lstm_h{horizon}_seed{seed}"
    trials = []
    best_predict, best_info, best_dir = None, None, None
    synchronize(device)
    begin = time.perf_counter()
    for trial, (hidden_size, learning_rate, weight_decay) in enumerate(candidates, start=1):
        trial_args = argparse.Namespace(**vars(args))
        trial_args.hidden_size = hidden_size
        trial_args.learning_rate = learning_rate
        trial_args.weight_decay = weight_decay
        trial_dir = output / f"{label}_tuning" / f"trial_{trial:02d}"
        trial_dir.mkdir(parents=True, exist_ok=False)
        print(f"  LSTM trial {trial}/{len(candidates)}: hidden={hidden_size}, "
              f"lr={learning_rate:g}, weight_decay={weight_decay:g}", flush=True)
        predict, info = fit_lstm_candidate(development_values, horizon, trial_args, device, seed, trial_dir)
        trial_info = {"trial": trial, "hidden_size": hidden_size, "learning_rate": learning_rate,
                      "weight_decay": weight_decay, **info}
        trials.append(trial_info)
        pd.DataFrame(trials).to_csv(output / f"{label}_tuning.csv", index=False)
        # Strict comparison resolves ties by the predefined grid order.
        if best_info is None or info["validation_mae"] < best_info["validation_mae"]:
            best_predict, best_info, best_dir = predict, trial_info.copy(), trial_dir
        predict = None
        gc.collect()
    synchronize(device)
    tuning_seconds = time.perf_counter() - begin
    for suffix in [".pt", "_training.csv"]:
        shutil.copyfile(best_dir / f"{label}{suffix}", output / f"{label}{suffix}")
    selection = {"selection_metric": "validation_mae", "seed": seed, "horizon": horizon,
                 "trial_count": len(trials), "selected": best_info,
                 "tuning_seconds": tuning_seconds, "test_used_for_selection": False,
                 "train_end_exclusive": TRAIN_END, "validation_end_exclusive": VALID_END,
                 "checkpoint": f"{label}.pt"}
    write_json(output / f"{label}_selection.json", selection)
    print(f"  Selected trial {best_info['trial']}: validation_MAE={best_info['validation_mae']:.5f}", flush=True)
    info = {**best_info, "selected_training_seconds": best_info["training_seconds"],
            "training_seconds": sum(trial["training_seconds"] for trial in trials),
            "tuning_seconds": tuning_seconds, "tuning_trials": len(trials),
            "selection_metric": "validation_mae"}
    return best_predict, info


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
                quantiles = model.forecast(
                    {"target": target, "target_mask": torch.ones_like(target, dtype=torch.bool),
                     "series_ids": torch.zeros(target.shape[:2], device=device, dtype=torch.long)},
                    horizon=h, decode_block_size=768, has_missing_values=False)
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

    runs = pd.read_csv(directory / "metrics.csv")
    windows = []
    for row_index, run in runs.iterrows():
        horizon = int(run.horizon)
        label = f"{run.model}_h{horizon}"
        if pd.notna(run.seed):
            label += f"_seed{int(run.seed)}"
        predictions = pd.read_csv(directory / f"{label}_predictions.csv")
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
        per_origin = per_origin.reset_index().assign(model=run.model, seed=run.seed, horizon=horizon)
        windows.append(per_origin)
    windows = pd.concat(windows, ignore_index=True)
    measures = ["mae", "rmse", "mase"]
    window_summary = windows.groupby(["model", "horizon", "seed"], dropna=False)[measures].agg(
        ["count", "mean", "std"])
    window_summary.columns = [f"{metric}_window_{stat}" for metric, stat in window_summary.columns]
    output = directory / "summary"
    output.mkdir(exist_ok=True)
    windows.to_csv(output / "per_origin_metrics.csv", index=False)
    window_summary.to_csv(output / "window_summary.csv")
    runs.to_csv(directory / "metrics.csv", index=False)
    (output / "README.txt").write_text(
        "per_origin_metrics.csv: each metric is computed across the forecast horizon at one origin.\n"
        "window_summary.csv: mean and sample std across origins, separately for each model/horizon/seed.\n"
        "Mean window RMSE is not pooled RMSE; metrics.csv retains pooled RMSE.\n"
        "metrics.csv includes *_window_mean and *_window_std for MAE, RMSE and MASE.\n"
        "Each fitted LSTM is reported separately; seed is an identifier only. No across-seed statistics are computed.\n"
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
    dates, values = load_data(args.data, args.download, args.input_mode)
    output = args.output or Path("results") / datetime.now(timezone.utc).strftime("etth1_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True, exist_ok=False)
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions() if d.metadata["Name"]}
    metadata = {"arguments": vars(args), "device": device, "hardware": hardware,
                "input_mode": args.input_mode, "features": FEATURES[:values.shape[1]], "target": "OT",
                "foundation_output": "Joint channel forecasts; only OT scored; no future covariates.",
                "packages": packages, "checkpoints": CHECKPOINTS, "data_url": DATA_URL,
                "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "split_end_indices_exclusive": [TRAIN_END, VALID_END, TEST_END],
                "split_dates": {"train": [str(dates[0]), str(dates[TRAIN_END - 1])],
                                "validation": [str(dates[TRAIN_END]), str(dates[VALID_END - 1])],
                                "test": [str(dates[VALID_END]), str(dates[TEST_END - 1])]},
                "pretraining_overlap": "Not audited; frozen inference does not establish unseen data.",
                "timing": "Warm adapter latency, CPU NumPy input/output, synchronized CUDA; excludes loading/training.",
                "smoke_test": args.max_origins is not None, "status": "running"}
    write_json(output / "run.json", metadata)
    rows, errors = [], []
    for horizon in args.horizons:
        origins = np.arange(VALID_END, TEST_END - horizon + 1, args.stride)
        if args.max_origins is not None:
            origins = origins[:args.max_origins]
        x, y = make_windows(values, origins, args.context, horizon)
        for name in args.models:
            for seed in args.seeds if name == "lstm" else [None]:
                label = f"{name}_h{horizon}" + (f"_seed{seed}" if seed is not None else "")
                print(f"Running {label}: {len(origins)} origins", flush=True)
                predict = None
                try:
                    model_device = "cpu" if name == "seasonal-naive" else device
                    synchronize(model_device)
                    begin = time.perf_counter()
                    info = {"training_seconds": 0.0}
                    if name == "seasonal-naive":
                        def predict(batch, h):
                            return batch[:, -args.season:, 0][:, np.arange(h) % args.season]
                    elif name == "lstm":
                        predict, info = train_lstm(values, horizon, args, device, seed, output)
                    else:
                        predict = foundation_adapter(name, device, args.batch_size)
                    synchronize(model_device)
                    setup_seconds = time.perf_counter() - begin
                    pred, timing = benchmark(predict, x, horizon, model_device, args.batch_size, args.warmup)
                    metrics = accuracy(y, pred, values[:TRAIN_END, 0], args.season)
                    row = {"model": name, "seed": seed, "horizon": horizon, "context": args.context,
                           "origins": len(origins), "batch_size": args.batch_size, "device": model_device,
                           "input_mode": "univariate" if name == "seasonal-naive" else args.input_mode,
                           "input_features": 1 if name == "seasonal-naive" else values.shape[1],
                           "setup_seconds_including_training": setup_seconds, **info, **metrics, **timing}
                    rows.append(row)
                    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
                    indices = origins[:, None] + np.arange(horizon)
                    pd.DataFrame({"origin": np.repeat(dates.iloc[origins].to_numpy(), horizon),
                                  "timestamp": dates.to_numpy()[indices].ravel(),
                                  "lead_hour": np.tile(np.arange(1, horizon + 1), len(origins)),
                                  "actual": y.ravel(), "prediction": pred.ravel()}).to_csv(
                                      output / f"{label}_predictions.csv", index=False)
                    print(f"  MAE={metrics['mae']:.5f} RMSE={metrics['rmse']:.5f}", flush=True)
                except Exception as exc:
                    error = {"run": label, "error": f"{type(exc).__name__}: {exc}"}
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
