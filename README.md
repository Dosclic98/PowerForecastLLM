# PowerForecastLLM
Evaluating the forecasting & anomaly detection capabilities of various LLM foundation models within the power domain.

## One campaign across all three datasets

Run the full comparison with one command (using the Python 3.12 benchmark environment):

```bash
.venv-benchmark/bin/python benchmark_campaign.py --download --device cuda
```

This evaluates `seasonal-naive`, `lstm`, `chronos2`, `timesfm3` and `toto2` on
**ETTh1, electricity and household**, at horizons **24 and 168 hours**.
Electricity defaults to a fixed subset of **10 customers**, fitted/evaluated independently.
The default campaign has **120 model evaluations and 48 LSTM fits**.
The client IDs are MT_124, MT_131, MT_132, MT_156, MT_158, MT_159, MT_161,
MT_162, MT_163 and MT_166, selected for observation availability in the initial
2011 training period. This avoids pre-activation gaps in the earliest CV folds; the
selection does not use test targets or forecasting performance.
Inspect the budget without downloading data or running models:

```bash
.venv-benchmark/bin/python benchmark_campaign.py --plan
```

Campaign settings are shared across every dataset:

| Setting | Campaign default |
|---|---|
| Sampling | Hourly, preserving missing timestamps/values |
| Input | Target-only (`--input-mode univariate`) for every model |
| Available history | 512 hours; fixed to the requested `--context` for the LSTM |
| Horizons | 24 and 168 hours |
| Test origins | Every 24 hours, identical across models within each series/horizon |
| Split | 80% development / 20% test of the full record, development boundary rounded down to a whole day |
| LSTM selection | Fixed configuration; one chronological holdout selects epoch count, seed 42; final refit on all development data |
| Foundation models | Frozen inference; no dataset-specific training |
| Seasonal-naive | Copy the last H target values, in chronological order, to forecast H hours |
| MASE scale | Differences over the full development period at lag 24 (`--season`), shared by every model |
| Timing | Same inference batch size and warmup; setup/training reported separately |

Toto 2 requires its input length to be a multiple of its patch size (32 for the
current checkpoint). The adapter left-pads non-aligned contexts with masked,
unobserved positions: 168 observed hours occupy 192 input positions. No extra
history is supplied, the forecast origin stays unchanged, and the requested
horizon is preserved. This padding is included in adapter latency.

The campaign uses **the full ETTh1 record with the same fractional split policy as
other datasets**. Its results are therefore separate from the legacy
`benchmark_etth1.py` default 16-development/4-test-month timeline described below. ETTh1 predicts oil
temperature; electricity and household predict power. Raw MAE/RMSE are reported
separately by dataset, never averaged across temperature and power.

Override shared settings once, for example:

```bash
.venv-benchmark/bin/python benchmark_campaign.py --download \
  --models seasonal-naive lstm --horizons 24 168 --device cuda \
  --clients MT_124 MT_131 MT_132 --output results/my_campaign
```

Use `--datasets` to choose a subset. All engine model/training options apply to
every dataset; see `benchmark_etth1.py --help`. `--input-mode multivariate` is an
explicit alternative: ETTh1 and household then have auxiliary historical
channels, while electricity remains univariate. `--data-dir` defaults to `data`;
`--etth1-data` can override the ETTh1 CSV path. `--prepare-only` prepares inputs
and saves a validated job manifest without running models. Output directories
must be new, so existing results are never overwritten.

For a quick pipeline check across all datasets:

```bash
.venv-benchmark/bin/python benchmark_campaign.py --download \
  --models seasonal-naive --clients MT_001 --max-origins 3
```

Campaign output contains:

- `campaign.json`: shared settings, source/code hashes, split boundaries, exact
  per-series arguments, requested jobs and completion/failure status.
- `dataset_summary.csv`: MAE, RMSE, MASE and batch latency by dataset/model/horizon.
  `mean`/`std` describe forecast windows (full batches for latency), pooling the
  actual observations with correct sample-variance formulas.
  `series_mean`/`series_std` describe the per-customer means with equal customer
  weights. For single-series datasets, series std is undefined.
- `per_series_metrics.csv`: all successful per-series results, including pooled
  RMSE, window mean/std, timings, repetition IDs and LSTM workload diagnostics.
- `resource_summary.csv`: mean/std across series within each repetition for
  training/setup/inference time and throughput, including all successful runs.
- `coverage.csv`: every requested series/model/horizon and whether it completed.
- `runs/DATASET/SERIES/`: predictions, per-origin metrics, CV results, checkpoints,
  hardware/package details, and individual error/coverage reports.

For each dataset/horizon, comparison summaries use only customers with results
for **all requested models and repetitions**, so a failed run cannot silently change the
comparison cohort. Requested/completed/eligible counts are explicit. Undefined
MASE values are excluded and metric-specific counts are reported. Standard
deviations use `ddof=1`, are blank for fewer than two observations, and describe
variation rather than confidence intervals. They are not across-seed variation.
Mean window RMSE is distinct from pooled RMSE. Horizons greater than the stride
have overlapping windows.

Jobs run sequentially to avoid competing for GPU memory. Failed jobs do not stop
later jobs; incomplete campaigns exit with a nonzero status. Rebuild summaries
without rerunning models with:

```bash
.venv-benchmark/bin/python benchmark_campaign.py --summarize results/my_campaign
```

## Repeated training and setup measurements

Use `--repeats 10` to perform ten fresh fits/setups for every model, series and
horizon. For the current multivariate experiment:

```bash
.venv-benchmark/bin/python benchmark_campaign.py \
  --download --device cuda --input-mode multivariate \
  --context 336 --early-stopping-hours 1440 --repeats 10
```

The default remains one repetition. Ten repetitions mean **480 LSTM fits**
(12 series × 2 horizons × 2 fits × 10 repetitions) and **1,200 model evaluations**
for all five models. `--plan --repeats 10` previews the budget without execution.

Every repetition uses the **same seed**, fixed configuration, split and forecast
windows. It retrains both LSTM stages from scratch and recreates each pretrained
adapter, reloads its weights, warms it up, and evaluates forecasts. This measures
repeatability of timing at a controlled workload, not variability across seeds.
Repetitions are sequential in the same process for each series. No library,
filesystem or checkpoint cache is flushed: initial imports/downloads can make the
first setup slower. Per-repetition measurements are retained for inspection.

- `per_series_metrics.csv` has one row per series/model/horizon/**repeat**.
- `repeat_summary.csv` reports mean/sample std **across repetitions**, separately
  for each series/model/horizon. Filter `metric` to `training_seconds` or
  `setup_seconds_including_training` for the requested timing statistics. It also
  summarizes accuracy and workload counts. Completed/requested repeat counts are
  explicit; a single successful repeat has undefined std.
- `dataset_summary.csv` and `resource_summary.csv` retain window/customer
  statistics separately **within each repeat**.
- `dataset_repeat_summary.csv` summarizes dataset-level mean accuracy and latency
  across repetitions. `statistic=mean` uses pooled windows; `series_mean` uses
  equal customer weights. These std values differ from window/customer std.
- With multiple repeats, raw forecasts, checkpoints and selection diagnostics go
  under `runs/DATASET/SERIES/repeat_01/`, `repeat_02/`, etc. A single repeat keeps
  the original layout. Each series' `summary/repeat_summary.csv` also reports its
  repetition statistics.

Dataset comparisons use a common customer cohort across all requested models
**and repetitions**. A missing repetition is recorded in `coverage.csv`; its
customer is excluded from that horizon's dataset comparisons. Individual
successful measurements remain available in `repeat_summary.csv`.

LSTM workload columns are stored in `metrics.csv`, the combined per-series CSV,
and the selection JSON:

| Column | Meaning |
|---|---|
| `initial_train_windows` | Valid optimization windows in the initial fit, excluding the holdout |
| `refit_train_windows` | Valid windows in the final full-development refit |
| `initial_epochs_run` | All epochs executed, including patience epochs after the best epoch |
| `refit_epochs` | Epochs executed during final refitting |
| `initial_training_batches`, `refit_training_batches` | Actual optimizer batches processed in each stage |
| `total_training_batches` | Optimizer batches across both stages |
| `total_early_stopping_batches` | Validation batches processed while deciding when to stop |
| `total_fit_batches` | Optimizer plus early-stopping batches across both stages |
| `total_training_window_presentations` | Training examples processed, counting repeated epochs |

Partial batches count as one batch. Unique window counts are reported per stage;
the initial and final window sets overlap, so their sum is not a unique-data count.
Batch counters exclude post-fit diagnostic scoring and test inference, which are
outside the optimization/early-stopping `training_seconds` timer. For optional
multi-fold CV, totals include **all** candidate/fold fits plus the final refit;
`selection_training_batches` reports candidate/fold optimizer batches.

## LSTM early stopping and full-development refit

The first **80%** is development data; the last **20%** is an untouched test set.
With the fixed LSTM configuration, the default now makes **two fits per task**:

1. Fit on the development prefix, reserving its last **720 hours** for early
   stopping. Scaling uses only the optimization prefix. Monitor holdout MAE and
   record the best epoch (not the later epoch at which patience expires).
2. Initialize a fresh model and fit on **all development data** for exactly that
   many epochs, with scaling fitted on all development observations. Evaluate
   this final model on test targets. There is no stopping holdout in the refit.

`--early-stopping-hours` controls the holdout length; `--cv-stop-hours` is a
backward-compatible alias. For a 600-day record and the default 30-day holdout,
the first fit trains on days 1–450 and stops using days 451–480. The final fit
uses days 1–480; testing covers days 481–600. Test data never selects epochs.
The same split rule and fixed configuration apply to all datasets/customers and
horizons, although each task can select a different epoch count.

The default `--cv-folds 1` means a single early-stopping holdout, not k-fold CV.
Optional `--cv-folds 5` restores the earlier expanding-window protocol: six
chronological development blocks supply an initial history and five scoring
folds; each fold has a separate inner stopping tail. Explicit parameter searches
require `--cv-folds` of at least 2. Only that optional mode uses mean outer-fold
MAE and the ceiling of median best fold epochs for final refitting.

MASE uses all development data. This is campaign protocol 5; existing results
remain unchanged. The standalone ETTh1 runner retains its 20-month timeline
(30-day months), using 16 development months and four test months. The campaign
uses the full record. Engine `--split-ends` accepts two exclusive endpoints:
`DEVELOPMENT TEST`.

Validation spacing defaults to `max(--stride, horizon)` to avoid overlapping
targets. All methods use the same test origins, whose default daily spacing still
produces overlapping targets at horizon 168. Past observations from earlier
validation/test origins can be used as subsequent history. No future load values
or future OT values are supplied as inputs. Dates only validate ordering and label
outputs. Neither non-overlapping folds nor fold standard deviations establish
statistical independence or confidence intervals.

## Fixed default LSTM

The same default configuration is used for every dataset, customer and horizon:

- Hidden size: **128 per layer**.
- Learning rate: **0.001**.
- Layers: **2**.
- Dropout: **0.3 between the two LSTM layers**. No dropout before the linear
  forecasting head. A one-layer override disables inter-layer dropout.
- Weight decay: **0**.
- Output mode: **last** (predict a residual relative to the last target value).
- Context: the requested **`--context`**, with no automatic context search.

Each task trains separate weights, and the output head has one value per forecast
hour. One early-stopping fit plus one full-development refit give **2 fits per
horizon**, or **4 per series** at horizons 24 and 168. Explicit multi-value
`--lstm-*` options require `--cv-folds` of at least 2 for parameter search. `--no-tune-lstm` is no longer necessary for the
default fixed configuration.

Activate the benchmark environment described in `python benchmark_etth1.py --help`,
then run:

```bash
.venv-benchmark/bin/python benchmark_etth1.py \
  --models seasonal-naive lstm --device cuda \
  --horizons 24 168 --seeds 42
```

Add `--lstm-plan` to print the configuration, holdout boundaries and fit count
without loading data or accessing CUDA. `--epochs` and `--patience` default to
30 and 5 for the initial early-stopping fit. `--no-tune-lstm` is optional and
keeps one configuration even if search options were supplied.

Optional ablations include `--lstm-contexts 24 168 512`,
`--lstm-output-modes direct last`, and `--input-mode univariate`. `last` centers OT
history on its latest observation and predicts an additive change; the other
channels retain training-only global scaling. These overrides permit further controlled comparisons. The default `last` mode
is supported by earlier validation results, but its CV performance still needs evaluation. Contexts must fit within
`--context`; inference crops inputs to the selected LSTM context and records it
in metrics. Dropout applies between stacked layers and is inactive with a single layer.

## Diagnostics and interpretation

- `*_epoch_selection/`: checkpoint, epoch history and holdout metrics from the
  initial fit. Its best epoch determines final training duration.
- `*_selection.json`: fixed configuration, optimization/holdout boundaries,
  selected epoch, final-fit details and explicit test-exclusion metadata.
- Root-level checkpoint/history files belong to the full-development refit.
- `metrics.csv`: `epoch_selection_mae` records the holdout result; `refit_epochs`
  records training duration. Final `validation_mae` and `early_stopping_mae` are
  empty because that fit has no validation set. `selected_train_mae` is a training
  diagnostic. `cv_folds=0` and `validation_protocol=single_holdout` identify the
  default method. No CV-fold mean/std is reported for a single holdout.
- Optional multi-fold CV still writes `*_cv_folds.csv`, `*_tuning.csv`, and trial
  checkpoints. Its selection metric is `cv_mae`; those artifacts are not generated
  by the default single-holdout workflow.

`training_seconds` includes both fits; `selected_training_seconds` is the final
fit only. `tuning_seconds` also includes selection diagnostics and checkpoint I/O.

Verification:

```bash
.venv-benchmark/bin/python -m unittest discover -s tests -v
```

## Additional power datasets

`benchmark_power.py` uses the same LSTM CV and foundation-model adapters for:

| CLI name | Source | Forecast target |
|---|---|---|
| `electricity` | [ElectricityLoadDiagrams20112014, UCI](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014), Trindade, DOI [10.24432/C58C86](https://doi.org/10.24432/C58C86) | Each of 370 customers' hourly mean power (kW), independently |
| `household` | [Individual Household Electric Power Consumption, UCI](https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption), Hebrail & Berard, DOI [10.24432/C58K54](https://doi.org/10.24432/C58K54) | Hourly mean `Global_active_power` (kW) |

Both sources are CC BY 4.0. Downloads, prepared hourly CSVs and preprocessing
manifests are stored under `data/power/` (ignored by Git). Source archives and
preprocessor code are identified by SHA-256 in the manifests. Initial downloads
are about 250 MB for electricity and 20 MB for household; prepared files require
additional disk space. No new dependencies beyond the benchmark environment are
needed (use its Python 3.12 interpreter).

Download and prepare without model execution:

```bash
.venv-benchmark/bin/python benchmark_power.py --dataset electricity --download --prepare-only
.venv-benchmark/bin/python benchmark_power.py --dataset household --download --prepare-only
```

Run the household benchmark or a small, explicitly selected electricity panel:

```bash
.venv-benchmark/bin/python benchmark_power.py --dataset household --device cuda
.venv-benchmark/bin/python benchmark_power.py --dataset electricity \
  --clients MT_124 MT_131 MT_132 --device cuda
```

Defaults include seasonal-naive, LSTM, Chronos 2,
TimesFM 3 and Toto 2, at horizons 24 and 168. `--models` overrides the selection;
for example, `--models seasonal-naive` needs no GPU.
Other model options, including CV/grid options, `--input-mode univariate`,
`--max-origins` and `--context`, are forwarded to `benchmark_etth1.py`.
For all datasets, seasonal-naive copies the last horizon-length block of target
values in chronological order: the last 24 hours for horizon 24, or the last
168 hours for horizon 168. Context must cover the largest requested horizon.
LSTM validation reports the same baseline, including for shorter LSTM trial
contexts. `--season` (24 by default) controls the shared training-derived MASE
denominator, independently of the baseline forecast.

Omit `--clients` to evaluate the same fixed **10-client subset** as the campaign.
Override it with explicit client IDs. Each client gets an independent LSTM search
and final model; this is not a globally trained panel LSTM. At the default grid
and horizons, ten clients entail **40 LSTM fits**. Use `--lstm-plan` to inspect
actual folds and fit counts. The subset is not claimed to represent all 370 clients.

### Splits, units and missing data

Both power datasets use **80% development / 20% test** of the full hourly timeline,
with the development boundary rounded down to a whole day. Electricity clients
share dates. One trailing holdout selects the epoch count inside development;
the final model trains on all development data for that fixed duration.

Preprocessing deliberately preserves missing hours on a regular timeline:

- Electricity interval-end timestamps are shifted back 15 minutes before hourly
  aggregation. All four quarter-hour measurements are required. Leading zero
  placeholders before the first nonzero reading are masked; later zeros remain
  valid. Both complete daylight-saving transition days (last Sundays in March
  and October) are masked because the source encodes artificial zeros and
  combined readings rather than a conventional timezone-aware clock.
- Household hours require all 60 minute readings for each channel. Power,
  voltage and current are hourly means; submeter energy channels are hourly
  sums in Wh. Partial boundary hours remain missing. By default, the model sees
  the target and six other historical electrical channels; univariate mode uses
  only active power.
- No interpolation or imputation is performed. A forecast window is excluded if
  any model input or future target is missing. Missing future covariates do not
  disqualify a window because they are not supplied to the model. All candidates
  use the full requested context to select eligible origins, even if a candidate
  uses a shorter context, ensuring comparable validation windows.
- Scaling uses finite observations from each fold's training prefix and all
  development observations for the final refit. MASE uses finite development pairs at the actual seasonal lag, without compressing gaps.
  Empty training/validation blocks fail explicitly instead of dropping folds.

Long contexts can leave no complete household holdout windows. Inspect coverage
and adjust the context or holdout length if needed; missing windows are never
imputed. `--context 168 --early-stopping-hours 1440` is an available common setup.
The optional five-fold protocol has additional household coverage constraints.

Strict completeness can remove many windows, especially household week-ahead
forecasts with long history. Inspect coverage before interpreting aggregate
accuracy. The source is used as published; its measurement availability and
foundation-model pretraining overlap have not been independently audited.

### Outputs

Each series has the normal benchmark outputs in its own subdirectory.
`coverage_h*.json` records scheduled, eligible and excluded test origins before
any `--max-origins` smoke-test limit. Fold validation CSVs similarly record
scheduled/excluded origins. `dataset.json` captures preprocessing, series selection
and split provenance, while `status.json` records failed series. Failures yield a
nonzero exit code and do not prevent remaining customers from running.

`per_series_metrics.csv` retains every model/customer result. `macro_metrics.csv`
reports equal-weight averages of per-series MAE, RMSE and MASE, with completed and
requested series counts. This is not a pooled RMSE; MASE has its own valid-series
count when a constant series has an undefined denominator. Check matching series
coverage before comparing models with partial failures. Metrics are never pooled
across the two datasets or with ETTh1.
