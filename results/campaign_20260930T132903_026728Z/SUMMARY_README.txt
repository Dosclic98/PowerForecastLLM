dataset_summary.csv contains one result per repetition, comparing the same series across all requested models/repetitions at each horizon.
Series missing any requested model/repetition are excluded for that horizon; coverage.csv lists every expected result.
mean/std describe individual forecast-window errors (or full inference-batch latency), pooled across eligible series.
series_mean/series_std describe the distribution of per-series means, giving customers equal weight.
metric_series excludes undefined metrics, e.g. MASE with zero training scale. Counts are explicit.
All std use ddof=1; fewer than two observations give an empty std, not zero.
Overlapping or serially correlated windows mean std is descriptive, not a confidence interval.
Mean window RMSE differs from pooled RMSE, retained in per_series_metrics.csv.
MAE/RMSE retain each target's units; datasets are never pooled. MASE uses training-only lag --season.
resource_summary.csv gives mean/std across successful series WITHIN each repetition, including runs outside the matched cohort.
Latency excludes setup/training and uses full batches only. Training cost includes CV and final fitting.
repeat_summary.csv gives mean/std across repeated setups/fits per series; fixed seed, with completed/requested counts.
dataset_repeat_summary.csv gives mean/std across repetitions of the matched-cohort dataset means.
Window std, customer std, and repetition std measure different variation and are never pooled together.
LSTM workload includes all executed optimization/early-stopping batches, even epochs after the best checkpoint.
Diagnostic evaluation after fitting is excluded from fit batch counts. Initial/refit window counts overlap and are not summed as unique windows.
Setup is repeated in-process; downloads, imports and filesystem caches can make the first repetition slower.
Raw forecasts, per-origin metrics, run metadata and CV artifacts live in each job directory.
