dataset_summary.csv compares the same series across all requested models at each horizon.
Series missing any requested model are excluded for that horizon; coverage.csv lists every expected result.
mean/std describe individual forecast-window errors (or full inference-batch latency), pooled across eligible series.
series_mean/series_std describe the distribution of per-series means, giving customers equal weight.
metric_series excludes undefined metrics, e.g. MASE with zero training scale. Counts are explicit.
All std use ddof=1; fewer than two observations give an empty std, not zero.
Overlapping or serially correlated windows mean std is descriptive, not a confidence interval.
Mean window RMSE differs from pooled RMSE, retained in per_series_metrics.csv.
MAE/RMSE retain each target's units; datasets are never pooled. MASE uses training-only lag --season.
resource_summary.csv gives mean/std across successful series, including runs outside the matched cohort.
Latency excludes setup/training and uses full batches only. Training cost includes CV and final fitting.
One LSTM seed is used; these are not across-seed standard deviations.
Raw forecasts, per-origin metrics, run metadata and CV artifacts live in each job directory.
