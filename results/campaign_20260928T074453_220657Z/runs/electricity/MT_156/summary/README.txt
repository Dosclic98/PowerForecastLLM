per_origin_metrics.csv: each metric is computed across the forecast horizon at one origin.
window_summary.csv: mean and sample std across origins, separately for each model/horizon/seed.
Mean window RMSE is not pooled RMSE; metrics.csv retains pooled RMSE.
metrics.csv includes *_window_mean and *_window_std for MAE, RMSE and MASE.
Each fitted LSTM is reported separately; seed is an identifier only. No across-seed statistics are computed.
Window std is available for every model with at least two forecast origins.
New runs also record mean/std batch latency in metrics.csv. Old latency std cannot be reconstructed.
All std use ddof=1. Windows may overlap and be serially correlated; std are descriptive, not confidence intervals.
