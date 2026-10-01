"""Sample statistics across repeated setups/fits, not across forecast windows."""
import pandas as pd

REPEAT_METRICS = (
    'setup_seconds_including_training', 'training_seconds', 'epoch_selection_training_seconds',
    'selected_training_seconds', 'inference_seconds', 'mean_batch_latency_ms', 'forecasts_per_second',
    'initial_train_windows', 'refit_train_windows', 'initial_epochs_run', 'refit_epochs',
    'initial_training_batches', 'refit_training_batches', 'selection_training_batches',
    'total_training_batches', 'total_early_stopping_batches', 'total_fit_batches',
    'total_training_window_presentations', 'mae', 'rmse', 'mase',
    'mae_window_mean', 'rmse_window_mean', 'mase_window_mean',
)


def repetition_summary(frame, keys, metrics, requested_repeats):
    """Each value contributes once per repetition; report incomplete groups explicitly."""
    columns = [*keys, 'metric', 'repeat_count', 'requested_repeats', 'mean', 'std']
    records = []
    if not frame.empty:
        if frame.duplicated([*keys, 'repeat']).any():
            raise ValueError('Duplicate rows for a repetition in repetition summary.')
        for values, group in frame.groupby(keys, dropna=False, sort=False):
            if len(keys) == 1:
                values = (values,)
            for metric in metrics:
                if metric not in group:
                    continue
                samples = pd.to_numeric(group[metric], errors='coerce')
                if not samples.count():
                    continue
                records.append({**dict(zip(keys, values)), 'metric': metric,
                                'repeat_count': int(samples.count()), 'requested_repeats': requested_repeats,
                                'mean': samples.mean(), 'std': samples.std(ddof=1)})
    return pd.DataFrame(records, columns=columns)
