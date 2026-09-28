import argparse
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import benchmark_etth1 as b


class LSTMValidationTests(unittest.TestCase):
    def test_folds_have_disjoint_targets_and_growing_training(self):
        args = b.parse_args(['--cv-folds', '5'])
        self.assertEqual(args.cv_folds, 5)
        previous_end = args.valid_end // 6
        previous_train = 0
        for fold in b.cross_validation_folds(args):
            config = argparse.Namespace(**{**vars(args), **fold})
            stopping, scoring = b.validation_origins(168, config)
            self.assertGreater(fold['fit_train_end'], previous_train)
            self.assertEqual(fold['fit_score_start'], previous_end)
            self.assertGreaterEqual(stopping[0], fold['fit_train_end'])
            self.assertLessEqual(stopping[-1] + 168, scoring[0])
            self.assertEqual(scoring[0], fold['fit_score_start'])
            self.assertLessEqual(scoring[-1] + 168, fold['fit_valid_end'])
            self.assertLessEqual(fold['fit_valid_end'], b.VALID_END)
            self.assertTrue(np.all(np.diff(scoring) >= 168))
            previous_end = fold['fit_valid_end']
            previous_train = fold['fit_train_end']
        self.assertEqual(previous_end, b.VALID_END)

    def test_default_is_one_fixed_configuration(self):
        args = b.parse_args([])
        expected = dict(hidden_size=128, learning_rate=.001, layers=2, dropout=.3,
                        weight_decay=0., output_mode='last', context=512, train_batch_size=64)
        self.assertEqual(b.lstm_grid(args), [expected])
        overridden = b.parse_args(['--context', '168'])
        self.assertEqual(b.lstm_grid(overridden), [{**expected, 'context': 168}])
        self.assertEqual(b.lstm_grid(b.parse_args(['--no-tune-lstm'])), [expected])

    def test_grid_deduplicates_and_rejects_invalid_contexts(self):
        args = b.parse_args(['--cv-folds', '5', '--lstm-contexts', '24', '168', '24',
                             '--lstm-layers', '1', '2', '--lstm-output-modes', 'direct', 'last'])
        self.assertEqual(len(b.lstm_grid(args)), 8)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            b.parse_args(['--lstm-contexts', '1024'])
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            b.parse_args(['--cv-folds', '100', '--horizons', '168'])

    def test_selection_uses_mean_fold_score_and_refits_without_test(self):
        args = b.parse_args(['--lstm-hidden-sizes', '4', '8', '--lstm-learning-rates', '.001',
                             '--cv-folds', '2', '--lstm-contexts', '512'])
        values = np.zeros((b.TEST_END, 1), dtype=np.float32)
        values[b.VALID_END:] = np.nan
        calls = []

        def fake_fit(development, horizon, config, device, seed, output):
            self.assertEqual(len(development), config.fit_valid_end)
            self.assertTrue(np.isfinite(development).all())
            calls.append(config)
            # Width 4 has the best individual fold, but width 8 has the best mean.
            score = ([1., 9.][config.fold - 1] if config.hidden_size == 4 else 4.) if hasattr(config, 'fold') else 20.
            return config.hidden_size, dict(validation_mae=score if not getattr(config, "refit_full", False) else None, training_seconds=1,
                                           best_epoch=config.fold + 1 if hasattr(config, "fold") else config.epochs,
                                           validation_seasonal_naive_mae=6., validation_persistence_mae=5.)

        with tempfile.TemporaryDirectory() as directory, patch.object(b, 'fit_lstm_candidate', fake_fit):
            predict, info = b.train_lstm(values, 24, args, 'cpu', 42, Path(directory))
            self.assertEqual(predict, 8)
            self.assertEqual(info['trial'], 2)
            self.assertEqual(info['cv_mae'], 4.)
            self.assertIsNone(info['validation_mae'])
            self.assertEqual(info['refit_epochs'], 3)
            self.assertEqual(len(calls), 5)
            self.assertEqual(calls[-1].fit_train_end, b.VALID_END)
            self.assertTrue(calls[-1].refit_full)
            self.assertEqual(calls[-1].epochs, 3)
            selection = json.loads((Path(directory) / 'lstm_h24_seed42_selection.json').read_text())
            self.assertFalse(selection['test_used_for_selection'])
            self.assertEqual(selection['selection_metric'], 'cv_mae')

    def test_training_checkpoint_and_residual_translation(self):
        torch.set_num_threads(1)
        # Tiny chronological experiment exercises both model variants and dropout.
        values = np.random.default_rng(7).normal(size=(160, 2)).astype(np.float32)
        args = b.parse_args(['--context', '24', '--epochs', '2', '--train-batch-size', '16',
                             '--hidden-size', '4', '--stride', '4'])
        with patch.multiple(b, VALID_END=128, TEST_END=160):
            for mode in ['direct', 'last']:
                config = argparse.Namespace(**{**vars(args), 'weight_decay': 0.,
                                                'dropout': .2, 'output_mode': mode,
                                                'fit_train_end': 72, 'fit_score_start': 96, 'fit_valid_end': 120})
                with tempfile.TemporaryDirectory() as directory:
                    predict, info = b.fit_lstm_candidate(values[:120], 4, config, 'cpu', 42, Path(directory))
                    # Longer incoming context must be cropped to the trained context.
                    x = values[100:132][None].copy()
                    pred = predict(x, 4)
                    self.assertEqual(pred.shape, (1, 4))
                    np.testing.assert_allclose(pred, predict(x[:, -24:], 4))
                    self.assertTrue(np.isfinite(pred).all())
                    if mode == 'last':
                        shifted = x.copy()
                        shifted[:, :, 0] += 10
                        np.testing.assert_allclose(predict(shifted, 4), pred + 10, atol=2e-6)
                    checkpoint = torch.load(Path(directory) / 'lstm_h4_seed42.pt', weights_only=True)
                    np.testing.assert_allclose(checkpoint['mean'], values[:72].mean(axis=0))
                    self.assertEqual(checkpoint['output_mode'], mode)
                    self.assertEqual(checkpoint['dropout_placement'], 'between_lstm_layers')
                    self.assertEqual(checkpoint['effective_dropout'], .2)
                    self.assertEqual(info['train_windows'], 45)
                    self.assertTrue(np.isfinite(info['selected_train_mae']))
                    self.assertIsNone(info['validation_block_mae_std'])

    def test_interlayer_dropout_and_single_layer_override(self):
        torch.set_num_threads(1)
        values = np.random.default_rng(2).normal(size=(64, 2)).astype(np.float32)
        args = b.parse_args(['--context', '24', '--hidden-size', '4', '--epochs', '1'])
        original_lstm = torch.nn.LSTM
        for layers in (1, 2):
            encoders = []

            def build_encoder(*args, **kwargs):
                encoder = original_lstm(*args, **kwargs)
                encoders.append(encoder)
                return encoder

            config = argparse.Namespace(**{**vars(args), 'layers': layers, 'weight_decay': 0.,
                                           'dropout': .3, 'output_mode': 'last', 'refit_full': True,
                                           'fit_train_end': 64, 'fit_valid_end': 64})
            with tempfile.TemporaryDirectory() as directory, \
                    patch.object(torch.nn, 'LSTM', side_effect=build_encoder), \
                    patch.object(torch.nn, 'Dropout', side_effect=AssertionError('output dropout must be absent')):
                predict, info = b.fit_lstm_candidate(values, 4, config, 'cpu', 42, Path(directory))
                self.assertEqual(encoders[0].dropout, .3 if layers == 2 else 0.)
                self.assertEqual(info['effective_dropout'], encoders[0].dropout)
                x = values[-24:][None].copy()
                np.testing.assert_array_equal(predict(x, 4), predict(x, 4))
                if layers == 2:
                    encoders[0].train()
                    with torch.no_grad():
                        first = encoders[0](torch.from_numpy(x))[0]
                        second = encoders[0](torch.from_numpy(x))[0]
                    self.assertFalse(torch.equal(first, second))

    def test_single_holdout_selects_epoch_then_refits_full_development(self):
        torch.set_num_threads(1)
        args = b.parse_args(['--models', 'lstm', '--split-ends', '192', '240',
                             '--context', '8', '--season', '1', '--horizons', '4',
                             '--early-stopping-hours', '32', '--epochs', '3', '--hidden-size', '4',
                             '--train-stride', '8', '--validation-stride', '4', '--train-batch-size', '8'])
        self.assertEqual(args.cv_folds, 1)
        self.assertEqual(args.train_end, 160)
        values = np.random.default_rng(42).normal(size=(240, 1)).astype(np.float32)
        values[192:] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(b, 'fit_lstm_candidate', wraps=b.fit_lstm_candidate) as fit:
                predict, info = b.train_lstm(values, 4, args, 'cpu', 42, output)
            self.assertEqual(fit.call_count, 2)
            self.assertEqual(fit.call_args_list[0].args[2].fit_train_end, 160)
            self.assertEqual(fit.call_args_list[1].args[2].fit_train_end, 192)
            selection = json.loads((output / 'lstm_h4_seed42_selection.json').read_text())
            self.assertEqual(info['refit_epochs'], selection['epoch_selection']['best_epoch'])
            self.assertEqual(info['epochs_run'], info['refit_epochs'])
            self.assertEqual(info['train_windows'], 23)
            self.assertEqual(info['cv_folds'], 0)
            self.assertIsNone(info['validation_mae'])
            self.assertFalse(selection['test_used_for_selection'])
            self.assertFalse(list(output.glob('*_cv_folds.csv')))
            self.assertTrue(np.isfinite(predict(values[184:192][None], 4)).all())
            selection_checkpoint = torch.load(output / 'lstm_h4_seed42_epoch_selection' / 'lstm_h4_seed42.pt', weights_only=True)
            final_checkpoint = torch.load(output / 'lstm_h4_seed42.pt', weights_only=True)
            np.testing.assert_allclose(selection_checkpoint['mean'], values[:160].mean(axis=0))
            np.testing.assert_allclose(final_checkpoint['mean'], values[:192].mean(axis=0))

    def test_holdout_rejects_search_and_impossible_tail(self):
        for flags in (['--lstm-hidden-sizes', '64', '128'], ['--early-stopping-hours', '12000']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                b.parse_args(flags)

    def test_five_fold_search_and_full_refit_end_to_end(self):
        torch.set_num_threads(1)
        args = b.parse_args(['--models', 'lstm', '--cv-folds', '5', '--split-ends', '192', '240',
                             '--context', '8', '--season', '1', '--horizons', '4',
                             '--cv-stop-hours', '8', '--epochs', '2', '--hidden-size', '4',
                             '--train-stride', '8', '--validation-stride', '4',
                             '--train-batch-size', '8', '--no-tune-lstm'])
        values = np.random.default_rng(42).normal(size=(240, 1)).astype(np.float32)
        values[192:] = np.nan  # Any accidental fitting/scaling on test data must fail.
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            predict, info = b.train_lstm(values, 4, args, 'cpu', 42, output)
            selection = json.loads((output / 'lstm_h4_seed42_selection.json').read_text())
            folds = pd.read_csv(output / 'lstm_h4_seed42_cv_folds.csv')
            epochs = int(np.ceil(np.median(folds.best_epoch)))
            self.assertEqual(len(folds), 5)
            self.assertEqual(info['epochs_run'], epochs)
            self.assertEqual(info['refit_epochs'], epochs)
            self.assertEqual(info['train_end_exclusive'], 192)
            self.assertEqual(info['train_windows'], 23)
            self.assertTrue(info['refit_full'])
            self.assertFalse(selection['test_used_for_selection'])
            self.assertIsNone(info['early_stopping_mae'])
            self.assertTrue(np.isfinite(predict(values[184:192][None], 4)).all())

    def test_full_refit_uses_all_development_without_validation(self):
        torch.set_num_threads(1)
        values = np.arange(320, dtype=np.float32).reshape(160, 2) / 100
        args = b.parse_args(['--context', '24', '--epochs', '2', '--hidden-size', '4',
                             '--train-batch-size', '16'])
        config = argparse.Namespace(**{**vars(args), 'weight_decay': 0., 'dropout': 0.,
                                       'output_mode': 'last', 'refit_full': True,
                                       'fit_train_end': 128, 'fit_valid_end': 128})
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(b, 'validation_origins', side_effect=AssertionError('no validation in full refit')):
            predict, info = b.fit_lstm_candidate(values[:128], 4, config, 'cpu', 42, Path(directory))
            checkpoint = torch.load(Path(directory) / 'lstm_h4_seed42.pt', weights_only=True)
            np.testing.assert_allclose(checkpoint['mean'], values[:128].mean(axis=0))
            self.assertEqual(info['train_windows'], 128 - 24 - 4 + 1)
            self.assertEqual(info['epochs_run'], 2)
            self.assertIsNone(info['validation_mae'])
            self.assertIsNone(info['early_stopping_mae'])
            self.assertEqual(checkpoint['validation_blocks'], 0)
            self.assertTrue(checkpoint['refit_full'])
            self.assertTrue(np.isfinite(predict(values[104:128][None], 4)).all())


if __name__ == '__main__':
    unittest.main()
