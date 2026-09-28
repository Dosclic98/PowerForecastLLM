import importlib.util
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

import benchmark_etth1 as benchmark


class TotoAdapterTests(unittest.TestCase):
    def test_preserves_history_origin_and_mask_for_each_context(self):
        class FakeModel:
            config = types.SimpleNamespace(patch_size=32)

            def to(self, device):
                return self

            def eval(self):
                return self

            def forecast(self, inputs, **kwargs):
                self.inputs, self.options = inputs, kwargs
                target = inputs['target']
                assert target.shape[-1] % self.config.patch_size == 0
                # Distinct quantiles/channels test median and target extraction.
                return (target[..., -1:].expand(9, *target.shape[:2], kwargs['horizon'])
                        + torch.arange(9)[:, None, None, None])

        for context in (1, 24, 168, 192, 512, 513):
            for channels in (1, 7):
                with self.subTest(context=context, channels=channels):
                    model = FakeModel()
                    factory = types.SimpleNamespace(from_pretrained=lambda *_args: model)
                    with patch.dict(sys.modules, {'toto2': types.SimpleNamespace(Toto2Model=factory)}):
                        predict = benchmark.foundation_adapter('toto2', 'cpu', 2)
                    x = np.arange(2 * context * channels, dtype=np.float32).reshape(2, context, channels)
                    original = x.copy()
                    result = predict(x, 168)
                    padding = (-context) % 32
                    target, mask = model.inputs['target'], model.inputs['target_mask']
                    np.testing.assert_array_equal(target[..., padding:].numpy(), x.transpose(0, 2, 1))
                    self.assertTrue(mask[..., padding:].all())
                    self.assertFalse(mask[..., :padding].any())
                    self.assertEqual(model.options['has_missing_values'], bool(padding))
                    self.assertEqual(model.options['horizon'], 168)
                    self.assertEqual(result.shape, (2, 168))
                    np.testing.assert_array_equal(result, np.repeat(x[:, -1, 0:1] + 4, 168, axis=1))
                    np.testing.assert_array_equal(x, original)

    @unittest.skipUnless(importlib.util.find_spec('toto2'), 'optional Toto 2 package not installed')
    def test_real_toto_forecast_with_168_hour_multivariate_context(self):
        from einops import EinopsError
        from toto2 import Toto2Model
        from toto2.configuration import Toto2ModelConfig

        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = Toto2ModelConfig(patch_size=32, d_model=32, num_heads=2,
                                 num_layers=2, layer_group_size=2,
                                 num_variate_layers_per_group=1, variate_layer_first=False,
                                 residual_attn_ratio=1.0)
        model = Toto2Model(config).eval()  # Small random model; no checkpoint download.
        x = np.random.default_rng(42).normal(size=(1, 168, 7)).astype(np.float32)
        target = torch.from_numpy(x.transpose(0, 2, 1).copy())
        with torch.inference_mode(), self.assertRaises(EinopsError):
            model.forecast({'target': target, 'target_mask': torch.ones_like(target, dtype=torch.bool),
                            'series_ids': torch.zeros((1, 7), dtype=torch.long)}, horizon=168)
        with patch.object(Toto2Model, 'from_pretrained', return_value=model):
            predict = benchmark.foundation_adapter('toto2', 'cpu', 1)
        for horizon in (24, 168):
            result = predict(x, horizon)
            self.assertEqual(result.shape, (1, horizon))
            self.assertTrue(np.isfinite(result).all())
