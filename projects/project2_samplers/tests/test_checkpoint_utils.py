import unittest

import torch

from projects.project2_samplers.checkpoint_utils import (
    load_model_weights,
    model_state_from_checkpoint,
)


class CheckpointUtilsTests(unittest.TestCase):
    def test_nested_ema_is_unwrapped(self):
        model = torch.nn.Linear(2, 2)
        expected = {
            name: torch.ones_like(value) for name, value in model.state_dict().items()
        }
        checkpoint = {
            "model": model.state_dict(),
            "ema": {"decay": 0.999, "model": expected},
        }
        self.assertIs(model_state_from_checkpoint(checkpoint), expected)
        self.assertEqual(load_model_weights(model, checkpoint), "EMA")
        for value in model.state_dict().values():
            self.assertTrue(torch.equal(value, torch.ones_like(value)))

    def test_flat_ema_is_supported(self):
        model = torch.nn.Linear(2, 2)
        flat = {
            name: torch.zeros_like(value) for name, value in model.state_dict().items()
        }
        checkpoint = {"model": model.state_dict(), "ema": flat}
        self.assertIs(model_state_from_checkpoint(checkpoint), flat)

    def test_multi_ema_defaults_to_largest_decay_and_can_select_one(self):
        low = {"weight": torch.tensor([0.25])}
        high = {"weight": torch.tensor([0.75])}
        checkpoint = {
            "model": {"weight": torch.tensor([0.0])},
            "ema": {"decays": [0.999, 0.9999], "models": [low, high]},
        }
        self.assertIs(model_state_from_checkpoint(checkpoint), high)
        self.assertIs(model_state_from_checkpoint(checkpoint, ema_decay=0.999), low)
        self.assertEqual(load_model_weights(torch.nn.Linear(1, 1, bias=False), {
            "model": {"weight": torch.zeros(1, 1)},
            "ema": {"decays": [0.999, 0.9999], "models": [
                {"weight": torch.full((1, 1), 0.25)},
                {"weight": torch.full((1, 1), 0.75)},
            ]},
        }, ema_decay=0.999), "EMA_0.999")

    def test_multi_ema_rejects_missing_requested_decay(self):
        checkpoint = {
            "model": {},
            "ema": {"decays": [0.999], "models": [{}]},
        }
        with self.assertRaisesRegex(ValueError, "unavailable"):
            model_state_from_checkpoint(checkpoint, ema_decay=0.9999)


if __name__ == "__main__":
    unittest.main()
