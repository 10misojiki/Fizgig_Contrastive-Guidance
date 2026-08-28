"""CPU tests for MiniMax H3 guidance-preserving image loss.

Run after installing Fizgig's environment:
    python -m unittest tests.test_h3_guidance
"""

import os
import sys
import unittest
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from fizgig.minimax.trainer import (
    compute_loss,
    contrastive_guidance_target,
    guidance_normalized_prediction,
    guidance_scale_for_sigma,
)


class _TinyH3(torch.nn.Module):
    """Small deterministic stand-in with the same forward surface used by compute_loss."""

    pack_audio_rows = False

    def __init__(self, *, pack_audio_rows=False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.25))
        self.calls = []
        self.audio_ptrs = []
        self.patch_size = (1, 1, 1)
        self.pack_audio_rows = pack_audio_rows
        self.config = SimpleNamespace(audio_latents_dim=32)

    def forward(self, latent, _t, text, _audio_noise=None):
        self.calls.append((torch.is_grad_enabled(), float(text.mean())))
        self.audio_ptrs.append(None if _audio_noise is None else _audio_noise.data_ptr())
        return torch.ones_like(latent) * (self.weight + text.mean())


class H3GuidanceMathTests(unittest.TestCase):
    def test_sigma_schedule_fades_to_plain_flow_at_clean_end(self):
        sigma = torch.tensor([0.0, 0.5, 1.0])
        torch.testing.assert_close(
            guidance_scale_for_sigma(3.0, sigma, "sigma"),
            torch.tensor([1.0, 2.0, 3.0]),
        )
        torch.testing.assert_close(
            guidance_scale_for_sigma(3.0, sigma, "constant"),
            torch.full_like(sigma, 3.0),
        )

    def test_contrastive_and_normalized_forms_have_same_optimum(self):
        target = torch.tensor([[[[[1.0]]]]])
        empty = torch.tensor([[[[[0.25]]]]])
        guided = torch.tensor([[[[[1.25]]]]])
        scale = torch.tensor([2.25])

        direct_target = contrastive_guidance_target(target, empty, scale)
        normalized = guidance_normalized_prediction(guided, empty, scale)
        direct_loss = torch.nn.functional.mse_loss(guided, direct_target)
        normalized_loss = torch.nn.functional.mse_loss(normalized, target)

        torch.testing.assert_close(direct_loss, normalized_loss * scale.square().mean())
        optimum = direct_target.clone()
        torch.testing.assert_close(
            guidance_normalized_prediction(optimum, empty, scale), target)

    def test_compute_loss_uses_one_detached_empty_forward(self):
        model = _TinyH3()
        latent = torch.ones(1, 1, 1, 1, 1)
        noise = torch.zeros_like(latent)
        sigma = torch.tensor([0.5])
        cond = torch.ones(1, 1, 1)
        empty = torch.zeros_like(cond)

        loss, used_sigma = compute_loss(
            model,
            latent,
            cond,
            sigma=sigma,
            noise=noise,
            guidance_uncond_text=empty,
            guidance_scale=3.5,
            guidance_loss_form="contrastive",
            guidance_schedule="sigma",
        )

        # empty prediction=.25, effective scale=2.25, guided target=1.9375,
        # prompt prediction=1.25
        torch.testing.assert_close(loss, torch.tensor((1.25 - 1.9375) ** 2))
        self.assertEqual(used_sigma, 0.5)
        self.assertEqual(model.calls, [(False, 0.0), (True, 1.0)])

        loss.backward()
        # The target's empty branch is detached; only the prompt prediction contributes grad.
        torch.testing.assert_close(model.weight.grad, torch.tensor(2.0 * (1.25 - 1.9375)))

    def test_empty_and_prompt_forwards_share_silent_audio_noise(self):
        model = _TinyH3(pack_audio_rows=True)
        latent = torch.ones(1, 1, 1, 1, 1)
        compute_loss(
            model,
            latent,
            torch.ones(1, 1, 1),
            sigma=torch.tensor([0.5]),
            noise=torch.zeros_like(latent),
            guidance_uncond_text=torch.zeros(1, 1, 1),
            guidance_scale=3.5,
        )
        self.assertEqual(len(model.audio_ptrs), 2)
        self.assertIsNotNone(model.audio_ptrs[0])
        self.assertEqual(model.audio_ptrs[0], model.audio_ptrs[1])

    def test_disabled_guidance_keeps_the_single_forward_path(self):
        model = _TinyH3()
        latent = torch.ones(1, 1, 1, 1, 1)
        loss, _ = compute_loss(
            model,
            latent,
            torch.ones(1, 1, 1),
            sigma=torch.tensor([0.5]),
            noise=torch.zeros_like(latent),
        )
        self.assertEqual(model.calls, [(True, 1.0)])
        torch.testing.assert_close(loss, torch.tensor(0.25**2))

    def test_invalid_scale_and_schedule_stop_before_training(self):
        with self.assertRaises(ValueError):
            guidance_scale_for_sigma(1.0, torch.tensor([0.5]), "sigma")
        with self.assertRaises(ValueError):
            guidance_scale_for_sigma(3.5, torch.tensor([0.5]), "mystery")


if __name__ == "__main__":
    unittest.main()
