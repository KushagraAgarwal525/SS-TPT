"""CPU checks for instance-adaptive alpha. No dataset and no GPU."""

import math
import unittest

import torch

from utils.instance_alpha import (
    ALPHA_MODES,
    combine_ss,
    compute_instance_alpha,
    compute_instance_alpha_details,
)


def _alpha(suit, stab, mode, fixed_alpha=0.4, **kwargs):
    return compute_instance_alpha(
        suit, stab, mode=mode, fixed_alpha=fixed_alpha, **kwargs
    )


class TestInstanceAlpha(unittest.TestCase):
    def test_fixed_returns_ss_ratio_for_any_input(self):
        suit = torch.tensor([1.0, 0.0, 0.2, 0.8])
        stab = torch.tensor([0.0, 1.0, 0.3, 0.1])
        alpha = _alpha(suit, stab, "fixed", fixed_alpha=0.4)
        self.assertEqual(alpha.shape, torch.Size([]))
        self.assertTrue(torch.allclose(alpha, torch.tensor(0.4)))
        mixed = combine_ss(suit, stab, alpha)
        manual = 0.4 * suit + (1.0 - 0.4) * stab
        self.assertTrue(torch.allclose(mixed, manual))

    def test_margin_suitability_dominates(self):
        # One clear suitability winner; stability's top two are nearly tied.
        suit = torch.tensor([1.0, 0.0, 0.0, 0.0])
        stab = torch.tensor([1.0, 0.9, 0.9, 0.9])
        alpha = _alpha(suit, stab, "margin")
        expected = 1.0 / (1.0 + 0.1 + 1e-8)
        self.assertTrue(torch.allclose(alpha, torch.tensor(expected), atol=1e-6))
        self.assertGreater(alpha.item(), 0.9)

    def test_margin_swap_stability_dominates(self):
        suit = torch.tensor([1.0, 0.9, 0.9, 0.9])
        stab = torch.tensor([1.0, 0.0, 0.0, 0.0])
        alpha = _alpha(suit, stab, "margin")
        expected = 0.1 / (0.1 + 1.0 + 1e-8)
        self.assertTrue(torch.allclose(alpha, torch.tensor(expected), atol=1e-6))
        self.assertLess(alpha.item(), 0.1)

    def test_both_raw_flat_falls_back_to_fixed_alpha(self):
        raw = torch.full((6,), 3.0)
        # Normalized values would look decisive if the raw flat check were skipped.
        peaked = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        alpha = _alpha(
            peaked, peaked, "margin", suit_raw=raw, stab_raw=raw
        )
        self.assertTrue(torch.allclose(alpha, torch.tensor(0.4)))
        self.assertEqual(float((raw.max() - raw.min()).item()), 0.0)

    def test_normalized_flat_without_raw_falls_back(self):
        flat = torch.zeros(5)
        alpha = _alpha(flat, flat, "margin")
        self.assertTrue(torch.allclose(alpha, torch.tensor(0.4)))

    def test_identical_nonflat_is_half(self):
        shared = torch.tensor([1.0, 0.3, 0.0, 0.7])
        for mode in ("margin", "entropy", "variance", "topk", "separation"):
            alpha = _alpha(shared, shared.clone(), mode)
            self.assertTrue(
                torch.allclose(alpha, torch.tensor(0.5), atol=1e-5),
                msg=f"{mode} alpha={alpha.item()}",
            )

    def test_detached_and_in_unit_interval(self):
        suit = torch.tensor([1.0, 0.0, 0.2, 0.5], requires_grad=True)
        stab = torch.tensor([0.2, 1.0, 0.0, 0.4], requires_grad=True)
        raw_s = torch.tensor([5.0, 1.0, 2.0, 3.0], requires_grad=True)
        raw_t = torch.tensor([1.0, 4.0, 0.5, 2.0], requires_grad=True)
        for mode in ALPHA_MODES:
            alpha = _alpha(
                suit, stab, mode, suit_raw=raw_s, stab_raw=raw_t, temp=0.25, topk=3
            )
            self.assertEqual(alpha.shape, torch.Size([]))
            self.assertEqual(alpha.device, suit.device)
            self.assertFalse(alpha.requires_grad)
            self.assertGreaterEqual(alpha.item(), 0.0)
            self.assertLessEqual(alpha.item(), 1.0)
            alpha, r_suit, r_stab = compute_instance_alpha_details(
                suit,
                stab,
                mode=mode,
                fixed_alpha=0.4,
                suit_raw=raw_s,
                stab_raw=raw_t,
            )
            self.assertFalse(r_suit.requires_grad)
            self.assertFalse(r_stab.requires_grad)
            self.assertGreaterEqual(r_suit.item(), 0.0)
            self.assertGreaterEqual(r_stab.item(), 0.0)

    def test_one_raw_flat_gives_the_other_score(self):
        suit_raw = torch.ones(4)
        stab_raw = torch.tensor([1.0, 2.0, 3.0, 4.0])
        suit_n = torch.tensor([1.0, 0.0, 0.0, 0.0])
        stab_n = (stab_raw - stab_raw.min()) / (stab_raw.max() - stab_raw.min() + 1e-8)
        alpha = _alpha(
            suit_n, stab_n, "margin", suit_raw=suit_raw, stab_raw=stab_raw
        )
        self.assertLess(alpha.item(), 1e-6)

    def test_entropy_variance_topk_separation_formulas(self):
        suit = torch.tensor([1.0, 0.0, 0.0, 0.0])
        stab = torch.tensor([1.0, 0.8, 0.2, 0.0])
        temp = 0.25
        topk = 2
        eps = 1e-8

        def entropy_r(score):
            probs = torch.softmax(score / temp, dim=0)
            h = -(probs * probs.log()).sum()
            return 1.0 - h / math.log(score.numel())

        def topk_r(score):
            probs = torch.softmax(score / temp, dim=0)
            idx = torch.topk(score, k=topk).indices
            return probs.index_select(0, idx).sum()

        def separation_r(score):
            ordered = torch.sort(score, descending=True).values
            half = score.numel() // 2
            return ordered[:half].mean() - ordered[-half:].mean()

        def variance_r(score):
            return torch.var(score, unbiased=False)

        expected = {
            "entropy": entropy_r,
            "topk": topk_r,
            "separation": separation_r,
            "variance": variance_r,
        }
        for mode, rel in expected.items():
            r_s = rel(suit)
            r_t = rel(stab)
            want = r_s / (r_s + r_t + eps)
            alpha, got_s, got_t = compute_instance_alpha_details(
                suit, stab, mode=mode, fixed_alpha=0.4, temp=temp, topk=topk, eps=eps
            )
            self.assertTrue(torch.allclose(got_s, r_s.reshape(()), atol=1e-6), msg=mode)
            self.assertTrue(torch.allclose(got_t, r_t.reshape(()), atol=1e-6), msg=mode)
            self.assertTrue(torch.allclose(alpha, want.reshape(()), atol=1e-6), msg=mode)

    def test_agreement_follows_fixed_when_rankings_match(self):
        shared = torch.tensor([1.0, 0.2, 0.0, 0.5])
        alpha = _alpha(shared, shared.clone(), "agreement", fixed_alpha=0.4)
        self.assertTrue(torch.allclose(alpha, torch.tensor(0.4), atol=1e-5))

    def test_agreement_follows_margin_when_rankings_disagree(self):
        suit = torch.tensor([1.0, 0.0, 0.0, 0.0])
        stab = torch.tensor([0.0, 1.0, 0.99, 0.99])
        alpha = _alpha(suit, stab, "agreement", fixed_alpha=0.4)
        margin = _alpha(suit, stab, "margin", fixed_alpha=0.4)
        self.assertGreater(margin.item(), 0.95)
        self.assertTrue(torch.allclose(alpha, margin, atol=1e-5))

    def test_details_match_public_alpha(self):
        suit = torch.tensor([1.0, 0.4, 0.0])
        stab = torch.tensor([0.2, 1.0, 0.5])
        for mode in ALPHA_MODES:
            alpha = _alpha(suit, stab, mode)
            alpha2, _, _ = compute_instance_alpha_details(
                suit, stab, mode=mode, fixed_alpha=0.4
            )
            self.assertTrue(torch.equal(alpha, alpha2))

    def test_fixed_reliabilities_are_zero(self):
        suit = torch.tensor([1.0, 0.0, 0.0])
        stab = torch.tensor([0.0, 1.0, 0.2])
        _, r_suit, r_stab = compute_instance_alpha_details(
            suit, stab, mode="fixed", fixed_alpha=0.4
        )
        self.assertEqual(r_suit.item(), 0.0)
        self.assertEqual(r_stab.item(), 0.0)

    def test_unknown_mode_raises(self):
        score = torch.tensor([1.0, 0.0])
        with self.assertRaises(ValueError):
            _alpha(score, score, "spatial")

    def test_combine_ss_has_no_spatial_argument(self):
        import inspect

        params = list(inspect.signature(combine_ss).parameters)
        self.assertEqual(params, ["suit_n", "stab_n", "alpha"])

    def test_clamp_fixed_alpha(self):
        score = torch.zeros(3)
        high = _alpha(score, score, "fixed", fixed_alpha=1.5)
        low = _alpha(score, score, "fixed", fixed_alpha=-0.2)
        self.assertTrue(torch.allclose(high, torch.tensor(1.0)))
        self.assertTrue(torch.allclose(low, torch.tensor(0.0)))


if __name__ == "__main__":
    unittest.main()
