import unittest

import numpy as np

from bpwo.binarization import (
    BinarizationScheme,
    transfer_s1,
    transfer_s2,
    transfer_v3,
)


class BinarizationTests(unittest.TestCase):
    def test_transfer_functions_are_bounded(self) -> None:
        values = np.array([-1e6, -2.0, 0.0, 2.0, 1e6])
        for transfer in (transfer_s1, transfer_s2, transfer_v3):
            probabilities = transfer(values)
            self.assertTrue(np.all(probabilities >= 0.0))
            self.assertTrue(np.all(probabilities <= 1.0))

    def test_s1_is_the_sharp_sigmoid_of_mirjalili(self) -> None:
        """S1 es ``1 / (1 + exp(-2x))``; con el latente anclado en +-4 decide
        el bit casi de forma determinista."""

        self.assertAlmostEqual(float(transfer_s1(np.array([0.0]))[0]), 0.5)
        expected = 1.0 / (1.0 + np.exp(-8.0))
        self.assertAlmostEqual(float(transfer_s1(np.array([4.0]))[0]), expected)
        self.assertAlmostEqual(float(transfer_s1(np.array([-4.0]))[0]), 1.0 - expected)

    def test_v3_is_zero_at_origin(self) -> None:
        self.assertEqual(float(transfer_v3(np.array([0.0]))[0]), 0.0)

    def test_standard_rule_is_reproducible(self) -> None:
        scheme = BinarizationScheme("S2", "STD")
        values = np.linspace(-2.0, 2.0, 20)
        current = np.zeros(20, dtype=np.int8)
        best = np.ones(20, dtype=np.int8)
        first = scheme.apply(
            values,
            best=best,
            current=current,
            rng=np.random.default_rng(123),
        )
        second = scheme.apply(
            values,
            best=best,
            current=current,
            rng=np.random.default_rng(123),
        )
        np.testing.assert_array_equal(first, second)

    def test_elitist_rule_only_copies_best_or_zero(self) -> None:
        scheme = BinarizationScheme("S2", "ELIT")
        best = np.array([1, 0, 1, 0, 1, 0], dtype=np.int8)
        result = scheme.apply(
            np.full(best.size, 100.0),
            best=best,
            current=1 - best,
            rng=np.random.default_rng(4),
        )
        np.testing.assert_array_equal(result, best)

    def test_complement_rule_flips_only_selected_bits(self) -> None:
        scheme = BinarizationScheme("V3", "COMP")
        values = np.array([0.0, 1e9, -1e9])
        current = np.array([0, 0, 1], dtype=np.int8)
        result = scheme.apply(
            values,
            best=np.ones(3, dtype=np.int8),
            current=current,
            rng=np.random.default_rng(4),
        )
        np.testing.assert_array_equal(result, np.array([0, 1, 0], dtype=np.int8))


if __name__ == "__main__":
    unittest.main()
