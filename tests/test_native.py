from pathlib import Path
import unittest

import numpy as np

from bpwo.native import NativeBPWOConfig, run_native_bpwo
from bpwo.scp import SCPInstance


FIXTURE = Path(__file__).parent / "fixtures" / "scp_toy.txt"


class NativeBPWOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = SCPInstance.from_orlib(FIXTURE, known_optimum=4)

    def test_native_variant_is_feasible_budgeted_and_reproducible(self) -> None:
        config = NativeBPWOConfig(population_size=5, max_evaluations=45)
        first = run_native_bpwo(self.instance, config, seed=12)
        second = run_native_bpwo(self.instance, config, seed=12)
        self.assertEqual(first.evaluations, 45)
        self.assertEqual(first.best_cost, self.instance.cost(first.best_solution))
        self.assertTrue(self.instance.is_feasible(first.best_solution))
        self.assertEqual(first.best_cost, second.best_cost)
        np.testing.assert_array_equal(first.best_solution, second.best_solution)
        self.assertEqual(first.history, second.history)


if __name__ == "__main__":
    unittest.main()
