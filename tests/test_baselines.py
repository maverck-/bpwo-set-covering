from pathlib import Path
import unittest

import numpy as np

from bpwo.baselines import BaselineConfig, run_bgwo, run_bpso
from bpwo.scp import SCPInstance


FIXTURE = Path(__file__).parent / "fixtures" / "scp_toy.txt"


class BaselineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = SCPInstance.from_orlib(FIXTURE, known_optimum=4)
        self.config = BaselineConfig(population_size=5, max_evaluations=45)

    def _assert_valid_and_reproducible(self, runner) -> None:
        first = runner(self.instance, self.config, seed=21)
        second = runner(self.instance, self.config, seed=21)
        self.assertEqual(first.evaluations, 45)
        self.assertEqual(first.proposals, first.evaluations)
        self.assertTrue(self.instance.is_feasible(first.best_solution))
        self.assertEqual(first.best_cost, self.instance.cost(first.best_solution))
        self.assertEqual(first.best_cost, second.best_cost)
        np.testing.assert_array_equal(first.best_solution, second.best_solution)
        self.assertEqual(first.history, second.history)

    def test_bpso(self) -> None:
        self._assert_valid_and_reproducible(run_bpso)

    def test_bgwo(self) -> None:
        self._assert_valid_and_reproducible(run_bgwo)


if __name__ == "__main__":
    unittest.main()
