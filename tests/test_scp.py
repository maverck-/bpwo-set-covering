from pathlib import Path
import unittest

import numpy as np

from bpwo.greedy import solve_greedy
from bpwo.repair import repair_solution
from bpwo.scp import SCPInstance


FIXTURE = Path(__file__).parent / "fixtures" / "scp_toy.txt"


def _reference_repair(instance: SCPInstance, solution: np.ndarray) -> np.ndarray:
    repaired = solution.copy()
    counts = instance.coverage.astype(np.int64) @ repaired.astype(np.int64)
    while np.any(counts == 0):
        uncovered = counts == 0
        new_coverage = instance.coverage[uncovered].sum(axis=0).astype(float)
        candidates = np.flatnonzero((repaired == 0) & (new_coverage > 0))
        ratios = instance.costs[candidates] / new_coverage[candidates]
        order = np.lexsort((candidates, instance.costs[candidates], ratios))
        selected = int(candidates[order[0]])
        repaired[selected] = 1
        counts += instance.coverage[:, selected]

    order = sorted(
        np.flatnonzero(repaired == 1).tolist(),
        key=lambda column: (instance.costs[column], column),
        reverse=True,
    )
    for column in order:
        covered_rows = instance.coverage[:, column] == 1
        if np.all(counts[covered_rows] >= 2):
            repaired[column] = 0
            counts -= instance.coverage[:, column]
    return repaired


class SCPTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = SCPInstance.from_orlib(FIXTURE, known_optimum=4)

    def test_parser_and_known_solution(self) -> None:
        self.assertEqual(self.instance.n_rows, 3)
        self.assertEqual(self.instance.n_columns, 4)
        solution = np.array([0, 1, 1, 0], dtype=np.int8)
        self.assertTrue(self.instance.is_feasible(solution))
        self.assertEqual(self.instance.cost(solution), 4.0)
        self.assertEqual(self.instance.rpd(4.0), 0.0)

    def test_repair_zero_solution_is_feasible(self) -> None:
        repaired, stats = repair_solution(
            self.instance,
            np.zeros(self.instance.n_columns, dtype=np.int8),
        )
        self.assertTrue(self.instance.is_feasible(repaired))
        self.assertFalse(stats.initial_feasible)
        self.assertGreater(stats.added_columns, 0)
        self.assertEqual(self.instance.cost(repaired), 4.0)

    def test_redundancy_removal_preserves_feasibility(self) -> None:
        repaired, stats = repair_solution(
            self.instance,
            np.ones(self.instance.n_columns, dtype=np.int8),
        )
        self.assertTrue(self.instance.is_feasible(repaired))
        self.assertTrue(stats.initial_feasible)
        self.assertGreater(stats.removed_columns, 0)

    def test_incremental_repair_matches_reference_rule(self) -> None:
        for encoded in range(2**self.instance.n_columns):
            solution = np.asarray(
                [
                    (encoded >> column) & 1
                    for column in range(self.instance.n_columns)
                ],
                dtype=np.int8,
            )
            repaired, _ = repair_solution(self.instance, solution)
            np.testing.assert_array_equal(
                repaired,
                _reference_repair(self.instance, solution),
            )

    def test_greedy_reaches_fixture_optimum(self) -> None:
        result = solve_greedy(self.instance)
        self.assertTrue(self.instance.is_feasible(result.solution))
        self.assertEqual(result.cost, 4.0)


if __name__ == "__main__":
    unittest.main()
