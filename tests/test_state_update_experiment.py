from pathlib import Path
import csv
import tempfile
import unittest

import numpy as np

from bpwo.algorithm import run_bpwo
from bpwo.scp import SCPInstance
from bpwo.state_update_experiment import (
    VARIANTS,
    StateUpdateTask,
    build_config,
    run_state_update_bpwo,
    run_state_update_task,
)


FIXTURE = Path(__file__).parent / "fixtures" / "scp_toy.txt"


class StateUpdateExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = SCPInstance.from_orlib(FIXTURE, known_optimum=4)

    def test_reference_is_identical_to_consolidated_anchor(self) -> None:
        config = build_config(population=5, evaluations=75)
        expected = run_bpwo(self.instance, config, seed=23)
        observed = run_state_update_bpwo(
            self.instance,
            config,
            VARIANTS[0],
            seed=23,
        ).result
        self.assertEqual(expected.best_cost, observed.best_cost)
        np.testing.assert_array_equal(expected.best_solution, observed.best_solution)
        self.assertEqual(expected.best_found_at_evaluation, observed.best_found_at_evaluation)
        self.assertEqual(expected.history, observed.history)

    def test_always_variants_keep_global_best_and_are_reproducible(self) -> None:
        config = build_config(population=5, evaluations=75)
        for variant in VARIANTS[1:]:
            first = run_state_update_bpwo(
                self.instance, config, variant, seed=7
            )
            second = run_state_update_bpwo(
                self.instance, config, variant, seed=7
            )
            self.assertEqual(first.result.best_cost, second.result.best_cost)
            np.testing.assert_array_equal(
                first.result.best_solution, second.result.best_solution
            )
            self.assertEqual(first.result.history, second.result.history)
            self.assertEqual(
                first.accepted_proposals, second.accepted_proposals
            )
            self.assertEqual(
                first.changed_binary_proposals,
                second.changed_binary_proposals,
            )
            self.assertEqual(first.result.evaluations, 75)
            self.assertTrue(self.instance.is_feasible(first.result.best_solution))
            self.assertEqual(
                first.result.best_cost,
                self.instance.cost(first.result.best_solution),
            )
            history_costs = [record.best_cost for record in first.result.history]
            self.assertTrue(
                all(left >= right for left, right in zip(history_costs, history_costs[1:]))
            )
            self.assertEqual(first.accepted_proposals, first.iterative_proposals)

    def test_checkpoint_is_isolated_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            task = StateUpdateTask(
                instance_path=FIXTURE,
                known_optimum=4,
                role="development",
                scale="toy",
                reference_type="optimum",
                seed=3,
                population=5,
                evaluations=50,
                checkpoint_dir=checkpoint_dir,
                variants=tuple(variant.algorithm for variant in VARIANTS),
            )
            stem, results, history = run_state_update_task(task)
            self.assertEqual(results, len(VARIANTS))
            self.assertGreater(history, 0)

            results_path = checkpoint_dir / f"{stem}.results.csv"
            with results_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["algorithm"] for row in rows],
                [variant.algorithm for variant in VARIANTS],
            )
            self.assertTrue(all(row["feasible"] == "True" for row in rows))
            self.assertEqual(run_state_update_task(task), (stem, 0, 0))


if __name__ == "__main__":
    unittest.main()
