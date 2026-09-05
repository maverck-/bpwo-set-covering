from pathlib import Path
import unittest

import numpy as np

from bpwo.algorithm import (
    BPWOConfig,
    _exploit_alpha,
    _rally_cohesion,
    run_bpwo,
)
from bpwo.binarization import BinarizationScheme
from bpwo.scp import SCPInstance


FIXTURE = Path(__file__).parent / "fixtures" / "scp_toy.txt"


class BPWOTests(unittest.TestCase):
    def setUp(self) -> None:
        self.instance = SCPInstance.from_orlib(FIXTURE, known_optimum=4)

    def test_budget_feasibility_and_consistency(self) -> None:
        config = BPWOConfig(
            population_size=6,
            max_evaluations=60,
            scheme=BinarizationScheme("V3", "ELIT"),
        )
        result = run_bpwo(self.instance, config, seed=17)
        self.assertEqual(result.evaluations, 60)
        self.assertLessEqual(result.best_found_at_evaluation, result.evaluations)
        self.assertEqual(result.proposals, result.evaluations)
        self.assertGreaterEqual(result.initially_feasible_proposals, 0)
        self.assertLessEqual(result.initially_feasible_proposals, result.proposals)
        self.assertGreaterEqual(result.added_columns, 0)
        self.assertGreaterEqual(result.removed_columns, 0)
        self.assertTrue(self.instance.is_feasible(result.best_solution))
        self.assertEqual(result.best_cost, self.instance.cost(result.best_solution))
        self.assertEqual(result.history[-1].evaluations, 60)

    def test_same_seed_reproduces_full_result(self) -> None:
        config = BPWOConfig(
            population_size=5,
            max_evaluations=45,
            scheme=BinarizationScheme("S2", "STD"),
        )
        first = run_bpwo(self.instance, config, seed=99)
        second = run_bpwo(self.instance, config, seed=99)
        self.assertEqual(first.best_cost, second.best_cost)
        np.testing.assert_array_equal(first.best_solution, second.best_solution)
        self.assertEqual(first.history, second.history)

    def test_history_contains_mechanism_metrics(self) -> None:
        config = BPWOConfig(population_size=4, max_evaluations=20)
        result = run_bpwo(self.instance, config, seed=5)
        self.assertGreater(len(result.history), 0)
        for record in result.history:
            self.assertGreaterEqual(record.rally_cohesion, 0.0)
            self.assertLessEqual(record.rally_cohesion, 1.0)
            self.assertGreaterEqual(record.initially_feasible_rate, 0.0)
            self.assertLessEqual(record.initially_feasible_rate, 1.0)

    def test_iid_ablation_is_reproducible_and_does_not_report_exploitation(self) -> None:
        config = BPWOConfig(
            population_size=4,
            max_evaluations=20,
            movement_mode="IID",
        )
        first = run_bpwo(self.instance, config, seed=8)
        second = run_bpwo(self.instance, config, seed=8)
        self.assertEqual(first.best_cost, second.best_cost)
        np.testing.assert_array_equal(first.best_solution, second.best_solution)
        self.assertEqual(first.history, second.history)
        self.assertTrue(all(not record.exploitation for record in first.history))

    def test_diversity_restart_is_budgeted_and_reported(self) -> None:
        config = BPWOConfig(
            population_size=5,
            max_evaluations=45,
            movement_mode="PWO-DR",
            diversity_restart_threshold=0.5,
            restart_fraction=0.25,
        )
        first = run_bpwo(self.instance, config, seed=23)
        second = run_bpwo(self.instance, config, seed=23)
        self.assertEqual(first.evaluations, 45)
        self.assertEqual(first.history, second.history)
        self.assertTrue(any(record.restart_rate > 0 for record in first.history))
        self.assertTrue(
            all(0.0 <= record.restart_rate < 1.0 for record in first.history)
        )

    def test_delta_probabilistic_variant_is_reproducible(self) -> None:
        config = BPWOConfig(
            population_size=5,
            max_evaluations=100,
            scheme=BinarizationScheme("V3", "COMP"),
            rally_mode="PROBABILISTIC",
            binarization_input="DELTA",
        )
        first = run_bpwo(self.instance, config, seed=31)
        second = run_bpwo(self.instance, config, seed=31)
        self.assertEqual(first.best_cost, second.best_cost)
        np.testing.assert_array_equal(first.best_solution, second.best_solution)
        self.assertEqual(first.history, second.history)
        self.assertTrue(self.instance.is_feasible(first.best_solution))
        self.assertTrue(
            all(0.0 <= record.attack_threshold <= 1.0 for record in first.history)
        )

    def test_variant_modes_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            BPWOConfig(rally_mode="unknown")
        with self.assertRaises(ValueError):
            BPWOConfig(binarization_input="unknown")
        with self.assertRaises(ValueError):
            BPWOConfig(degenerate_cohesion="unknown")
        with self.assertRaises(ValueError):
            BPWOConfig(latent_amplitude=0.0)
        with self.assertRaises(ValueError):
            BPWOConfig(latent_amplitude=4.0)

    def test_homogeneous_population_collapses_the_rally(self) -> None:
        """Documenta el defecto diagnosticado y su corrección.

        Con costos iguales no hay señal de calidad. El modo heredado devuelve
        cohesión 1, con lo que la explotación deja de perturbar y cada
        propuesta se reduce al alfa; el modo corregido devuelve 0.
        """

        costs = np.full(10, 524.0)
        self.assertEqual(_rally_cohesion(costs, 0, 0.5, "MAX"), 1.0)
        self.assertEqual(_rally_cohesion(costs, 0, 0.5, "MIN"), 0.0)

        alpha = np.array([1.0, -1.0, 1.0, -1.0])
        current = np.array([-1.0, -1.0, 1.0, 1.0])
        collapsed = _exploit_alpha(
            current, alpha, a=1.0, rally_cohesion=1.0, rng=np.random.default_rng(1)
        )
        np.testing.assert_array_equal(collapsed, alpha)
        perturbed = _exploit_alpha(
            current, alpha, a=1.0, rally_cohesion=0.0, rng=np.random.default_rng(1)
        )
        self.assertFalse(np.array_equal(perturbed, alpha))

    def test_latent_amplitude_scales_the_synchronised_state(self) -> None:
        config = BPWOConfig(
            population_size=4,
            max_evaluations=40,
            scheme=BinarizationScheme("S1", "STD"),
            latent_amplitude=4.0,
            lower_bound=-6.0,
            upper_bound=6.0,
        )
        result = run_bpwo(self.instance, config, seed=5)
        self.assertEqual(result.evaluations, 40)
        self.assertTrue(self.instance.is_feasible(result.best_solution))

    def test_alpha_control_is_reproducible_and_reports_no_exploitation(self) -> None:
        config = BPWOConfig(
            population_size=5,
            max_evaluations=50,
            scheme=BinarizationScheme("S1", "STD"),
            movement_mode="ALPHA",
            latent_amplitude=4.0,
            lower_bound=-6.0,
            upper_bound=6.0,
        )
        first = run_bpwo(self.instance, config, seed=9)
        second = run_bpwo(self.instance, config, seed=9)
        self.assertEqual(first.history, second.history)
        self.assertEqual(first.evaluations, 50)
        self.assertTrue(self.instance.is_feasible(first.best_solution))
        self.assertFalse(any(record.exploitation for record in first.history))


if __name__ == "__main__":
    unittest.main()
