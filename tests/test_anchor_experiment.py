from pathlib import Path
import csv
import tempfile
import unittest

from bpwo.anchor_experiment import VARIANTS, AnchorTask, build_config, run_anchor_task


FIXTURE = Path(__file__).parent / "fixtures" / "scp_toy.txt"


class AnchorExperimentTests(unittest.TestCase):
    def test_variants_declare_a_reference_and_a_control(self) -> None:
        names = [variant.algorithm for variant in VARIANTS]
        self.assertEqual(
            names,
            [
                "BPWO_BASE",
                "BPWO_RALLY",
                "BPWO_ANCHOR",
                "BPWO_S2_STD",
                "BPWO_V3_COMP",
                "ALPHA_S1",
            ],
        )
        base = VARIANTS[0]
        self.assertEqual(base.degenerate_cohesion, "MAX")
        self.assertEqual(base.latent_amplitude, 1.0)
        self.assertEqual(base.scheme.name, "V3-ELIT")
        control = VARIANTS[-1]
        self.assertEqual(control.movement_mode, "ALPHA")
        self.assertEqual(control.scheme.name, "S1-STD")

    def test_base_variant_rebuilds_the_frozen_configuration(self) -> None:
        config = build_config(VARIANTS[0], population=10, evaluations=6000)
        self.assertEqual(config.scheme.name, "V3-ELIT")
        self.assertEqual(config.movement_mode, "PWO")
        self.assertEqual(config.degenerate_cohesion, "MAX")
        self.assertEqual(config.latent_amplitude, 1.0)
        self.assertEqual((config.lower_bound, config.upper_bound), (-1.0, 1.0))

    def test_checkpoint_contains_every_variant_and_is_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_dir = Path(directory)
            task = AnchorTask(
                instance_path=FIXTURE,
                known_optimum=4,
                role="calibration",
                scale="toy",
                reference_type="optimum",
                seed=3,
                population=5,
                evaluations=50,
                checkpoint_dir=checkpoint_dir,
                variants=tuple(variant.algorithm for variant in VARIANTS),
            )
            stem, results, history = run_anchor_task(task)
            self.assertEqual(results, len(VARIANTS))
            self.assertGreater(history, 0)

            results_path = checkpoint_dir / f"{stem}.results.csv"
            with results_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(
                [row["algorithm"] for row in rows],
                [variant.algorithm for variant in VARIANTS],
            )
            for row in rows:
                self.assertEqual(int(row["evaluations"]), 50)
                self.assertEqual(row["feasible"], "True")
                self.assertEqual(row["role"], "calibration")

            self.assertEqual(run_anchor_task(task), (stem, 0, 0))


if __name__ == "__main__":
    unittest.main()
