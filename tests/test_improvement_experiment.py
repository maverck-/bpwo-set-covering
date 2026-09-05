import tempfile
import unittest
from pathlib import Path

from bpwo.improvement_experiment import ImprovementTask, run_improvement_task


FIXTURE = Path(__file__).parent / "fixtures" / "scp_toy.txt"


class ImprovementExperimentTests(unittest.TestCase):
    def test_checkpoint_contains_four_variants_and_is_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = ImprovementTask(
                instance_path=FIXTURE,
                known_optimum=4,
                role="development",
                scale="small",
                reference_type="optimum",
                seed=0,
                population=4,
                evaluations=20,
                checkpoint_dir=Path(directory),
            )
            _, result_count, history_count = run_improvement_task(task)
            self.assertEqual(result_count, 4)
            self.assertGreater(history_count, 0)
            _, reused_results, reused_history = run_improvement_task(task)
            self.assertEqual((reused_results, reused_history), (0, 0))


if __name__ == "__main__":
    unittest.main()
