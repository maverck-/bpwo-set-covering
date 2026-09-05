import tempfile
import unittest
from pathlib import Path

from bpwo.final_experiment import SeedTask, combine_checkpoints, run_seed_task


FIXTURE = Path(__file__).parent / "fixtures" / "scp_toy.txt"


class FinalExperimentTests(unittest.TestCase):
    def test_seed_checkpoint_is_complete_and_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = SeedTask(
                instance_path=FIXTURE,
                known_optimum=4,
                role="test",
                scale="small",
                reference_type="optimum",
                seed=0,
                population=4,
                evaluations=20,
                checkpoint_dir=root / "checkpoints",
            )
            _, result_count, history_count = run_seed_task(task)
            self.assertEqual(result_count, 5)
            self.assertGreater(history_count, 0)
            _, reused_results, reused_history = run_seed_task(task)
            self.assertEqual((reused_results, reused_history), (0, 0))

            combined_results, combined_history = combine_checkpoints(
                task.checkpoint_dir,
                output=root / "results.csv",
                history_output=root / "history.csv",
            )
            self.assertEqual(combined_results, 5)
            self.assertEqual(combined_history, history_count)


if __name__ == "__main__":
    unittest.main()
