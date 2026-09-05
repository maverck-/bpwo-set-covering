import tempfile
import unittest
from pathlib import Path

from bpwo.calibration_experiment import CalibrationTask, run_calibration_task


FIXTURE = Path(__file__).parent / "fixtures" / "scp_toy.txt"


class CalibrationExperimentTests(unittest.TestCase):
    def test_checkpoint_contains_two_schemes_and_is_reusable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            task = CalibrationTask(
                instance_path=FIXTURE,
                known_optimum=4,
                role="calibration",
                scale="small",
                reference_type="optimum",
                seed=0,
                population=4,
                evaluations=20,
                checkpoint_dir=Path(directory),
            )
            _, result_count, history_count = run_calibration_task(task)
            self.assertEqual(result_count, 2)
            self.assertGreater(history_count, 0)
            _, reused_results, reused_history = run_calibration_task(task)
            self.assertEqual((reused_results, reused_history), (0, 0))


if __name__ == "__main__":
    unittest.main()
