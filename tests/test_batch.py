from pathlib import Path
import unittest

from bpwo.batch import load_manifest


FIXTURE = Path(__file__).parent / "fixtures" / "screening_manifest.csv"


class BatchTests(unittest.TestCase):
    def test_manifest_preserves_predeclared_roles_and_optima(self) -> None:
        specs = load_manifest(FIXTURE)
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0].filename, "scp41.txt")
        self.assertEqual(specs[0].known_optimum, 429)
        self.assertEqual(specs[0].role, "calibration")
        self.assertEqual(specs[1].role, "test")


if __name__ == "__main__":
    unittest.main()
