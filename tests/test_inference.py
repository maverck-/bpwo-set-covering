import unittest

from bpwo.inference import cliffs_delta, holm_adjust


class InferenceTests(unittest.TestCase):
    def test_cliffs_delta_direction_for_minimization(self) -> None:
        self.assertEqual(cliffs_delta([1, 2], [3, 4]), -1.0)
        self.assertEqual(cliffs_delta([3, 4], [1, 2]), 1.0)
        self.assertEqual(cliffs_delta([1, 2], [1, 2]), 0.0)

    def test_holm_is_monotone_in_sorted_order(self) -> None:
        adjusted = holm_adjust([0.03, 0.01, 0.04])
        self.assertEqual(adjusted, [0.06, 0.03, 0.06])


if __name__ == "__main__":
    unittest.main()
