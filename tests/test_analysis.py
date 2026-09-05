import csv
import tempfile
import unittest
from pathlib import Path

from bpwo.analysis import (
    rank_summaries,
    read_results,
    summarize_history,
    summarize_results,
)


class AnalysisTests(unittest.TestCase):
    def test_summary_and_average_rank_for_ties(self) -> None:
        rows = []
        for algorithm, costs in (("A", (10, 12, 14)), ("B", (12, 12, 12))):
            for seed, cost in enumerate(costs):
                rows.append(
                    {
                        "instance": "toy",
                        "algorithm": algorithm,
                        "movement_mode": "TEST",
                        "scheme": "TEST",
                        "seed": str(seed),
                        "population": "3",
                        "evaluation_budget": "10",
                        "evaluations": "10",
                        "best_found_at_evaluation": "5",
                        "cost": str(cost),
                        "rpd": str(cost - 10),
                        "feasible": "True",
                        "initially_feasible_rate": "0.5",
                        "added_columns": "10",
                        "removed_columns": "20",
                        "runtime_seconds": "1",
                        "python": "test",
                        "numpy": "test",
                    }
                )

        summaries = summarize_results(rows)
        self.assertEqual(len(summaries), 2)
        self.assertEqual(summaries[0]["cost_median"], 12)
        self.assertEqual(summaries[0]["cost_iqr"], 2)
        self.assertEqual(summaries[0]["added_per_evaluation_mean"], 1)
        ranks = rank_summaries(summaries)
        self.assertEqual([row["rank"] for row in ranks], [1.5, 1.5])

    def test_reader_rejects_missing_columns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            with path.open("w", encoding="utf-8", newline="") as output_file:
                writer = csv.DictWriter(output_file, fieldnames=["instance"])
                writer.writeheader()
                writer.writerow({"instance": "toy"})
            with self.assertRaises(ValueError):
                read_results(path)

    def test_history_is_aggregated_between_seeds(self) -> None:
        rows = []
        for seed, diversities in (("0", (0.4, 0.2)), ("1", (0.3, 0.1))):
            for iteration, diversity in enumerate(diversities, start=1):
                rows.append(
                    {
                        "instance": "toy",
                        "algorithm": "BPWO",
                        "movement_mode": "PWO",
                        "scheme": "S2-ELIT",
                        "seed": seed,
                        "iteration": str(iteration),
                        "binary_diversity": str(diversity),
                        "exploitation": str(iteration == 2),
                        "initially_feasible_rate": "0.5",
                        "added_columns": "2",
                        "removed_columns": "4",
                    }
                )
        summary = summarize_history(rows)
        self.assertEqual(len(summary), 1)
        self.assertAlmostEqual(summary[0]["final_diversity_median"], 0.15)
        self.assertAlmostEqual(summary[0]["exploitation_rate_median"], 0.5)


if __name__ == "__main__":
    unittest.main()
