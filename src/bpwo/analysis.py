"""Resumen descriptivo reproducible de resultados experimentales BPWO."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .experiment import write_csv


CONFIGURATION_FIELDS = ("algorithm", "movement_mode", "scheme")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--summary-output", required=True, type=Path)
    parser.add_argument("--ranking-output", required=True, type=Path)
    parser.add_argument("--history-input", type=Path)
    parser.add_argument("--mechanism-output", type=Path)
    return parser.parse_args()


def read_results(path: Path) -> list[dict[str, str]]:
    """Lee el CSV agregado y comprueba las columnas necesarias."""

    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        required = {
            "instance",
            "algorithm",
            "movement_mode",
            "scheme",
            "cost",
            "rpd",
            "feasible",
            "runtime_seconds",
            "best_found_at_evaluation",
            "evaluations",
            "added_columns",
            "removed_columns",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required.difference(reader.fieldnames or [])
            raise ValueError(f"Faltan columnas requeridas: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError("El archivo de resultados está vacío.")
    return rows


def read_history(path: Path) -> list[dict[str, str]]:
    """Lee un CSV de trayectoria sin imponer columnas del resumen agregado."""

    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)
    if not rows:
        raise ValueError("El archivo de trayectoria está vacío.")
    return rows


def _number(value: str) -> float:
    return float(value)


def _boolean(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise ValueError(f"Valor booleano no reconocido: {value!r}")


def _quartiles(values: list[float]) -> tuple[float, float]:
    if len(values) == 1:
        return values[0], values[0]
    q1, _, q3 = statistics.quantiles(values, n=4, method="inclusive")
    return q1, q3


def _sample_std(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_results(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    """Agrupa por instancia y configuración sin mezclar semillas."""

    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["instance"], *(row[field] for field in CONFIGURATION_FIELDS))
        groups[key].append(row)

    summaries: list[dict[str, object]] = []
    for key, group in sorted(groups.items()):
        costs = [_number(row["cost"]) for row in group]
        rpds = [_number(row["rpd"]) for row in group]
        runtimes = [_number(row["runtime_seconds"]) for row in group]
        best_evaluations = [
            _number(row["best_found_at_evaluation"]) for row in group
        ]
        evaluations = [_number(row["evaluations"]) for row in group]
        added = [_number(row["added_columns"]) for row in group]
        removed = [_number(row["removed_columns"]) for row in group]
        q1, q3 = _quartiles(costs)
        instance, algorithm, movement_mode, scheme = key
        metadata = group[0]
        summaries.append(
            {
                "instance": instance,
                "role": metadata.get("role", "unspecified"),
                "scale": metadata.get("scale", "unspecified"),
                "reference_type": metadata.get("reference_type", "unspecified"),
                "algorithm": algorithm,
                "movement_mode": movement_mode,
                "scheme": scheme,
                "runs": len(group),
                "cost_best": min(costs),
                "cost_mean": statistics.fmean(costs),
                "cost_std": _sample_std(costs),
                "cost_median": statistics.median(costs),
                "cost_q1": q1,
                "cost_q3": q3,
                "cost_iqr": q3 - q1,
                "cost_worst": max(costs),
                "rpd_mean": statistics.fmean(rpds),
                "rpd_median": statistics.median(rpds),
                "feasible_rate": statistics.fmean(
                    float(_boolean(row["feasible"])) for row in group
                ),
                "runtime_mean_seconds": statistics.fmean(runtimes),
                "best_evaluation_median": statistics.median(best_evaluations),
                "added_per_evaluation_mean": statistics.fmean(
                    value / count for value, count in zip(added, evaluations)
                ),
                "removed_per_evaluation_mean": statistics.fmean(
                    value / count for value, count in zip(removed, evaluations)
                ),
            }
        )
    return summaries


def summarize_history(rows: Iterable[dict[str, str]]) -> list[dict[str, object]]:
    """Resume primero cada trayectoria y después agrega entre semillas."""

    required = {
        "instance",
        "algorithm",
        "movement_mode",
        "scheme",
        "seed",
        "binary_diversity",
        "initially_feasible_rate",
        "added_columns",
        "removed_columns",
    }
    grouped_runs: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if not required.issubset(row):
            missing = required.difference(row)
            raise ValueError(f"Faltan columnas de trayectoria: {sorted(missing)}")
        key = (
            row["instance"],
            row["algorithm"],
            row["movement_mode"],
            row["scheme"],
            row["seed"],
        )
        grouped_runs[key].append(row)

    per_configuration: dict[tuple[str, ...], list[dict[str, float]]] = defaultdict(list)
    metadata: dict[tuple[str, ...], dict[str, str]] = {}
    for run_key, run in grouped_runs.items():
        instance, algorithm, movement_mode, scheme, _ = run_key
        configuration = (instance, algorithm, movement_mode, scheme)
        diversities = [_number(row["binary_diversity"]) for row in run]
        exploitation_values = [
            _boolean(row["exploitation"])
            for row in run
            if row.get("exploitation", "N/A") != "N/A"
        ]
        per_configuration[configuration].append(
            {
                "final_diversity": diversities[-1],
                "minimum_diversity": min(diversities),
                "median_diversity": statistics.median(diversities),
                "exploitation_rate": (
                    statistics.fmean(float(value) for value in exploitation_values)
                    if exploitation_values
                    else math.nan
                ),
                "initially_feasible_rate": statistics.fmean(
                    _number(row["initially_feasible_rate"]) for row in run
                ),
                "added_per_iteration": statistics.fmean(
                    _number(row["added_columns"]) for row in run
                ),
                "removed_per_iteration": statistics.fmean(
                    _number(row["removed_columns"]) for row in run
                ),
                "restart_rate": statistics.fmean(
                    _number(row.get("restart_rate", "0"))
                    for row in run
                    if row.get("restart_rate", "N/A") != "N/A"
                )
                if any(row.get("restart_rate", "N/A") != "N/A" for row in run)
                else math.nan,
            }
        )
        metadata[configuration] = run[0]

    summaries: list[dict[str, object]] = []
    for configuration, runs in sorted(per_configuration.items()):
        instance, algorithm, movement_mode, scheme = configuration
        source = metadata[configuration]
        exploitation = [
            row["exploitation_rate"]
            for row in runs
            if not math.isnan(row["exploitation_rate"])
        ]
        restart_rates = [
            row["restart_rate"]
            for row in runs
            if not math.isnan(row["restart_rate"])
        ]
        summaries.append(
            {
                "instance": instance,
                "role": source.get("role", "unspecified"),
                "scale": source.get("scale", "unspecified"),
                "algorithm": algorithm,
                "movement_mode": movement_mode,
                "scheme": scheme,
                "runs": len(runs),
                "final_diversity_median": statistics.median(
                    row["final_diversity"] for row in runs
                ),
                "minimum_diversity_median": statistics.median(
                    row["minimum_diversity"] for row in runs
                ),
                "trajectory_diversity_median": statistics.median(
                    row["median_diversity"] for row in runs
                ),
                "exploitation_rate_median": (
                    statistics.median(exploitation) if exploitation else "N/A"
                ),
                "restart_rate_median": (
                    statistics.median(restart_rates) if restart_rates else "N/A"
                ),
                "initially_feasible_rate_mean": statistics.fmean(
                    row["initially_feasible_rate"] for row in runs
                ),
                "added_per_iteration_mean": statistics.fmean(
                    row["added_per_iteration"] for row in runs
                ),
                "removed_per_iteration_mean": statistics.fmean(
                    row["removed_per_iteration"] for row in runs
                ),
            }
        )
    return summaries


def rank_summaries(
    summaries: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Asigna rangos por mediana de RPD con empates por rango promedio."""

    by_instance: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in summaries:
        by_instance[str(row["instance"])].append(row)

    ranked: list[dict[str, object]] = []
    for instance, group in sorted(by_instance.items()):
        ordered = sorted(
            group,
            key=lambda row: (
                float(row["rpd_median"]),
                float(row["cost_iqr"]),
                float(row["runtime_mean_seconds"]),
            ),
        )
        index = 0
        while index < len(ordered):
            tied_until = index + 1
            value = float(ordered[index]["rpd_median"])
            while (
                tied_until < len(ordered)
                and math.isclose(
                    float(ordered[tied_until]["rpd_median"]),
                    value,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            ):
                tied_until += 1
            average_rank = ((index + 1) + tied_until) / 2.0
            for row in ordered[index:tied_until]:
                ranked.append(
                    {
                        "instance": instance,
                        "rank": average_rank,
                        "algorithm": row["algorithm"],
                        "movement_mode": row["movement_mode"],
                        "scheme": row["scheme"],
                        "runs": row["runs"],
                        "rpd_median": row["rpd_median"],
                        "cost_median": row["cost_median"],
                        "cost_iqr": row["cost_iqr"],
                        "runtime_mean_seconds": row["runtime_mean_seconds"],
                    }
                )
            index = tied_until

    return sorted(ranked, key=lambda row: (str(row["instance"]), float(row["rank"])))


def main() -> int:
    args = _parse_args()
    summaries = summarize_results(read_results(args.input))
    rankings = rank_summaries(summaries)
    write_csv(args.summary_output, summaries)
    write_csv(args.ranking_output, rankings)
    mechanism_count = 0
    if (args.history_input is None) != (args.mechanism_output is None):
        raise ValueError(
            "--history-input y --mechanism-output deben usarse juntos."
        )
    if args.history_input is not None and args.mechanism_output is not None:
        history = read_history(args.history_input)
        mechanisms = summarize_history(history)
        write_csv(args.mechanism_output, mechanisms)
        mechanism_count = len(mechanisms)
    print(
        f"Se escribieron {len(summaries)} resúmenes, {len(rankings)} rangos "
        f"y {mechanism_count} resúmenes de mecanismo."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
