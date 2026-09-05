"""Ejecución reproducible de greedy y BPWO con salida CSV."""

from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .algorithm import BPWOConfig, run_bpwo
from .baselines import BaselineConfig, run_bgwo, run_bpso
from .binarization import BinarizationScheme
from .greedy import solve_greedy
from .native import NativeBPWOConfig, run_native_bpwo
from .scp import SCPInstance


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance", required=True, type=Path)
    parser.add_argument("--known-optimum", type=float)
    parser.add_argument("--schemes", nargs="+", default=["V3-ELIT"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--evaluations", type=int, default=1_000)
    parser.add_argument("--movement-modes", nargs="+", default=["PWO"])
    parser.add_argument("--include-native", action="store_true")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--baseline-scheme", default="V3-ELIT")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--history-output",
        type=Path,
        help="CSV opcional con la trayectoria y métricas de mecanismo por iteración.",
    )
    return parser.parse_args()


def run_experiment(
    instance: SCPInstance,
    *,
    scheme_names: list[str],
    seeds: list[int],
    population: int,
    evaluations: int,
    movement_modes: list[str] | tuple[str, ...] = ("PWO",),
    include_native: bool = False,
    include_baselines: bool = False,
    baseline_scheme: str = "V3-ELIT",
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Ejecuta greedy y las configuraciones BPWO para una instancia."""

    rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []

    started = time.perf_counter()
    greedy = solve_greedy(instance)
    rows.append(
        {
            "instance": instance.name,
            "algorithm": "GREEDY",
            "movement_mode": "N/A",
            "scheme": "N/A",
            "seed": "N/A",
            "population": "N/A",
            "evaluation_budget": 1,
            "evaluations": 1,
            "best_found_at_evaluation": 1,
            "cost": greedy.cost,
            "rpd": instance.rpd(greedy.cost),
            "feasible": instance.is_feasible(greedy.solution),
            "initially_feasible_rate": "N/A",
            "added_columns": greedy.repair.added_columns,
            "removed_columns": greedy.repair.removed_columns,
            "runtime_seconds": time.perf_counter() - started,
            "python": platform.python_version(),
            "numpy": np.__version__,
        }
    )

    for movement_mode in movement_modes:
        for scheme_name in scheme_names:
            scheme = BinarizationScheme.parse(scheme_name)
            config = BPWOConfig(
                population_size=population,
                max_evaluations=evaluations,
                scheme=scheme,
                movement_mode=movement_mode,
            )
            for seed in seeds:
                algorithm_name = {
                    "PWO": "BPWO",
                    "PWO-DR": "BPWO_DR",
                    "IID": "ABLATION_IID",
                }[config.movement_mode]
                started = time.perf_counter()
                result = run_bpwo(instance, config, seed=seed)
                rows.append(
                    {
                        "instance": instance.name,
                        "algorithm": algorithm_name,
                        "movement_mode": config.movement_mode,
                        "scheme": scheme.name,
                        "seed": seed,
                        "population": config.population_size,
                        "evaluation_budget": config.max_evaluations,
                        "evaluations": result.evaluations,
                        "best_found_at_evaluation": result.best_found_at_evaluation,
                        "cost": result.best_cost,
                        "rpd": instance.rpd(result.best_cost),
                        "feasible": instance.is_feasible(result.best_solution),
                        "initially_feasible_rate": (
                            result.initially_feasible_proposals / result.proposals
                        ),
                        "added_columns": result.added_columns,
                        "removed_columns": result.removed_columns,
                        "runtime_seconds": time.perf_counter() - started,
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                    }
                )
                for record in result.history:
                    history_rows.append(
                        {
                            "instance": instance.name,
                            "algorithm": algorithm_name,
                            "movement_mode": config.movement_mode,
                            "scheme": scheme.name,
                            "seed": seed,
                            "iteration": record.iteration,
                            "evaluations": record.evaluations,
                            "best_cost": record.best_cost,
                            "rally_cohesion": record.rally_cohesion,
                            "attack_threshold": record.attack_threshold,
                            "exploitation": record.exploitation,
                            "binary_diversity": record.binary_diversity,
                            "initially_feasible_rate": record.initially_feasible_rate,
                            "added_columns": record.added_columns,
                            "removed_columns": record.removed_columns,
                            "restart_rate": record.restart_rate,
                        }
                    )

    if include_native:
        native_config = NativeBPWOConfig(
            population_size=population,
            max_evaluations=evaluations,
        )
        for seed in seeds:
            started = time.perf_counter()
            result = run_native_bpwo(instance, native_config, seed=seed)
            rows.append(
                {
                    "instance": instance.name,
                    "algorithm": "NBPWO",
                    "movement_mode": "BINARY",
                    "scheme": "NATIVE",
                    "seed": seed,
                    "population": native_config.population_size,
                    "evaluation_budget": native_config.max_evaluations,
                    "evaluations": result.evaluations,
                    "best_found_at_evaluation": result.best_found_at_evaluation,
                    "cost": result.best_cost,
                    "rpd": instance.rpd(result.best_cost),
                    "feasible": instance.is_feasible(result.best_solution),
                    "initially_feasible_rate": (
                        result.initially_feasible_proposals / result.proposals
                    ),
                    "added_columns": result.added_columns,
                    "removed_columns": result.removed_columns,
                    "runtime_seconds": time.perf_counter() - started,
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                }
            )
            for record in result.history:
                history_rows.append(
                    {
                        "instance": instance.name,
                        "algorithm": "NBPWO",
                        "movement_mode": "BINARY",
                        "scheme": "NATIVE",
                        "seed": seed,
                        "iteration": record.iteration,
                        "evaluations": record.evaluations,
                        "best_cost": record.best_cost,
                        "rally_cohesion": record.rally_cohesion,
                        "attack_threshold": record.attack_threshold,
                        "exploitation": record.exploitation,
                        "binary_diversity": record.binary_diversity,
                        "initially_feasible_rate": record.initially_feasible_rate,
                        "added_columns": record.added_columns,
                        "removed_columns": record.removed_columns,
                        "restart_rate": record.restart_rate,
                    }
                )

    if include_baselines:
        baseline_config = BaselineConfig(
            population_size=population,
            max_evaluations=evaluations,
            scheme=BinarizationScheme.parse(baseline_scheme),
        )
        for algorithm_name, runner in (("BPSO", run_bpso), ("BGWO", run_bgwo)):
            for seed in seeds:
                started = time.perf_counter()
                result = runner(instance, baseline_config, seed=seed)
                rows.append(
                    {
                        "instance": instance.name,
                        "algorithm": algorithm_name,
                        "movement_mode": "REFERENCE",
                        "scheme": baseline_config.scheme.name,
                        "seed": seed,
                        "population": baseline_config.population_size,
                        "evaluation_budget": baseline_config.max_evaluations,
                        "evaluations": result.evaluations,
                        "best_found_at_evaluation": result.best_found_at_evaluation,
                        "cost": result.best_cost,
                        "rpd": instance.rpd(result.best_cost),
                        "feasible": instance.is_feasible(result.best_solution),
                        "initially_feasible_rate": (
                            result.initially_feasible_proposals / result.proposals
                        ),
                        "added_columns": result.added_columns,
                        "removed_columns": result.removed_columns,
                        "runtime_seconds": time.perf_counter() - started,
                        "python": platform.python_version(),
                        "numpy": np.__version__,
                    }
                )
                for record in result.history:
                    history_rows.append(
                        {
                            "instance": instance.name,
                            "algorithm": algorithm_name,
                            "movement_mode": "REFERENCE",
                            "scheme": baseline_config.scheme.name,
                            "seed": seed,
                            "iteration": record.iteration,
                            "evaluations": record.evaluations,
                            "best_cost": record.best_cost,
                            "rally_cohesion": "N/A",
                            "attack_threshold": "N/A",
                            "exploitation": "N/A",
                            "binary_diversity": record.binary_diversity,
                            "initially_feasible_rate": (
                                record.initially_feasible_rate
                            ),
                            "added_columns": record.added_columns,
                            "removed_columns": record.removed_columns,
                            "restart_rate": "N/A",
                        }
                    )

    return rows, history_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Escribe filas homogéneas y rechaza una salida vacía."""

    if not rows:
        raise ValueError("No hay resultados para escribir.")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    instance = SCPInstance.from_orlib(
        args.instance,
        known_optimum=args.known_optimum,
    )
    rows, history_rows = run_experiment(
        instance,
        scheme_names=args.schemes,
        seeds=args.seeds,
        population=args.population,
        evaluations=args.evaluations,
        movement_modes=args.movement_modes,
        include_native=args.include_native,
        include_baselines=args.include_baselines,
        baseline_scheme=args.baseline_scheme,
    )

    write_csv(args.output, rows)

    if args.history_output is not None:
        write_csv(args.history_output, history_rows)

    print(f"Se escribieron {len(rows)} resultados en {args.output}")
    if args.history_output is not None:
        print(
            f"Se escribieron {len(history_rows)} registros de trayectoria "
            f"en {args.history_output}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
