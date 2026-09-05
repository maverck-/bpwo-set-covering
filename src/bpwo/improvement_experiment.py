"""Cribado paralelo y reanudable de mejoras posteriores al primer test."""

from __future__ import annotations

import argparse
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .algorithm import BPWOConfig, BPWOResult, run_bpwo
from .batch import load_manifest
from .binarization import BinarizationScheme
from .final_experiment import _atomic_write, _tag_rows, combine_checkpoints
from .scp import SCPInstance


@dataclass(frozen=True)
class ImprovementTask:
    instance_path: Path
    known_optimum: float
    role: str
    scale: str
    reference_type: str
    seed: int
    population: int
    evaluations: int
    checkpoint_dir: Path
    variants: tuple[str, ...] = (
        "BPWO_BASE",
        "BPWO_PROB",
        "BPWO_DELTA",
        "BPWO_DELTA_PROB",
    )


@dataclass(frozen=True)
class VariantSpec:
    algorithm: str
    scheme: BinarizationScheme
    rally_mode: str
    binarization_input: str


VARIANTS = (
    VariantSpec(
        algorithm="BPWO_BASE",
        scheme=BinarizationScheme("V3", "ELIT"),
        rally_mode="THRESHOLD",
        binarization_input="POSITION",
    ),
    VariantSpec(
        algorithm="BPWO_PROB",
        scheme=BinarizationScheme("V3", "ELIT"),
        rally_mode="PROBABILISTIC",
        binarization_input="POSITION",
    ),
    VariantSpec(
        algorithm="BPWO_DELTA",
        scheme=BinarizationScheme("V3", "COMP"),
        rally_mode="THRESHOLD",
        binarization_input="DELTA",
    ),
    VariantSpec(
        algorithm="BPWO_DELTA_PROB",
        scheme=BinarizationScheme("V3", "COMP"),
        rally_mode="PROBABILISTIC",
        binarization_input="DELTA",
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--instances-root", required=True, type=Path)
    parser.add_argument("--role", default="development")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--evaluations", type=int, default=1_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=[variant.algorithm for variant in VARIANTS],
        default=[variant.algorithm for variant in VARIANTS],
    )
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--history-output", required=True, type=Path)
    return parser.parse_args()


def _paths(task: ImprovementTask) -> tuple[Path, Path]:
    stem = f"{task.instance_path.stem}__seed-{task.seed:02d}"
    return (
        task.checkpoint_dir / f"{stem}.results.csv",
        task.checkpoint_dir / f"{stem}.history.csv",
    )


def _result_row(
    *,
    instance: SCPInstance,
    variant: VariantSpec,
    config: BPWOConfig,
    result: BPWOResult,
    runtime_seconds: float,
) -> dict[str, object]:
    return {
        "instance": instance.name,
        "algorithm": variant.algorithm,
        "movement_mode": config.movement_mode,
        "scheme": config.scheme.name,
        "rally_mode": config.rally_mode,
        "binarization_input": config.binarization_input,
        "seed": result.seed,
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
        "runtime_seconds": runtime_seconds,
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def _history_rows(
    *,
    instance: SCPInstance,
    variant: VariantSpec,
    config: BPWOConfig,
    result: BPWOResult,
) -> list[dict[str, object]]:
    return [
        {
            "instance": instance.name,
            "algorithm": variant.algorithm,
            "movement_mode": config.movement_mode,
            "scheme": config.scheme.name,
            "rally_mode": config.rally_mode,
            "binarization_input": config.binarization_input,
            "seed": result.seed,
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
        for record in result.history
    ]


def run_improvement_task(task: ImprovementTask) -> tuple[str, int, int]:
    """Ejecuta las cuatro variantes exactas para una instancia y semilla."""

    results_path, history_path = _paths(task)
    stem = f"{task.instance_path.stem}__seed-{task.seed:02d}"
    if results_path.exists() and history_path.exists():
        return stem, 0, 0

    instance = SCPInstance.from_orlib(
        task.instance_path,
        known_optimum=task.known_optimum,
    )
    rows: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    selected = [
        variant for variant in VARIANTS if variant.algorithm in task.variants
    ]
    if len(selected) != len(set(task.variants)):
        raise ValueError(f"Lista de variantes no reconocida: {task.variants!r}")
    for variant in selected:
        config = BPWOConfig(
            population_size=task.population,
            max_evaluations=task.evaluations,
            scheme=variant.scheme,
            movement_mode="PWO",
            rally_mode=variant.rally_mode,
            binarization_input=variant.binarization_input,
        )
        started = time.perf_counter()
        result = run_bpwo(instance, config, seed=task.seed)
        rows.append(
            _result_row(
                instance=instance,
                variant=variant,
                config=config,
                result=result,
                runtime_seconds=time.perf_counter() - started,
            )
        )
        history.extend(
            _history_rows(
                instance=instance,
                variant=variant,
                config=config,
                result=result,
            )
        )

    _tag_rows(rows, task)  # type: ignore[arg-type]
    _tag_rows(history, task)  # type: ignore[arg-type]
    task.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(results_path, rows)
    _atomic_write(history_path, history)
    return stem, len(rows), len(history)


def main() -> int:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("workers debe ser al menos 1.")
    role = args.role.strip().lower()
    specs = [spec for spec in load_manifest(args.manifest) if spec.role == role]
    if not specs:
        raise ValueError(f"No hay instancias con role={role!r}.")
    tasks = [
        ImprovementTask(
            instance_path=args.instances_root / spec.filename,
            known_optimum=spec.known_optimum,
            role=role,
            scale=spec.scale,
            reference_type=spec.reference_type,
            seed=seed,
            population=args.population,
            evaluations=args.evaluations,
            checkpoint_dir=args.checkpoint_dir,
            variants=tuple(args.variants),
        )
        for spec in specs
        for seed in args.seeds
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_improvement_task, task): task for task in tasks}
        for future in as_completed(futures):
            stem, result_count, history_count = future.result()
            status = "reutilizado" if result_count == 0 else "completado"
            print(
                f"{stem}: {status}, {result_count} resultados, "
                f"{history_count} registros de trayectoria",
                flush=True,
            )

    result_count, history_count = combine_checkpoints(
        args.checkpoint_dir,
        output=args.output,
        history_output=args.history_output,
    )
    print(
        f"Consolidado: {result_count} resultados y {history_count} registros.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
