"""Confirmación paralela de S2-ELIT y V3-ELIT al presupuesto final."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .batch import load_manifest
from .experiment import run_experiment
from .final_experiment import _atomic_write, _tag_rows, combine_checkpoints
from .scp import SCPInstance


@dataclass(frozen=True)
class CalibrationTask:
    instance_path: Path
    known_optimum: float
    role: str
    scale: str
    reference_type: str
    seed: int
    population: int
    evaluations: int
    checkpoint_dir: Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--instances-root", required=True, type=Path)
    parser.add_argument("--role", default="calibration")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--evaluations", type=int, default=6_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--history-output", required=True, type=Path)
    return parser.parse_args()


def _paths(task: CalibrationTask) -> tuple[Path, Path]:
    stem = f"{task.instance_path.stem}__seed-{task.seed:02d}"
    return (
        task.checkpoint_dir / f"{stem}.results.csv",
        task.checkpoint_dir / f"{stem}.history.csv",
    )


def run_calibration_task(task: CalibrationTask) -> tuple[str, int, int]:
    results_path, history_path = _paths(task)
    stem = f"{task.instance_path.stem}__seed-{task.seed:02d}"
    if results_path.exists() and history_path.exists():
        return stem, 0, 0

    instance = SCPInstance.from_orlib(
        task.instance_path,
        known_optimum=task.known_optimum,
    )
    rows, history = run_experiment(
        instance,
        scheme_names=["S2-ELIT", "V3-ELIT"],
        seeds=[task.seed],
        population=task.population,
        evaluations=task.evaluations,
        movement_modes=["PWO"],
    )
    rows = [row for row in rows if row["algorithm"] != "GREEDY"]
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
        CalibrationTask(
            instance_path=args.instances_root / spec.filename,
            known_optimum=spec.known_optimum,
            role=role,
            scale=spec.scale,
            reference_type=spec.reference_type,
            seed=seed,
            population=args.population,
            evaluations=args.evaluations,
            checkpoint_dir=args.checkpoint_dir,
        )
        for spec in specs
        for seed in args.seeds
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_calibration_task, task): task for task in tasks}
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
