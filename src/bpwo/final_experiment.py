"""Experimento final paralelo, reanudable y separado de la calibración."""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from .batch import InstanceSpec, load_manifest
from .experiment import run_experiment, write_csv
from .scp import SCPInstance


@dataclass(frozen=True)
class SeedTask:
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
    parser.add_argument("--role", default="test")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(31)))
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--evaluations", type=int, default=6_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--history-output", required=True, type=Path)
    return parser.parse_args()


def _stem(task: SeedTask) -> str:
    return f"{task.instance_path.stem}__seed-{task.seed:02d}"


def _checkpoint_paths(task: SeedTask) -> tuple[Path, Path]:
    stem = _stem(task)
    return (
        task.checkpoint_dir / f"{stem}.results.csv",
        task.checkpoint_dir / f"{stem}.history.csv",
    )


def _tag_rows(rows: list[dict[str, object]], task: SeedTask) -> None:
    for row in rows:
        row["role"] = task.role
        row["scale"] = task.scale
        row["reference_type"] = task.reference_type


def _atomic_write(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    write_csv(temporary, rows)
    temporary.replace(path)


def run_seed_task(task: SeedTask) -> tuple[str, int, int]:
    """Ejecuta la suite congelada para una instancia y una semilla."""

    results_path, history_path = _checkpoint_paths(task)
    if results_path.exists() and history_path.exists():
        return _stem(task), 0, 0

    instance = SCPInstance.from_orlib(
        task.instance_path,
        known_optimum=task.known_optimum,
    )
    s2_rows, s2_history = run_experiment(
        instance,
        scheme_names=["S2-ELIT"],
        seeds=[task.seed],
        population=task.population,
        evaluations=task.evaluations,
        movement_modes=["PWO"],
    )
    reference_rows, reference_history = run_experiment(
        instance,
        scheme_names=["V3-ELIT"],
        seeds=[task.seed],
        population=task.population,
        evaluations=task.evaluations,
        movement_modes=["PWO", "IID"],
        include_baselines=True,
        baseline_scheme="V3-ELIT",
    )

    rows = [row for row in s2_rows if row["algorithm"] != "GREEDY"]
    rows.extend(
        row for row in reference_rows if row["algorithm"] != "GREEDY"
    )
    history = [*s2_history, *reference_history]
    _tag_rows(rows, task)
    _tag_rows(history, task)
    task.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(results_path, rows)
    _atomic_write(history_path, history)
    return _stem(task), len(rows), len(history)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _greedy_checkpoint(
    spec: InstanceSpec,
    *,
    instances_root: Path,
    role: str,
    checkpoint_dir: Path,
) -> Path:
    output = checkpoint_dir / f"{Path(spec.filename).stem}__greedy.results.csv"
    if output.exists():
        return output
    instance = SCPInstance.from_orlib(
        instances_root / spec.filename,
        known_optimum=spec.known_optimum,
    )
    rows, _ = run_experiment(
        instance,
        scheme_names=[],
        seeds=[],
        population=3,
        evaluations=3,
    )
    for row in rows:
        row["role"] = role
        row["scale"] = spec.scale
        row["reference_type"] = spec.reference_type
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write(output, rows)
    return output


def combine_checkpoints(
    checkpoint_dir: Path,
    *,
    output: Path,
    history_output: Path,
) -> tuple[int, int]:
    """Combina solo checkpoints completos y conserva las filas crudas."""

    result_rows: list[dict[str, str]] = []
    history_rows: list[dict[str, str]] = []
    for path in sorted(checkpoint_dir.glob("*.results.csv")):
        result_rows.extend(_read_csv(path))
    for path in sorted(checkpoint_dir.glob("*.history.csv")):
        history_rows.extend(_read_csv(path))
    result_rows.sort(
        key=lambda row: (
            row["instance"],
            row["algorithm"],
            row["scheme"],
            row["seed"],
        )
    )
    history_rows.sort(
        key=lambda row: (
            row["instance"],
            row["algorithm"],
            row["scheme"],
            row["seed"],
            int(row["iteration"]),
        )
    )
    write_csv(output, result_rows)
    write_csv(history_output, history_rows)
    return len(result_rows), len(history_rows)


def main() -> int:
    args = _parse_args()
    if args.workers < 1:
        raise ValueError("workers debe ser al menos 1.")
    role = args.role.strip().lower()
    specs = [spec for spec in load_manifest(args.manifest) if spec.role == role]
    if not specs:
        raise ValueError(f"No hay instancias con role={role!r}.")

    for spec in specs:
        _greedy_checkpoint(
            spec,
            instances_root=args.instances_root,
            role=role,
            checkpoint_dir=args.checkpoint_dir,
        )

    tasks = [
        SeedTask(
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
        futures = {executor.submit(run_seed_task, task): task for task in tasks}
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
