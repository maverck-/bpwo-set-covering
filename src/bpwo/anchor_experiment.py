"""Confirmación del diagnóstico del rally y del anclaje latente con amplitud.

Este experimento responde dos preguntas abiertas por la auditoría del primer
test, y se ejecuta solo sobre instancias de calibración para no consumir el
conjunto de prueba ya observado ni las particiones reservadas.

1. ¿La cohesión degenerada explica el colapso de BPWO? Con la población
   homogénea en costo, la convención heredada devuelve cohesión 1, el
   coeficiente de ataque se anula y cada propuesta se reduce al alfa. La
   variante ``BPWO_RALLY`` cambia esa única convención.
2. ¿El anclaje latente con amplitud devuelve al movimiento el control sobre
   los bits? ``BPWO_ANCHOR`` sincroniza el estado real en ``±kappa`` y usa la
   sigmoide S1 con regla estándar, de modo que el signo del latente decide el
   bit y el desplazamiento decide qué signos cambian.

``ALPHA_S1`` es el control de atribución: conserva el anclaje y el calendario
de amplitud, pero reemplaza las ecuaciones de PWO por ruido isotrópico
alrededor del alfa. Si iguala a ``BPWO_ANCHOR``, la mejora pertenece al
anclaje y no a la dinámica.

3. ¿La regla binaria descartada en el cribado resistía el presupuesto final?
   ``BPWO_S2_STD`` y ``BPWO_V3_COMP`` reevalúan las dos reglas que sí pueden
   encender una columna ausente del alfa, sobre el estado sin anclar de la
   configuración congelada.

Las seis variantes comparten ejecutor, semillas, presupuesto e instancias, de
modo que sus filas son comparables entre sí.
"""

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
class VariantSpec:
    algorithm: str
    scheme: BinarizationScheme
    movement_mode: str
    degenerate_cohesion: str
    latent_amplitude: float
    bound: float
    purpose: str


VARIANTS = (
    VariantSpec(
        algorithm="BPWO_BASE",
        scheme=BinarizationScheme("V3", "ELIT"),
        movement_mode="PWO",
        degenerate_cohesion="MAX",
        latent_amplitude=1.0,
        bound=1.0,
        purpose="configuración congelada del primer test, como referencia",
    ),
    VariantSpec(
        algorithm="BPWO_RALLY",
        scheme=BinarizationScheme("V3", "ELIT"),
        movement_mode="PWO",
        degenerate_cohesion="MIN",
        latent_amplitude=1.0,
        bound=1.0,
        purpose="aísla la corrección de la cohesión degenerada",
    ),
    VariantSpec(
        algorithm="BPWO_ANCHOR",
        scheme=BinarizationScheme("S1", "STD"),
        movement_mode="PWO",
        degenerate_cohesion="MIN",
        latent_amplitude=4.0,
        bound=6.0,
        purpose="rally corregido más anclaje latente con amplitud",
    ),
    VariantSpec(
        algorithm="BPWO_S2_STD",
        scheme=BinarizationScheme("S2", "STD"),
        movement_mode="PWO",
        degenerate_cohesion="MAX",
        latent_amplitude=1.0,
        bound=1.0,
        purpose=(
            "reevalúa al presupuesto final la regla estándar descartada en el "
            "cribado, que decide cada bit por la transferencia y puede encender "
            "una columna ausente del alfa"
        ),
    ),
    VariantSpec(
        algorithm="BPWO_V3_COMP",
        scheme=BinarizationScheme("V3", "COMP"),
        movement_mode="PWO",
        degenerate_cohesion="MAX",
        latent_amplitude=1.0,
        bound=1.0,
        purpose=(
            "reevalúa al presupuesto final la regla de complemento descartada "
            "en el cribado, que invierte el bit vigente en vez de copiar el alfa"
        ),
    ),
    VariantSpec(
        algorithm="ALPHA_S1",
        scheme=BinarizationScheme("S1", "STD"),
        movement_mode="ALPHA",
        degenerate_cohesion="MIN",
        latent_amplitude=4.0,
        bound=6.0,
        purpose="control de atribución: anclaje sin ecuaciones de PWO",
    ),
)


@dataclass(frozen=True)
class AnchorTask:
    instance_path: Path
    known_optimum: float
    role: str
    scale: str
    reference_type: str
    seed: int
    population: int
    evaluations: int
    checkpoint_dir: Path
    variants: tuple[str, ...]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--instances-root", required=True, type=Path)
    parser.add_argument("--role", default="calibration")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--evaluations", type=int, default=6_000)
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


def _paths(task: AnchorTask) -> tuple[Path, Path]:
    stem = f"{task.instance_path.stem}__seed-{task.seed:02d}"
    return (
        task.checkpoint_dir / f"{stem}.results.csv",
        task.checkpoint_dir / f"{stem}.history.csv",
    )


def build_config(variant: VariantSpec, *, population: int, evaluations: int) -> BPWOConfig:
    return BPWOConfig(
        population_size=population,
        max_evaluations=evaluations,
        scheme=variant.scheme,
        movement_mode=variant.movement_mode,
        degenerate_cohesion=variant.degenerate_cohesion,
        latent_amplitude=variant.latent_amplitude,
        lower_bound=-variant.bound,
        upper_bound=variant.bound,
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
        "degenerate_cohesion": config.degenerate_cohesion,
        "latent_amplitude": config.latent_amplitude,
        "bound": config.upper_bound,
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
            "degenerate_cohesion": config.degenerate_cohesion,
            "latent_amplitude": config.latent_amplitude,
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


def run_anchor_task(task: AnchorTask) -> tuple[str, int, int]:
    """Ejecuta las variantes seleccionadas para una instancia y una semilla."""

    results_path, history_path = _paths(task)
    stem = f"{task.instance_path.stem}__seed-{task.seed:02d}"
    if results_path.exists() and history_path.exists():
        return stem, 0, 0

    instance = SCPInstance.from_orlib(
        task.instance_path,
        known_optimum=task.known_optimum,
    )
    selected = [variant for variant in VARIANTS if variant.algorithm in task.variants]
    if len(selected) != len(set(task.variants)):
        raise ValueError(f"Lista de variantes no reconocida: {task.variants!r}")

    rows: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    for variant in selected:
        config = build_config(
            variant,
            population=task.population,
            evaluations=task.evaluations,
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
        AnchorTask(
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
        futures = {executor.submit(run_anchor_task, task): task for task in tasks}
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
