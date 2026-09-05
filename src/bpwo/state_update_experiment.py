"""Ablación aislada de selección y actualización del estado latente.

El experimento conserva la inicialización, el movimiento, la binarización y la
reparación de ``run_bpwo``. Solo varía qué ocurre después de evaluar una
propuesta:

* ``GREEDY_SYNC`` acepta una propuesta no empeorante y sincroniza el estado
  latente en ``±kappa``. Es la referencia ya evaluada.
* ``ALWAYS_SYNC`` reemplaza el estado del agente en cada evaluación y conserva
  la sincronización dura.
* ``ALWAYS_FEEDBACK`` reemplaza el estado en cada evaluación y devuelve al
  latente únicamente los bits modificados por el reparador.
* ``ALPHA_ALWAYS_FEEDBACK`` aplica el mecanismo anterior, pero sustituye las
  ecuaciones de PWO por el control de perturbación alrededor del alfa.

La mejor solución global se conserva aparte en los brazos ``ALWAYS``. Esto
permite que la población se mueva sin perder el mejor valor encontrado.
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

from .algorithm import (
    BPWOConfig,
    BPWOResult,
    IterationRecord,
    _binary_diversity,
    _exploit_alpha,
    _explore_alpha_mean,
    _explore_random_agent,
    _perturb_alpha,
    _rally_cohesion,
    _rally_decision,
)
from .batch import load_manifest
from .binarization import BinarizationScheme
from .final_experiment import _atomic_write, _tag_rows, combine_checkpoints
from .repair import repair_solution
from .scp import SCPInstance


@dataclass(frozen=True)
class VariantSpec:
    algorithm: str
    selection_mode: str
    state_update_mode: str
    movement_mode: str
    purpose: str

    def __post_init__(self) -> None:
        selection = self.selection_mode.upper()
        update = self.state_update_mode.upper()
        movement = self.movement_mode.upper()
        if selection not in {"GREEDY", "ALWAYS"}:
            raise ValueError("selection_mode debe ser GREEDY o ALWAYS.")
        if update not in {"SYNC", "REPAIR_FEEDBACK"}:
            raise ValueError(
                "state_update_mode debe ser SYNC o REPAIR_FEEDBACK."
            )
        if movement not in {"PWO", "ALPHA"}:
            raise ValueError("movement_mode debe ser PWO o ALPHA.")
        object.__setattr__(self, "selection_mode", selection)
        object.__setattr__(self, "state_update_mode", update)
        object.__setattr__(self, "movement_mode", movement)


VARIANTS = (
    VariantSpec(
        algorithm="ANCHOR_GREEDY_SYNC",
        selection_mode="GREEDY",
        state_update_mode="SYNC",
        movement_mode="PWO",
        purpose="referencia ya evaluada: selección no empeorante y anclaje duro",
    ),
    VariantSpec(
        algorithm="ANCHOR_ALWAYS_SYNC",
        selection_mode="ALWAYS",
        state_update_mode="SYNC",
        movement_mode="PWO",
        purpose="aísla el efecto de permitir que todos los agentes se muevan",
    ),
    VariantSpec(
        algorithm="ALPHA_ALWAYS_SYNC",
        selection_mode="ALWAYS",
        state_update_mode="SYNC",
        movement_mode="ALPHA",
        purpose="control de atribución para la variante promovida en el cribado",
    ),
    VariantSpec(
        algorithm="PWO_ALWAYS_FEEDBACK",
        selection_mode="ALWAYS",
        state_update_mode="REPAIR_FEEDBACK",
        movement_mode="PWO",
        purpose="añade realimentación de los bits modificados por la reparación",
    ),
    VariantSpec(
        algorithm="ALPHA_ALWAYS_FEEDBACK",
        selection_mode="ALWAYS",
        state_update_mode="REPAIR_FEEDBACK",
        movement_mode="ALPHA",
        purpose="control de atribución sin ecuaciones de movimiento PWO",
    ),
)


@dataclass(frozen=True)
class InstrumentedResult:
    result: BPWOResult
    iterative_proposals: int
    accepted_proposals: int
    changed_binary_proposals: int
    global_improvements: int


@dataclass(frozen=True)
class StateUpdateTask:
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
    parser.add_argument("--role", default="development")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--evaluations", type=int, default=300)
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


def build_config(*, population: int, evaluations: int) -> BPWOConfig:
    return BPWOConfig(
        population_size=population,
        max_evaluations=evaluations,
        scheme=BinarizationScheme("S1", "STD"),
        movement_mode="PWO",
        degenerate_cohesion="MIN",
        latent_amplitude=4.0,
        lower_bound=-6.0,
        upper_bound=6.0,
    )


def run_state_update_bpwo(
    instance: SCPInstance,
    config: BPWOConfig,
    variant: VariantSpec,
    *,
    seed: int,
    repair_feedback: float = 1.0,
) -> InstrumentedResult:
    """Ejecuta una variante sin alterar el algoritmo consolidado del proyecto."""

    if config.scheme.name != "S1-STD":
        raise ValueError("La ablación está fijada a S1-STD.")
    if config.degenerate_cohesion != "MIN":
        raise ValueError("La ablación requiere la convención MIN del rally.")
    if repair_feedback < 0.0:
        raise ValueError("repair_feedback no puede ser negativo.")

    rng = np.random.default_rng(seed)
    n = config.population_size
    dim = instance.n_columns
    continuous = rng.uniform(config.lower_bound, config.upper_bound, size=(n, dim))
    binary = rng.integers(0, 2, size=(n, dim), dtype=np.int8)
    costs = np.empty(n, dtype=float)
    evaluations = 0
    total_proposals = 0
    initially_feasible_proposals = 0
    total_added_columns = 0
    total_removed_columns = 0
    best_cost_seen = float("inf")
    best_solution_seen = np.zeros(dim, dtype=np.int8)
    best_found_at_evaluation = 0

    for agent in range(n):
        binary[agent], repair = repair_solution(
            instance,
            binary[agent],
            remove_redundant=config.remove_redundant,
        )
        costs[agent] = instance.cost(binary[agent])
        evaluations += 1
        total_proposals += 1
        initially_feasible_proposals += int(repair.initial_feasible)
        total_added_columns += repair.added_columns
        total_removed_columns += repair.removed_columns
        if costs[agent] < best_cost_seen:
            best_cost_seen = float(costs[agent])
            best_solution_seen = binary[agent].copy()
            best_found_at_evaluation = evaluations

    continuous = config.latent_amplitude * (2.0 * binary.astype(float) - 1.0)

    history: list[IterationRecord] = []
    iteration = 0
    accepted_proposals = 0
    changed_binary_proposals = 0
    global_improvements = 0

    while evaluations < config.max_evaluations:
        iteration += 1
        snapshot_continuous = continuous.copy()
        snapshot_binary = binary.copy()
        snapshot_costs = costs.copy()
        alpha_index = int(np.argmin(snapshot_costs))
        if variant.selection_mode == "GREEDY":
            alpha_continuous = snapshot_continuous[alpha_index].copy()
            alpha_binary = snapshot_binary[alpha_index].copy()
        else:
            alpha_binary = best_solution_seen.copy()
            alpha_continuous = config.latent_amplitude * (
                2.0 * alpha_binary.astype(float) - 1.0
            )
        mean_position = snapshot_continuous.mean(axis=0)
        cohesion = _rally_cohesion(
            snapshot_costs,
            alpha_index,
            config.vote_threshold,
            config.degenerate_cohesion,
        )
        diversity_before_move = _binary_diversity(snapshot_binary)
        progress = evaluations / config.max_evaluations
        exploitation, attack_threshold = _rally_decision(
            cohesion=cohesion,
            progress=progress,
            mode=config.rally_mode,
            rng=rng,
        )
        a = max(0.0, 2.0 * (1.0 - progress))

        feasible_before_repair = 0
        added_columns = 0
        removed_columns = 0
        iteration_proposals = 0

        for agent in range(n):
            if evaluations >= config.max_evaluations:
                break

            current = snapshot_continuous[agent]
            if variant.movement_mode == "ALPHA":
                candidate_continuous = _perturb_alpha(
                    alpha_continuous, a, config.latent_amplitude, rng
                )
            elif exploitation:
                candidate_continuous = _exploit_alpha(
                    current, alpha_continuous, a, cohesion, rng
                )
            elif rng.random() < 0.5:
                random_index = int(rng.integers(0, n))
                candidate_continuous = _explore_random_agent(
                    current, snapshot_continuous[random_index], a, rng
                )
            else:
                candidate_continuous = _explore_alpha_mean(
                    current, alpha_continuous, mean_position, a, rng
                )

            candidate_continuous = np.clip(
                candidate_continuous,
                config.lower_bound,
                config.upper_bound,
            )
            decoded_binary = config.scheme.apply(
                candidate_continuous,
                best=alpha_binary,
                current=snapshot_binary[agent],
                rng=rng,
            )
            candidate_binary, repair = repair_solution(
                instance,
                decoded_binary,
                remove_redundant=config.remove_redundant,
            )
            candidate_cost = instance.cost(candidate_binary)
            evaluations += 1
            iteration_proposals += 1
            total_proposals += 1
            feasible_before_repair += int(repair.initial_feasible)
            added_columns += repair.added_columns
            removed_columns += repair.removed_columns
            initially_feasible_proposals += int(repair.initial_feasible)
            total_added_columns += repair.added_columns
            total_removed_columns += repair.removed_columns

            accepted = (
                variant.selection_mode == "ALWAYS"
                or candidate_cost <= snapshot_costs[agent]
            )
            if accepted:
                accepted_proposals += 1
                changed_binary_proposals += int(
                    not np.array_equal(candidate_binary, snapshot_binary[agent])
                )
                binary[agent] = candidate_binary
                costs[agent] = candidate_cost
                if variant.state_update_mode == "SYNC":
                    continuous[agent] = config.latent_amplitude * (
                        2.0 * candidate_binary.astype(float) - 1.0
                    )
                else:
                    continuous[agent] = np.clip(
                        candidate_continuous
                        + repair_feedback
                        * (candidate_binary.astype(float) - decoded_binary.astype(float)),
                        config.lower_bound,
                        config.upper_bound,
                    )

            if candidate_cost < best_cost_seen:
                best_cost_seen = candidate_cost
                best_solution_seen = candidate_binary.copy()
                best_found_at_evaluation = evaluations
                global_improvements += 1

        if variant.selection_mode == "GREEDY":
            current_best = float(np.min(costs))
        else:
            current_best = best_cost_seen
        history.append(
            IterationRecord(
                iteration=iteration,
                evaluations=evaluations,
                best_cost=current_best,
                rally_cohesion=cohesion,
                attack_threshold=attack_threshold,
                exploitation=(
                    exploitation and variant.movement_mode == "PWO"
                ),
                binary_diversity=_binary_diversity(binary),
                initially_feasible_rate=(
                    feasible_before_repair / iteration_proposals
                ),
                added_columns=added_columns,
                removed_columns=removed_columns,
                restart_rate=0.0,
            )
        )

    if variant.selection_mode == "GREEDY":
        best_index = int(np.argmin(costs))
        best_solution = binary[best_index].copy()
        best_cost = float(costs[best_index])
    else:
        best_solution = best_solution_seen.copy()
        best_cost = best_cost_seen
    if not instance.is_feasible(best_solution):
        raise AssertionError("La mejor solución conservada es infactible.")

    result = BPWOResult(
        best_solution=best_solution,
        best_cost=best_cost,
        evaluations=evaluations,
        best_found_at_evaluation=best_found_at_evaluation,
        proposals=total_proposals,
        initially_feasible_proposals=initially_feasible_proposals,
        added_columns=total_added_columns,
        removed_columns=total_removed_columns,
        seed=seed,
        history=tuple(history),
    )
    return InstrumentedResult(
        result=result,
        iterative_proposals=evaluations - n,
        accepted_proposals=accepted_proposals,
        changed_binary_proposals=changed_binary_proposals,
        global_improvements=global_improvements,
    )


def _paths(task: StateUpdateTask) -> tuple[Path, Path]:
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
    instrumented: InstrumentedResult,
    runtime_seconds: float,
) -> dict[str, object]:
    result = instrumented.result
    denominator = instrumented.iterative_proposals
    return {
        "instance": instance.name,
        "algorithm": variant.algorithm,
        "movement_mode": variant.movement_mode,
        "scheme": config.scheme.name,
        "selection_mode": variant.selection_mode,
        "state_update_mode": variant.state_update_mode,
        "degenerate_cohesion": config.degenerate_cohesion,
        "latent_amplitude": config.latent_amplitude,
        "repair_feedback": 1.0,
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
        "acceptance_rate": (
            instrumented.accepted_proposals / denominator if denominator else 0.0
        ),
        "binary_state_change_rate": (
            instrumented.changed_binary_proposals / denominator
            if denominator
            else 0.0
        ),
        "global_improvements": instrumented.global_improvements,
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
            "movement_mode": variant.movement_mode,
            "scheme": config.scheme.name,
            "selection_mode": variant.selection_mode,
            "state_update_mode": variant.state_update_mode,
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


def run_state_update_task(task: StateUpdateTask) -> tuple[str, int, int]:
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
    config = build_config(population=task.population, evaluations=task.evaluations)

    rows: list[dict[str, object]] = []
    history: list[dict[str, object]] = []
    for variant in selected:
        started = time.perf_counter()
        instrumented = run_state_update_bpwo(
            instance,
            config,
            variant,
            seed=task.seed,
        )
        rows.append(
            _result_row(
                instance=instance,
                variant=variant,
                config=config,
                instrumented=instrumented,
                runtime_seconds=time.perf_counter() - started,
            )
        )
        history.extend(
            _history_rows(
                instance=instance,
                variant=variant,
                config=config,
                result=instrumented.result,
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
        StateUpdateTask(
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
        futures = {
            executor.submit(run_state_update_task, task): task for task in tasks
        }
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
