"""BPSO y BGWO de referencia bajo el mismo pipeline de SCP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .binarization import BinarizationScheme
from .repair import repair_solution
from .scp import BinaryVector, SCPInstance


@dataclass(frozen=True)
class BaselineConfig:
    population_size: int = 10
    max_evaluations: int = 6_000
    scheme: BinarizationScheme = BinarizationScheme("V3", "ELIT")
    remove_redundant: bool = True
    inertia_max: float = 0.9
    inertia_min: float = 0.1
    cognitive: float = 2.0
    social: float = 2.0

    def __post_init__(self) -> None:
        if self.population_size < 3:
            raise ValueError("Los comparadores requieren al menos tres agentes.")
        if self.max_evaluations < self.population_size:
            raise ValueError("El presupuesto debe evaluar la población inicial.")
        if not 0.0 <= self.inertia_min <= self.inertia_max:
            raise ValueError("Se requiere 0 <= inertia_min <= inertia_max.")


@dataclass(frozen=True)
class BaselineRecord:
    iteration: int
    evaluations: int
    best_cost: float
    binary_diversity: float
    initially_feasible_rate: float
    added_columns: int
    removed_columns: int


@dataclass(frozen=True)
class BaselineResult:
    best_solution: BinaryVector
    best_cost: float
    evaluations: int
    best_found_at_evaluation: int
    proposals: int
    initially_feasible_proposals: int
    added_columns: int
    removed_columns: int
    seed: int
    history: tuple[BaselineRecord, ...]


@dataclass
class _InitialState:
    binary: np.ndarray
    costs: np.ndarray
    evaluations: int
    proposals: int
    initially_feasible: int
    added_columns: int
    removed_columns: int
    best_solution: BinaryVector
    best_cost: float
    best_found_at_evaluation: int


def _binary_diversity(population: np.ndarray) -> float:
    frequencies = population.mean(axis=0)
    return float(np.mean(2.0 * frequencies * (1.0 - frequencies)))


def _initialize(
    instance: SCPInstance,
    config: BaselineConfig,
    rng: np.random.Generator,
) -> _InitialState:
    binary = rng.integers(
        0,
        2,
        size=(config.population_size, instance.n_columns),
        dtype=np.int8,
    )
    costs = np.empty(config.population_size, dtype=float)
    initially_feasible = 0
    added_columns = 0
    removed_columns = 0
    best_cost = float("inf")
    best_solution = binary[0].copy()
    best_found_at_evaluation = 0

    for agent in range(config.population_size):
        binary[agent], repair = repair_solution(
            instance,
            binary[agent],
            remove_redundant=config.remove_redundant,
        )
        costs[agent] = instance.cost(binary[agent])
        initially_feasible += int(repair.initial_feasible)
        added_columns += repair.added_columns
        removed_columns += repair.removed_columns
        if costs[agent] < best_cost:
            best_cost = float(costs[agent])
            best_solution = binary[agent].copy()
            best_found_at_evaluation = agent + 1

    return _InitialState(
        binary=binary,
        costs=costs,
        evaluations=config.population_size,
        proposals=config.population_size,
        initially_feasible=initially_feasible,
        added_columns=added_columns,
        removed_columns=removed_columns,
        best_solution=best_solution,
        best_cost=best_cost,
        best_found_at_evaluation=best_found_at_evaluation,
    )


def _result(state: _InitialState, seed: int, history: list[BaselineRecord]) -> BaselineResult:
    return BaselineResult(
        best_solution=state.best_solution.copy(),
        best_cost=state.best_cost,
        evaluations=state.evaluations,
        best_found_at_evaluation=state.best_found_at_evaluation,
        proposals=state.proposals,
        initially_feasible_proposals=state.initially_feasible,
        added_columns=state.added_columns,
        removed_columns=state.removed_columns,
        seed=seed,
        history=tuple(history),
    )


def run_bpso(
    instance: SCPInstance,
    config: BaselineConfig,
    *,
    seed: int,
) -> BaselineResult:
    """Ejecuta BPSO con memoria personal y global, V3-ELIT y reparación."""

    rng = np.random.default_rng(seed)
    state = _initialize(instance, config, rng)
    velocities = np.zeros_like(state.binary, dtype=float)
    personal_best = state.binary.copy()
    personal_costs = state.costs.copy()
    history: list[BaselineRecord] = []
    iteration = 0

    while state.evaluations < config.max_evaluations:
        iteration += 1
        snapshot_binary = state.binary.copy()
        snapshot_velocities = velocities.copy()
        global_index = int(np.argmin(personal_costs))
        global_best = personal_best[global_index].copy()
        progress = state.evaluations / config.max_evaluations
        inertia = config.inertia_max - (
            config.inertia_max - config.inertia_min
        ) * progress
        feasible_before_repair = 0
        added_columns = 0
        removed_columns = 0
        iteration_proposals = 0

        for agent in range(config.population_size):
            if state.evaluations >= config.max_evaluations:
                break
            r1 = rng.random(instance.n_columns)
            r2 = rng.random(instance.n_columns)
            velocity = (
                inertia * snapshot_velocities[agent]
                + config.cognitive
                * r1
                * (personal_best[agent].astype(float) - snapshot_binary[agent])
                + config.social
                * r2
                * (global_best.astype(float) - snapshot_binary[agent])
            )
            candidate = config.scheme.apply(
                velocity,
                best=global_best,
                current=snapshot_binary[agent],
                rng=rng,
            )
            candidate, repair = repair_solution(
                instance,
                candidate,
                remove_redundant=config.remove_redundant,
            )
            candidate_cost = instance.cost(candidate)
            velocities[agent] = velocity
            state.binary[agent] = candidate
            state.costs[agent] = candidate_cost
            state.evaluations += 1
            state.proposals += 1
            iteration_proposals += 1
            state.initially_feasible += int(repair.initial_feasible)
            feasible_before_repair += int(repair.initial_feasible)
            state.added_columns += repair.added_columns
            state.removed_columns += repair.removed_columns
            added_columns += repair.added_columns
            removed_columns += repair.removed_columns

            if candidate_cost < personal_costs[agent]:
                personal_best[agent] = candidate
                personal_costs[agent] = candidate_cost
            if candidate_cost < state.best_cost:
                state.best_solution = candidate.copy()
                state.best_cost = candidate_cost
                state.best_found_at_evaluation = state.evaluations

        history.append(
            BaselineRecord(
                iteration=iteration,
                evaluations=state.evaluations,
                best_cost=state.best_cost,
                binary_diversity=_binary_diversity(state.binary),
                initially_feasible_rate=(
                    feasible_before_repair / iteration_proposals
                ),
                added_columns=added_columns,
                removed_columns=removed_columns,
            )
        )

    return _result(state, seed, history)


def run_bgwo(
    instance: SCPInstance,
    config: BaselineConfig,
    *,
    seed: int,
) -> BaselineResult:
    """Ejecuta BGWO con tres líderes, V3-ELIT y reparación."""

    rng = np.random.default_rng(seed)
    state = _initialize(instance, config, rng)
    continuous = 2.0 * state.binary.astype(float) - 1.0
    history: list[BaselineRecord] = []
    iteration = 0

    while state.evaluations < config.max_evaluations:
        iteration += 1
        snapshot_binary = state.binary.copy()
        snapshot_continuous = continuous.copy()
        leader_indices = np.argsort(state.costs, kind="stable")[:3]
        leaders = snapshot_continuous[leader_indices]
        alpha_binary = snapshot_binary[int(leader_indices[0])].copy()
        progress = state.evaluations / config.max_evaluations
        a = 2.0 * (1.0 - progress)
        feasible_before_repair = 0
        added_columns = 0
        removed_columns = 0
        iteration_proposals = 0

        for agent in range(config.population_size):
            if state.evaluations >= config.max_evaluations:
                break
            current = snapshot_continuous[agent]
            estimates = []
            for leader in leaders:
                coefficient_a = 2.0 * a * rng.random(instance.n_columns) - a
                coefficient_c = 2.0 * rng.random(instance.n_columns)
                distance = np.abs(coefficient_c * leader - current)
                estimates.append(leader - coefficient_a * distance)
            proposal = np.mean(estimates, axis=0)
            candidate = config.scheme.apply(
                proposal,
                best=alpha_binary,
                current=snapshot_binary[agent],
                rng=rng,
            )
            candidate, repair = repair_solution(
                instance,
                candidate,
                remove_redundant=config.remove_redundant,
            )
            candidate_cost = instance.cost(candidate)
            state.binary[agent] = candidate
            state.costs[agent] = candidate_cost
            continuous[agent] = 2.0 * candidate.astype(float) - 1.0
            state.evaluations += 1
            state.proposals += 1
            iteration_proposals += 1
            state.initially_feasible += int(repair.initial_feasible)
            feasible_before_repair += int(repair.initial_feasible)
            state.added_columns += repair.added_columns
            state.removed_columns += repair.removed_columns
            added_columns += repair.added_columns
            removed_columns += repair.removed_columns

            if candidate_cost < state.best_cost:
                state.best_solution = candidate.copy()
                state.best_cost = candidate_cost
                state.best_found_at_evaluation = state.evaluations

        history.append(
            BaselineRecord(
                iteration=iteration,
                evaluations=state.evaluations,
                best_cost=state.best_cost,
                binary_diversity=_binary_diversity(state.binary),
                initially_feasible_rate=(
                    feasible_before_repair / iteration_proposals
                ),
                added_columns=added_columns,
                removed_columns=removed_columns,
            )
        )

    return _result(state, seed, history)
