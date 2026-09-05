"""Variante binaria nativa de PWO para comparar con la técnica de dos pasos."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .algorithm import BPWOResult, IterationRecord
from .repair import repair_solution
from .scp import SCPInstance


@dataclass(frozen=True)
class NativeBPWOConfig:
    population_size: int = 10
    max_evaluations: int = 6_000
    vote_threshold: float = 0.5
    remove_redundant: bool = True

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("NBPWO requiere al menos dos agentes.")
        if self.max_evaluations < self.population_size:
            raise ValueError("El presupuesto debe permitir evaluar la población inicial.")
        if not 0.0 <= self.vote_threshold <= 1.0:
            raise ValueError("vote_threshold debe pertenecer a [0, 1].")


def _binary_diversity(population: np.ndarray) -> float:
    frequencies = population.mean(axis=0)
    return float(np.mean(2.0 * frequencies * (1.0 - frequencies)))


def _rally_cohesion(costs: np.ndarray, alpha_index: int, threshold: float) -> float:
    best = float(np.min(costs))
    worst = float(np.max(costs))
    denominator = worst - best
    if denominator <= np.finfo(float).eps:
        quality = np.zeros_like(costs)
    else:
        quality = (costs - best) / denominator
    voters = np.ones(costs.size, dtype=bool)
    voters[alpha_index] = False
    return float(np.count_nonzero((quality <= threshold) & voters) / (costs.size - 1))


def _mutate(
    candidate: np.ndarray,
    progress: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Retira una columna con probabilidad decreciente para reconstruir cobertura."""

    mutated = candidate.copy()
    selected = np.flatnonzero(mutated == 1)
    if selected.size > 0 and rng.random() < max(0.0, 1.0 - progress):
        mutated[int(rng.choice(selected))] = 0
    return mutated


def _explore_random(
    current: np.ndarray,
    random_agent: np.ndarray,
    progress: float,
    rng: np.random.Generator,
) -> np.ndarray:
    crossover = rng.random(current.size) < 0.5
    candidate = np.where(crossover, random_agent, current).astype(np.int8)
    return _mutate(candidate, progress, rng)


def _explore_alpha_majority(
    current: np.ndarray,
    alpha: np.ndarray,
    majority: np.ndarray,
    progress: float,
    rng: np.random.Generator,
) -> np.ndarray:
    target = np.where(rng.random(current.size) < 0.5, alpha, majority)
    crossover = rng.random(current.size) < 0.5
    candidate = np.where(crossover, target, current).astype(np.int8)
    return _mutate(candidate, progress, rng)


def _exploit_alpha(
    current: np.ndarray,
    alpha: np.ndarray,
    cohesion: float,
    progress: float,
    rng: np.random.Generator,
) -> np.ndarray:
    copy_probability = 0.5 + 0.5 * progress * cohesion
    copy_mask = (current != alpha) & (rng.random(current.size) < copy_probability)
    candidate = current.copy()
    candidate[copy_mask] = alpha[copy_mask]
    return _mutate(candidate, progress, rng)


def run_native_bpwo(
    instance: SCPInstance,
    config: NativeBPWOConfig,
    *,
    seed: int,
) -> BPWOResult:
    """Ejecuta la variante binaria nativa con actualización sincrónica."""

    rng = np.random.default_rng(seed)
    n = config.population_size
    dim = instance.n_columns
    binary = rng.integers(0, 2, size=(n, dim), dtype=np.int8)
    costs = np.empty(n, dtype=float)
    evaluations = 0
    total_proposals = 0
    initially_feasible_proposals = 0
    total_added_columns = 0
    total_removed_columns = 0
    best_cost_seen = float("inf")
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
            best_cost_seen = costs[agent]
            best_found_at_evaluation = evaluations

    history: list[IterationRecord] = []
    iteration = 0

    while evaluations < config.max_evaluations:
        iteration += 1
        snapshot = binary.copy()
        snapshot_costs = costs.copy()
        alpha_index = int(np.argmin(snapshot_costs))
        alpha = snapshot[alpha_index].copy()
        majority = (snapshot.mean(axis=0) >= 0.5).astype(np.int8)
        cohesion = _rally_cohesion(snapshot_costs, alpha_index, config.vote_threshold)
        diversity_before_move = _binary_diversity(snapshot)
        progress = evaluations / config.max_evaluations
        attack_threshold = max(0.0, 1.0 - progress)
        exploitation = (
            cohesion >= attack_threshold
            and diversity_before_move > np.finfo(float).eps
        )

        feasible_before_repair = 0
        added_columns = 0
        removed_columns = 0
        iteration_proposals = 0

        for agent in range(n):
            if evaluations >= config.max_evaluations:
                break
            current = snapshot[agent]
            if exploitation:
                candidate = _exploit_alpha(current, alpha, cohesion, progress, rng)
            elif rng.random() < 0.5:
                random_index = int(rng.integers(0, n))
                candidate = _explore_random(
                    current,
                    snapshot[random_index],
                    progress,
                    rng,
                )
            else:
                candidate = _explore_alpha_majority(
                    current,
                    alpha,
                    majority,
                    progress,
                    rng,
                )

            candidate, repair = repair_solution(
                instance,
                candidate,
                remove_redundant=config.remove_redundant,
            )
            candidate_cost = instance.cost(candidate)
            evaluations += 1
            total_proposals += 1
            iteration_proposals += 1
            initially_feasible_proposals += int(repair.initial_feasible)
            feasible_before_repair += int(repair.initial_feasible)
            total_added_columns += repair.added_columns
            total_removed_columns += repair.removed_columns
            added_columns += repair.added_columns
            removed_columns += repair.removed_columns

            if candidate_cost <= snapshot_costs[agent]:
                binary[agent] = candidate
                costs[agent] = candidate_cost
            if candidate_cost < best_cost_seen:
                best_cost_seen = candidate_cost
                best_found_at_evaluation = evaluations

        best_index = int(np.argmin(costs))
        history.append(
            IterationRecord(
                iteration=iteration,
                evaluations=evaluations,
                best_cost=float(costs[best_index]),
                rally_cohesion=cohesion,
                attack_threshold=attack_threshold,
                exploitation=exploitation,
                binary_diversity=_binary_diversity(binary),
                initially_feasible_rate=(
                    feasible_before_repair / iteration_proposals
                ),
                added_columns=added_columns,
                removed_columns=removed_columns,
                restart_rate=0.0,
            )
        )

    best_index = int(np.argmin(costs))
    best_solution = binary[best_index].copy()
    if not instance.is_feasible(best_solution):
        raise AssertionError("NBPWO terminó con una mejor solución infactible.")

    return BPWOResult(
        best_solution=best_solution,
        best_cost=float(costs[best_index]),
        evaluations=evaluations,
        best_found_at_evaluation=best_found_at_evaluation,
        proposals=total_proposals,
        initially_feasible_proposals=initially_feasible_proposals,
        added_columns=total_added_columns,
        removed_columns=total_removed_columns,
        seed=seed,
        history=tuple(history),
    )
