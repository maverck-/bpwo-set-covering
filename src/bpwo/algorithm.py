"""Primera especificación ejecutable de BPWO para Set Covering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .binarization import BinarizationScheme
from .repair import repair_solution
from .scp import BinaryVector, SCPInstance


@dataclass(frozen=True)
class BPWOConfig:
    population_size: int = 10
    max_evaluations: int = 6_000
    scheme: BinarizationScheme = BinarizationScheme("V3", "ELIT")
    vote_threshold: float = 0.5
    lower_bound: float = -1.0
    upper_bound: float = 1.0
    remove_redundant: bool = True
    movement_mode: str = "PWO"
    rally_mode: str = "THRESHOLD"
    binarization_input: str = "POSITION"
    synchronize_latent: bool = True
    latent_amplitude: float = 1.0
    degenerate_cohesion: str = "MAX"
    diversity_restart_threshold: float = 0.001
    restart_fraction: float = 0.2

    def __post_init__(self) -> None:
        if self.population_size < 2:
            raise ValueError("BPWO requiere al menos dos agentes.")
        if self.max_evaluations < self.population_size:
            raise ValueError("El presupuesto debe permitir evaluar la población inicial.")
        if not 0.0 <= self.vote_threshold <= 1.0:
            raise ValueError("vote_threshold debe pertenecer a [0, 1].")
        if not self.lower_bound < self.upper_bound:
            raise ValueError("El límite inferior debe ser menor que el superior.")
        if self.latent_amplitude <= 0.0:
            raise ValueError("latent_amplitude debe ser positiva.")
        if (
            self.latent_amplitude > self.upper_bound
            or -self.latent_amplitude < self.lower_bound
        ):
            raise ValueError(
                "latent_amplitude debe caber dentro de los límites del estado latente."
            )
        degenerate_cohesion = self.degenerate_cohesion.upper()
        if degenerate_cohesion not in {"MAX", "MIN"}:
            raise ValueError("degenerate_cohesion debe ser MAX o MIN.")
        movement_mode = self.movement_mode.upper()
        if movement_mode not in {"PWO", "PWO-DR", "IID", "ALPHA"}:
            raise ValueError("movement_mode debe ser PWO, PWO-DR, IID o ALPHA.")
        rally_mode = self.rally_mode.upper()
        if rally_mode not in {"THRESHOLD", "PROBABILISTIC"}:
            raise ValueError(
                "rally_mode debe ser THRESHOLD o PROBABILISTIC."
            )
        binarization_input = self.binarization_input.upper()
        if binarization_input not in {"POSITION", "DELTA"}:
            raise ValueError(
                "binarization_input debe ser POSITION o DELTA."
            )
        if not 0.0 <= self.diversity_restart_threshold <= 0.5:
            raise ValueError("diversity_restart_threshold debe pertenecer a [0, 0.5].")
        if not 0.0 < self.restart_fraction < 1.0:
            raise ValueError("restart_fraction debe pertenecer a (0, 1).")
        object.__setattr__(self, "movement_mode", movement_mode)
        object.__setattr__(self, "rally_mode", rally_mode)
        object.__setattr__(self, "binarization_input", binarization_input)
        object.__setattr__(self, "degenerate_cohesion", degenerate_cohesion)


@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    evaluations: int
    best_cost: float
    rally_cohesion: float
    attack_threshold: float
    exploitation: bool
    binary_diversity: float
    initially_feasible_rate: float
    added_columns: int
    removed_columns: int
    restart_rate: float


@dataclass(frozen=True)
class BPWOResult:
    best_solution: BinaryVector
    best_cost: float
    evaluations: int
    best_found_at_evaluation: int
    proposals: int
    initially_feasible_proposals: int
    added_columns: int
    removed_columns: int
    seed: int
    history: tuple[IterationRecord, ...]


def _binary_diversity(population: NDArray[np.int8]) -> float:
    frequencies = population.mean(axis=0)
    return float(np.mean(2.0 * frequencies * (1.0 - frequencies)))


def _rally_cohesion(
    costs: NDArray[np.float64],
    alpha_index: int,
    threshold: float,
    degenerate: str = "MAX",
) -> float:
    """Fracción de agentes no alfa que vota, según la calidad relativa.

    Cuando la población es homogénea en costo no existe señal de calidad. El
    modo ``MAX`` conserva la convención original, en la que todos los agentes
    votan y la cohesión vale 1; el modo ``MIN`` devuelve 0, que es la lectura
    coherente con la ausencia de señal y la que evita que la explotación quede
    fijada de forma permanente.
    """

    best = float(np.min(costs))
    worst = float(np.max(costs))
    denominator = worst - best
    if denominator <= np.finfo(float).eps:
        if degenerate == "MIN":
            return 0.0
        quality = np.zeros_like(costs)
    else:
        quality = (costs - best) / denominator
    voters = np.ones(costs.size, dtype=bool)
    voters[alpha_index] = False
    return float(np.count_nonzero((quality <= threshold) & voters) / (costs.size - 1))


def _explore_random_agent(
    current: NDArray[np.float64],
    random_agent: NDArray[np.float64],
    a: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    step = a * (2.0 * rng.random(current.size) - 1.0)
    return current + step * np.abs(random_agent - current)


def _explore_alpha_mean(
    current: NDArray[np.float64],
    alpha: NDArray[np.float64],
    mean: NDArray[np.float64],
    a: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    global_direction = rng.random(current.size) * (alpha - mean)
    local_spread = (2.0 * rng.random(current.size) - 1.0) * np.abs(alpha - current)
    return current + a * (global_direction + local_spread)


def _exploit_alpha(
    current: NDArray[np.float64],
    alpha: NDArray[np.float64],
    a: float,
    rally_cohesion: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    attack = a * (2.0 * rng.random(current.size) - 1.0) * (1.0 - rally_cohesion)
    return alpha - attack * np.abs(alpha - current)


def _perturb_alpha(
    alpha: NDArray[np.float64],
    a: float,
    amplitude: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Control de atribución: ruido isotrópico decreciente alrededor del alfa.

    Reemplaza las tres ecuaciones de PWO por una perturbación sin estructura
    que conserva el anclaje y el calendario de amplitud. Sirve para separar el
    aporte del anclaje del aporte de la dinámica.
    """

    return alpha + a * 0.5 * amplitude * (2.0 * rng.random(alpha.size) - 1.0)


def _rally_decision(
    *,
    cohesion: float,
    progress: float,
    mode: str,
    rng: np.random.Generator,
) -> tuple[bool, float]:
    """Decide explotación y devuelve el umbral o probabilidad aplicada."""

    if mode == "THRESHOLD":
        threshold = max(0.0, 1.0 - progress)
        return cohesion >= threshold, threshold
    exploitation_probability = float(np.clip(progress * cohesion, 0.0, 1.0))
    return rng.random() < exploitation_probability, exploitation_probability


def run_bpwo(instance: SCPInstance, config: BPWOConfig, *, seed: int) -> BPWOResult:
    """Ejecuta BPWO con actualización sincrónica y selección greedy por agente."""

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

    if config.synchronize_latent:
        continuous = config.latent_amplitude * (2.0 * binary.astype(float) - 1.0)

    history: list[IterationRecord] = []
    iteration = 0

    while evaluations < config.max_evaluations:
        iteration += 1
        snapshot_continuous = continuous.copy()
        snapshot_binary = binary.copy()
        snapshot_costs = costs.copy()
        alpha_index = int(np.argmin(snapshot_costs))
        alpha_continuous = snapshot_continuous[alpha_index].copy()
        alpha_binary = snapshot_binary[alpha_index].copy()
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
        restart_agents: set[int] = set()
        if (
            config.movement_mode == "PWO-DR"
            and diversity_before_move < config.diversity_restart_threshold
        ):
            candidates = [
                int(index)
                for index in np.argsort(snapshot_costs, kind="stable")[::-1]
                if int(index) != alpha_index
            ]
            restart_count = max(
                1,
                int(np.ceil(config.restart_fraction * (n - 1))),
            )
            restart_agents = set(candidates[:restart_count])
        restarted_proposals = 0

        for agent in range(n):
            if evaluations >= config.max_evaluations:
                break

            current = snapshot_continuous[agent]
            if config.movement_mode == "IID" or agent in restart_agents:
                candidate_continuous = rng.uniform(
                    config.lower_bound,
                    config.upper_bound,
                    size=dim,
                )
                restarted_proposals += int(agent in restart_agents)
            elif config.movement_mode == "ALPHA":
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
            binarization_values = (
                candidate_continuous
                if config.binarization_input == "POSITION"
                else candidate_continuous - current
            )
            candidate_binary = config.scheme.apply(
                binarization_values,
                best=alpha_binary,
                current=snapshot_binary[agent],
                rng=rng,
            )
            candidate_binary, repair = repair_solution(
                instance,
                candidate_binary,
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

            if candidate_cost <= snapshot_costs[agent]:
                binary[agent] = candidate_binary
                costs[agent] = candidate_cost
                if config.synchronize_latent:
                    continuous[agent] = config.latent_amplitude * (
                        2.0 * candidate_binary.astype(float) - 1.0
                    )
                else:
                    continuous[agent] = candidate_continuous

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
                exploitation=(
                    exploitation and config.movement_mode in {"PWO", "PWO-DR"}
                ),
                binary_diversity=_binary_diversity(binary),
                initially_feasible_rate=(
                    feasible_before_repair / iteration_proposals
                ),
                added_columns=added_columns,
                removed_columns=removed_columns,
                restart_rate=restarted_proposals / iteration_proposals,
            )
        )

    best_index = int(np.argmin(costs))
    best_solution = binary[best_index].copy()
    if not instance.is_feasible(best_solution):
        raise AssertionError("BPWO terminó con una mejor solución infactible.")

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
