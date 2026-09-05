"""Línea base greedy determinista para Set Covering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .repair import RepairStats, repair_solution
from .scp import BinaryVector, SCPInstance


@dataclass(frozen=True)
class GreedyResult:
    solution: BinaryVector
    cost: float
    repair: RepairStats


def solve_greedy(instance: SCPInstance) -> GreedyResult:
    """Construye una cobertura desde cero con la misma heurística de reparación."""

    initial = np.zeros(instance.n_columns, dtype=np.int8)
    solution, stats = repair_solution(instance, initial, remove_redundant=True)
    return GreedyResult(solution=solution, cost=instance.cost(solution), repair=stats)

