"""Reparación determinista e instrumentada para Set Covering."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .scp import BinaryVector, SCPInstance


@dataclass(frozen=True)
class RepairStats:
    initial_feasible: bool
    added_columns: int
    removed_columns: int


def repair_solution(
    instance: SCPInstance,
    solution: NDArray[np.integer] | list[int],
    *,
    remove_redundant: bool = True,
) -> tuple[BinaryVector, RepairStats]:
    """Repara filas no cubiertas y elimina columnas redundantes."""

    repaired = instance.validate_solution(solution).copy()
    counts = instance.coverage_counts(repaired)
    initial_feasible = bool(np.all(counts >= 1))
    added = 0
    uncovered = counts == 0
    new_coverage = instance.coverage.T @ uncovered.astype(np.int64)

    while np.any(uncovered):
        candidates = np.flatnonzero((repaired == 0) & (new_coverage > 0))
        if candidates.size == 0:
            raise ValueError("La instancia no permite reparar la solución recibida.")

        ratios = instance.costs[candidates] / new_coverage[candidates]
        order = np.lexsort((candidates, instance.costs[candidates], ratios))
        selected = int(candidates[order[0]])
        repaired[selected] = 1
        covered_rows = instance.column_rows[selected]
        newly_covered = covered_rows[counts[covered_rows] == 0]
        counts[covered_rows] += 1
        uncovered[newly_covered] = False
        for row in newly_covered:
            new_coverage[instance.row_columns[int(row)]] -= 1
        added += 1

    removed = 0
    if remove_redundant:
        selected_columns = np.flatnonzero(repaired == 1)
        order = sorted(
            selected_columns.tolist(),
            key=lambda column: (instance.costs[column], column),
            reverse=True,
        )
        for column in order:
            covered_rows = instance.column_rows[column]
            if np.all(counts[covered_rows] >= 2):
                repaired[column] = 0
                counts[covered_rows] -= 1
                removed += 1

    if not np.all(counts >= 1):
        raise AssertionError("La reparación produjo una solución infactible.")

    return repaired, RepairStats(
        initial_feasible=initial_feasible,
        added_columns=added,
        removed_columns=removed,
    )
