"""Modelo y lector de instancias OR-Library para Set Covering."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


BinaryVector = NDArray[np.int8]


def _as_binary_vector(solution: NDArray[np.integer] | list[int], size: int) -> BinaryVector:
    vector = np.asarray(solution, dtype=np.int8)
    if vector.shape != (size,):
        raise ValueError(f"Se esperaba una solución de dimensión {size}, no {vector.shape}.")
    if not np.all((vector == 0) | (vector == 1)):
        raise ValueError("La solución debe contener solamente valores 0 y 1.")
    return vector


@dataclass(frozen=True)
class SCPInstance:
    """Instancia ponderada del Set Covering Problem."""

    name: str
    costs: NDArray[np.float64]
    coverage: NDArray[np.int8]
    known_optimum: float | None = None
    column_rows: tuple[NDArray[np.int64], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    row_columns: tuple[NDArray[np.int64], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        costs = np.asarray(self.costs, dtype=float)
        coverage = np.asarray(self.coverage, dtype=np.int8)

        if coverage.ndim != 2:
            raise ValueError("La matriz de cobertura debe tener dos dimensiones.")
        if costs.ndim != 1 or costs.size != coverage.shape[1]:
            raise ValueError("Debe existir un costo por cada columna de cobertura.")
        if np.any(costs < 0):
            raise ValueError("Los costos de SCP no pueden ser negativos.")
        if not np.all((coverage == 0) | (coverage == 1)):
            raise ValueError("La matriz de cobertura debe ser binaria.")
        if np.any(coverage.sum(axis=1) == 0):
            raise ValueError("Cada fila debe poder ser cubierta por al menos una columna.")

        object.__setattr__(self, "costs", costs)
        object.__setattr__(self, "coverage", coverage)
        object.__setattr__(
            self,
            "column_rows",
            tuple(
                np.flatnonzero(coverage[:, column]).astype(np.int64)
                for column in range(coverage.shape[1])
            ),
        )
        object.__setattr__(
            self,
            "row_columns",
            tuple(
                np.flatnonzero(coverage[row]).astype(np.int64)
                for row in range(coverage.shape[0])
            ),
        )

    @property
    def n_rows(self) -> int:
        return int(self.coverage.shape[0])

    @property
    def n_columns(self) -> int:
        return int(self.coverage.shape[1])

    @classmethod
    def from_orlib(
        cls,
        path: str | Path,
        *,
        known_optimum: float | None = None,
        name: str | None = None,
    ) -> "SCPInstance":
        """Lee el formato SCP de OR-Library basado en una secuencia de enteros."""

        source = Path(path)
        try:
            tokens = [int(token) for token in source.read_text(encoding="utf-8").split()]
        except OSError as exc:
            raise ValueError(f"No se pudo leer la instancia {source}: {exc}") from exc
        except ValueError as exc:
            raise ValueError(f"La instancia {source} contiene un token no entero.") from exc

        if len(tokens) < 2:
            raise ValueError("La instancia no contiene filas y columnas.")

        cursor = 0
        n_rows, n_columns = tokens[cursor], tokens[cursor + 1]
        cursor += 2
        if n_rows <= 0 or n_columns <= 0:
            raise ValueError("Las dimensiones de la instancia deben ser positivas.")
        if len(tokens) < cursor + n_columns:
            raise ValueError("La instancia no contiene todos los costos.")

        costs = np.asarray(tokens[cursor : cursor + n_columns], dtype=float)
        cursor += n_columns
        coverage = np.zeros((n_rows, n_columns), dtype=np.int8)

        for row in range(n_rows):
            if cursor >= len(tokens):
                raise ValueError(f"Falta la definición de cobertura para la fila {row + 1}.")
            count = tokens[cursor]
            cursor += 1
            if count <= 0 or cursor + count > len(tokens):
                raise ValueError(f"Cantidad de columnas inválida en la fila {row + 1}.")
            columns = tokens[cursor : cursor + count]
            cursor += count
            for column in columns:
                if not 1 <= column <= n_columns:
                    raise ValueError(
                        f"Índice de columna {column} fuera de rango en la fila {row + 1}."
                    )
                coverage[row, column - 1] = 1

        if cursor != len(tokens):
            raise ValueError("La instancia contiene datos adicionales no interpretados.")

        return cls(
            name=name or source.stem,
            costs=costs,
            coverage=coverage,
            known_optimum=known_optimum,
        )

    def validate_solution(self, solution: NDArray[np.integer] | list[int]) -> BinaryVector:
        return _as_binary_vector(solution, self.n_columns)

    def coverage_counts(self, solution: NDArray[np.integer] | list[int]) -> NDArray[np.int64]:
        vector = self.validate_solution(solution)
        return self.coverage @ vector.astype(np.int64)

    def is_feasible(self, solution: NDArray[np.integer] | list[int]) -> bool:
        return bool(np.all(self.coverage_counts(solution) >= 1))

    def cost(self, solution: NDArray[np.integer] | list[int]) -> float:
        vector = self.validate_solution(solution)
        return float(self.costs @ vector)

    def rpd(self, cost: float) -> float | None:
        if self.known_optimum is None:
            return None
        if self.known_optimum == 0:
            raise ValueError("El valor de referencia debe ser distinto de cero para calcular RPD.")
        return 100.0 * (float(cost) - self.known_optimum) / self.known_optimum
