"""Funciones de transferencia y reglas de binarización del cribado 2 x 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatVector = NDArray[np.float64]
BinaryVector = NDArray[np.int8]


def _stable_logistic(values: NDArray[np.floating] | list[float], scale: float) -> FloatVector:
    """Logística ``1 / (1 + exp(-x / scale))`` evaluada sin desbordamiento."""

    x = np.asarray(values, dtype=float) / scale
    result = np.empty_like(x, dtype=float)
    positive = x >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1.0 + exp_x)
    return result


def transfer_s1(values: NDArray[np.floating] | list[float]) -> FloatVector:
    """Sigmoide S1 de Mirjalili y Lewis: ``1 / (1 + exp(-2x))``.

    Es la más abrupta de la familia S. Con el estado latente anclado en
    ``±kappa``, decide el bit de forma casi determinista, lo que devuelve al
    movimiento el control sobre qué componentes cambian.
    """

    return _stable_logistic(values, 0.5)


def transfer_s2(values: NDArray[np.floating] | list[float]) -> FloatVector:
    """Sigmoide S2: ``1 / (1 + exp(-x))``, evaluada de forma estable."""

    return _stable_logistic(values, 1.0)


def transfer_v3(values: NDArray[np.floating] | list[float]) -> FloatVector:
    """Función V3: ``abs(x / sqrt(1 + x**2))``."""

    x = np.asarray(values, dtype=float)
    return np.abs(x / np.sqrt(1.0 + np.square(x)))


TRANSFER_FUNCTIONS = {
    "S1": transfer_s1,
    "S2": transfer_s2,
    "V3": transfer_v3,
}


@dataclass(frozen=True)
class BinarizationScheme:
    """Combinación explícita de función de transferencia y regla binaria."""

    transfer: str = "V3"
    rule: str = "ELIT"

    def __post_init__(self) -> None:
        transfer = self.transfer.upper()
        rule = self.rule.upper()
        if transfer not in TRANSFER_FUNCTIONS:
            raise ValueError(f"Función de transferencia no soportada: {self.transfer}")
        if rule not in {"STD", "COMP", "ELIT"}:
            raise ValueError(f"Regla de binarización no soportada: {self.rule}")
        object.__setattr__(self, "transfer", transfer)
        object.__setattr__(self, "rule", rule)

    @classmethod
    def parse(cls, value: str) -> "BinarizationScheme":
        try:
            transfer, rule = value.split("-", maxsplit=1)
        except ValueError as exc:
            raise ValueError("El esquema debe usar el formato TRANSFER-REGLA.") from exc
        return cls(transfer=transfer, rule=rule)

    @property
    def name(self) -> str:
        return f"{self.transfer}-{self.rule}"

    def probabilities(self, values: NDArray[np.floating] | list[float]) -> FloatVector:
        probabilities = TRANSFER_FUNCTIONS[self.transfer](values)
        return np.clip(probabilities, 0.0, 1.0)

    def apply(
        self,
        values: NDArray[np.floating] | list[float],
        *,
        best: NDArray[np.integer] | list[int],
        current: NDArray[np.integer] | list[int],
        rng: np.random.Generator,
    ) -> BinaryVector:
        probabilities = self.probabilities(values)
        best_vector = np.asarray(best, dtype=np.int8)
        current_vector = np.asarray(current, dtype=np.int8)
        if probabilities.shape != best_vector.shape or best_vector.shape != current_vector.shape:
            raise ValueError("Los vectores continuo, mejor y actual deben tener igual dimensión.")
        if not np.all((best_vector == 0) | (best_vector == 1)):
            raise ValueError("La mejor solución debe ser binaria.")
        if not np.all((current_vector == 0) | (current_vector == 1)):
            raise ValueError("La solución actual debe ser binaria.")

        selected = rng.random(probabilities.shape) < probabilities
        if self.rule == "STD":
            return selected.astype(np.int8)
        if self.rule == "COMP":
            return np.where(selected, 1 - current_vector, current_vector).astype(
                np.int8
            )
        return np.where(selected, best_vector, 0).astype(np.int8)
