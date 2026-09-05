"""Pruebas inferenciales para el experimento final previamente congelado."""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from .analysis import read_results
from .experiment import write_csv


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", default="BPWO|PWO|V3-ELIT")
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def configuration_id(row: dict[str, str]) -> str:
    return "|".join(
        (row["algorithm"], row["movement_mode"], row["scheme"])
    )


def cliffs_delta(reference: list[float], comparison: list[float]) -> float:
    """Calcula delta de Cliff; valor negativo favorece a la referencia."""

    if not reference or not comparison:
        raise ValueError("Cliff requiere dos muestras no vacías.")
    greater = 0
    lower = 0
    for left in reference:
        for right in comparison:
            greater += int(left > right)
            lower += int(left < right)
    return (greater - lower) / (len(reference) * len(comparison))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Ajusta p-valores por Holm y conserva el orden de entrada."""

    count = len(p_values)
    ordered = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * count
    previous = 0.0
    for rank, (original_index, p_value) in enumerate(ordered):
        candidate = min(1.0, (count - rank) * p_value)
        previous = max(previous, candidate)
        adjusted[original_index] = previous
    return adjusted


def _constant(values: Iterable[float]) -> bool:
    values = list(values)
    return bool(values) and min(values) == max(values)


def _scipy_stats():
    try:
        from scipy import stats
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "El análisis inferencial requiere la dependencia opcional "
            "'scipy'. Instale el extra del proyecto con: "
            "python -m pip install -e '.[analysis]'"
        ) from exc
    return stats


def _row(
    *,
    scope: str,
    instance: str,
    test: str,
    reference: str,
    comparison: str,
    statistic: float,
    p_value: float,
    p_value_holm: float | str,
    alpha: float,
    effect_name: str = "N/A",
    effect_value: float | str = "N/A",
    n_reference: int | str = "N/A",
    n_comparison: int | str = "N/A",
    notes: str = "",
) -> dict[str, object]:
    adjusted = p_value if p_value_holm == "N/A" else float(p_value_holm)
    return {
        "scope": scope,
        "instance": instance,
        "test": test,
        "reference": reference,
        "comparison": comparison,
        "statistic": statistic,
        "p_value": p_value,
        "p_value_holm": p_value_holm,
        "alpha": alpha,
        "reject_null": adjusted < alpha,
        "effect_name": effect_name,
        "effect_value": effect_value,
        "n_reference": n_reference,
        "n_comparison": n_comparison,
        "notes": notes,
    }


def inferential_analysis(
    rows: Iterable[dict[str, str]],
    *,
    reference: str,
    alpha: float = 0.05,
) -> list[dict[str, object]]:
    """Ejecuta pruebas por instancia y entre medianas de instancias."""

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha debe pertenecer a (0, 1).")
    stats = _scipy_stats()
    by_instance: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        if row["algorithm"] == "GREEDY":
            continue
        by_instance[row["instance"]][configuration_id(row)].append(
            float(row["cost"])
        )
    if not by_instance:
        raise ValueError("No hay resultados estocásticos para analizar.")

    output: list[dict[str, object]] = []
    for instance, groups in sorted(by_instance.items()):
        if reference not in groups:
            raise ValueError(
                f"La referencia {reference!r} no está presente en {instance}."
            )
        for configuration, values in sorted(groups.items()):
            if len(values) < 3:
                continue
            if _constant(values):
                statistic, p_value = 1.0, 1.0
                normality_note = (
                    "Muestra constante: la normalidad no es informativa y se "
                    "registra p=1 para evitar un valor indefinido."
                )
            else:
                statistic, p_value = stats.shapiro(values)
                normality_note = (
                    "Prueba diagnóstica de normalidad; no selecciona por sí "
                    "sola la prueba principal."
                )
            output.append(
                _row(
                    scope="instance",
                    instance=instance,
                    test="Shapiro-Wilk",
                    reference=configuration,
                    comparison="N/A",
                    statistic=float(statistic),
                    p_value=float(p_value),
                    p_value_holm="N/A",
                    alpha=alpha,
                    n_reference=len(values),
                    notes=normality_note,
                )
            )

        if len(groups) >= 2:
            combined = [value for values in groups.values() for value in values]
            if _constant(combined):
                statistic, p_value = 0.0, 1.0
            else:
                statistic, p_value = stats.kruskal(*groups.values())
            output.append(
                _row(
                    scope="instance",
                    instance=instance,
                    test="Kruskal-Wallis",
                    reference="ALL",
                    comparison="ALL",
                    statistic=float(statistic),
                    p_value=float(p_value),
                    p_value_holm="N/A",
                    alpha=alpha,
                    notes="Contraste global entre configuraciones estocásticas.",
                )
            )

        reference_values = groups[reference]
        comparisons = [key for key in sorted(groups) if key != reference]
        pairwise: list[tuple[str, float, float, float]] = []
        for comparison in comparisons:
            comparison_values = groups[comparison]
            if _constant([*reference_values, *comparison_values]):
                statistic = len(reference_values) * len(comparison_values) / 2
                p_value = 1.0
            else:
                statistic, p_value = stats.mannwhitneyu(
                    reference_values,
                    comparison_values,
                    alternative="two-sided",
                    method="auto",
                )
            pairwise.append(
                (
                    comparison,
                    float(statistic),
                    float(p_value),
                    cliffs_delta(reference_values, comparison_values),
                )
            )
        adjusted = holm_adjust([item[2] for item in pairwise])
        for item, adjusted_p in zip(pairwise, adjusted):
            comparison, statistic, p_value, effect = item
            output.append(
                _row(
                    scope="instance",
                    instance=instance,
                    test="Mann-Whitney U",
                    reference=reference,
                    comparison=comparison,
                    statistic=statistic,
                    p_value=p_value,
                    p_value_holm=adjusted_p,
                    alpha=alpha,
                    effect_name="Cliff delta",
                    effect_value=effect,
                    n_reference=len(reference_values),
                    n_comparison=len(groups[comparison]),
                    notes="Delta negativo favorece a la referencia en un problema de minimización.",
                )
            )

    common = set.intersection(*(set(groups) for groups in by_instance.values()))
    if reference not in common:
        raise ValueError("La referencia no está disponible en todas las instancias.")
    configurations = sorted(common)
    instance_names = sorted(by_instance)
    medians = {
        configuration: [
            statistics.median(by_instance[instance][configuration])
            for instance in instance_names
        ]
        for configuration in configurations
    }
    if len(configurations) >= 3 and len(instance_names) >= 3:
        statistic, p_value = stats.friedmanchisquare(
            *(medians[configuration] for configuration in configurations)
        )
        output.append(
            _row(
                scope="global",
                instance="ALL",
                test="Friedman",
                reference="ALL",
                comparison="ALL",
                statistic=float(statistic),
                p_value=float(p_value),
                p_value_holm="N/A",
                alpha=alpha,
                n_reference=len(instance_names),
                notes="Bloques: medianas de costo de las instancias de test.",
            )
        )

    global_pairwise: list[tuple[str, float, float]] = []
    for comparison in configurations:
        if comparison == reference:
            continue
        differences = [
            left - right
            for left, right in zip(medians[reference], medians[comparison])
        ]
        if all(value == 0 for value in differences):
            statistic, p_value = 0.0, 1.0
        else:
            statistic, p_value = stats.wilcoxon(
                medians[reference],
                medians[comparison],
                alternative="two-sided",
                zero_method="zsplit",
            )
        global_pairwise.append((comparison, float(statistic), float(p_value)))
    adjusted = holm_adjust([item[2] for item in global_pairwise])
    for item, adjusted_p in zip(global_pairwise, adjusted):
        comparison, statistic, p_value = item
        output.append(
            _row(
                scope="global",
                instance="ALL",
                test="Wilcoxon signed-rank",
                reference=reference,
                comparison=comparison,
                statistic=statistic,
                p_value=p_value,
                p_value_holm=adjusted_p,
                alpha=alpha,
                n_reference=len(instance_names),
                n_comparison=len(instance_names),
                notes="Comparación post-hoc de medianas por instancia, con corrección de Holm.",
            )
        )
    return output


def main() -> int:
    args = _parse_args()
    rows = inferential_analysis(
        read_results(args.input),
        reference=args.reference,
        alpha=args.alpha,
    )
    write_csv(args.output, rows)
    print(f"Se escribieron {len(rows)} resultados inferenciales en {args.output}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
