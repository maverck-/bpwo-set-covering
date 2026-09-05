"""Ejecución por lotes a partir de un manifiesto congelado de instancias."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from .experiment import run_experiment, write_csv
from .scp import SCPInstance


@dataclass(frozen=True)
class InstanceSpec:
    filename: str
    known_optimum: float
    role: str
    scale: str = "unspecified"
    reference_type: str = "unspecified"


def load_manifest(path: Path) -> tuple[InstanceSpec, ...]:
    """Lee y valida las columnas mínimas del manifiesto experimental."""

    with path.open(encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        required = {"filename", "known_optimum", "role"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(
                "El manifiesto requiere filename, known_optimum y role."
            )
        specs = tuple(
            InstanceSpec(
                filename=row["filename"].strip(),
                known_optimum=float(row["known_optimum"]),
                role=row["role"].strip().lower(),
                scale=row.get("scale", "unspecified").strip().lower(),
                reference_type=row.get(
                    "reference_type", "unspecified"
                ).strip().lower(),
            )
            for row in reader
        )

    if not specs:
        raise ValueError("El manifiesto no contiene instancias.")
    if any(not spec.filename or not spec.role for spec in specs):
        raise ValueError("filename y role no pueden estar vacíos.")
    return specs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--instances-root", required=True, type=Path)
    parser.add_argument("--role", default="calibration")
    parser.add_argument(
        "--schemes",
        nargs="+",
        default=["S2-STD", "S2-ELIT", "V3-COMP", "V3-ELIT"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--evaluations", type=int, default=300)
    parser.add_argument("--movement-modes", nargs="+", default=["PWO"])
    parser.add_argument("--include-native", action="store_true")
    parser.add_argument("--include-baselines", action="store_true")
    parser.add_argument("--baseline-scheme", default="V3-ELIT")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--history-output", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    selected_role = args.role.strip().lower()
    specs = tuple(
        spec for spec in load_manifest(args.manifest) if spec.role == selected_role
    )
    if not specs:
        raise ValueError(f"No hay instancias con role={selected_role!r}.")

    all_rows: list[dict[str, object]] = []
    all_history: list[dict[str, object]] = []
    for spec in specs:
        instance = SCPInstance.from_orlib(
            args.instances_root / spec.filename,
            known_optimum=spec.known_optimum,
        )
        rows, history = run_experiment(
            instance,
            scheme_names=args.schemes,
            seeds=args.seeds,
            population=args.population,
            evaluations=args.evaluations,
            movement_modes=args.movement_modes,
            include_native=args.include_native,
            include_baselines=args.include_baselines,
            baseline_scheme=args.baseline_scheme,
        )
        for row in rows:
            row["role"] = selected_role
            row["scale"] = spec.scale
            row["reference_type"] = spec.reference_type
        for row in history:
            row["role"] = selected_role
            row["scale"] = spec.scale
            row["reference_type"] = spec.reference_type
        all_rows.extend(rows)
        all_history.extend(history)

    write_csv(args.output, all_rows)
    write_csv(args.history_output, all_history)
    print(
        f"Se procesaron {len(specs)} instancias, {len(all_rows)} resultados "
        f"y {len(all_history)} registros de trayectoria."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
