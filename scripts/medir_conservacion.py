#!/usr/bin/env python3

"""Mide el solapamiento real entre el incumbente y la propuesta, antes de reparar.

La sección 5.6 del informe usa esta medición como evidencia principal y conserva
el proxy $\\widehat C_{rep}$ como aproximación diagnóstica. Este script produce
las 165 filas que sostienen la Tabla 6 sin modificar el código de producción.

La regla ELIT de (15) escribe el bit del incumbente cuando el sorteo cae bajo la
transferencia y cero en otro caso, de modo que para una columna activa en el
incumbente la probabilidad de que siga activa es exactamente la transferencia
evaluada en esa componente. Eso permite anotar tres cantidades por propuesta:

* `esperada`      media de la transferencia sobre las columnas activas del
  incumbente. Es la predicción, calculada sin mirar el resultado.
* `real`          fracción de esas columnas que efectivamente sobrevive.
* `estado_medio`  media del valor absoluto de la variable de estado sobre esas
  mismas columnas. Verifica el supuesto que sostiene el cálculo analítico, que
  la sincronización (4) deja el estado en ±1, y explica el caso de BPSO, cuya
  velocidad no converge a un valor común entre instancias.

Cada fila de la salida corresponde a una corrida, esto es a una combinación de
configuración, instancia y semilla. Las propuestas dentro de una corrida están
relacionadas temporalmente y no son observaciones independientes, de modo que la
incertidumbre debe estimarse entre corridas y no entre propuestas.

El envoltorio llama a la implementación original de `BinarizationScheme.apply` y
solo observa sus argumentos y su retorno. El consumo del generador aleatorio no
cambia, de modo que las corridas son idénticas a las que produciría el
experimento sin instrumentar.

Las instancias no se versionan en este repositorio. Hay que indicar dónde están
con `--instancias-raiz`.

Uso:

    .venv/bin/python scripts/medir_conservacion.py \\
        --instancias-raiz ../OII-450-1-2024/src/problem/SCP/Instances

Una corrida completa de once instancias por tres semillas toma alrededor de dos
horas, dominada por las dos instancias de diez mil columnas. Para una prueba
rápida conviene `--instancias scp62 --semillas 1 --evaluaciones 600`.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "src"))

from bpwo.binarization import BinarizationScheme  # noqa: E402
from bpwo.scp import SCPInstance  # noqa: E402
from bpwo.experiment import run_experiment  # noqa: E402
from bpwo.baselines import BaselineConfig, run_bgwo, run_bpso  # noqa: E402

SALIDA = RAIZ / "results/conservacion/solapamiento-medido.csv"

# Óptimos conocidos de las once instancias de test, los mismos del manifiesto.
OPTIMOS = {
    "scp42": 512.0, "scp52": 302.0, "scp62": 146.0, "scpa2": 252.0,
    "scpb2": 76.0, "scpc2": 219.0, "scpd2": 66.0, "scpnre2": 30.0,
    "scpnrf2": 15.0, "scpnrg2": 154.0, "scpnrh2": 63.0,
}

# Valores de referencia de la transferencia. En BPWO e IID son predicciones; en
# BGWO, T(1) es una referencia para medir el déficit; en BPSO, T(0) es una cota.
ANALITICO = {
    "BPSO": 0.0, "IID": 0.4142, "BGWO": 0.7071,
    "BPWO-V3": 0.7071, "BPWO-S2": 0.7311,
}

_acumulador: dict[tuple[str, str, int], dict[str, float]] = defaultdict(
    lambda: defaultdict(float)
)
_etiqueta: dict[str, tuple[str, str, int]] = {"actual": ("?", "?", -1)}
_original = BinarizationScheme.apply


def _observado(self, values, *, best, current, rng):
    """Envuelve la binarización para anotar la conservación de cada propuesta."""

    resultado = _original(self, values, best=best, current=current, rng=rng)
    if self.rule != "ELIT":
        return resultado
    activo = np.asarray(best, dtype=bool)
    activas = int(activo.sum())
    if activas:
        componentes = np.asarray(values, dtype=float)[activo]
        serie = _acumulador[_etiqueta["actual"]]
        serie["esperada"] += float(self.probabilities(componentes).mean())
        serie["real"] += float(np.asarray(resultado, dtype=bool)[activo].sum()) / activas
        serie["estado"] += float(np.abs(componentes).mean())
        serie["activas"] += activas
        serie["propuestas"] += 1.0
    return resultado


BinarizationScheme.apply = _observado


def medir(instancia: SCPInstance, nombre: str, semilla: int, evaluaciones: int) -> None:
    """Ejecuta las cinco configuraciones del test sobre una instancia y semilla."""

    for esquema, modo, rotulo in (
        ("S2-ELIT", "PWO", "BPWO-S2"),
        ("V3-ELIT", "PWO", "BPWO-V3"),
        ("V3-ELIT", "IID", "IID"),
    ):
        _etiqueta["actual"] = (nombre, rotulo, semilla)
        run_experiment(
            instancia,
            scheme_names=[esquema],
            seeds=[semilla],
            population=10,
            evaluations=evaluaciones,
            movement_modes=[modo],
        )
    configuracion = BaselineConfig(
        population_size=10,
        max_evaluations=evaluaciones,
        scheme=BinarizationScheme("V3", "ELIT"),
    )
    for rotulo, ejecutor in (("BPSO", run_bpso), ("BGWO", run_bgwo)):
        _etiqueta["actual"] = (nombre, rotulo, semilla)
        ejecutor(instancia, configuracion, seed=semilla)


def main() -> None:
    analizador = argparse.ArgumentParser(description=__doc__)
    analizador.add_argument("--instancias-raiz", type=Path, required=True)
    analizador.add_argument("--instancias", nargs="+", default=list(OPTIMOS))
    analizador.add_argument("--semillas", type=int, default=3)
    analizador.add_argument("--evaluaciones", type=int, default=6000)
    analizador.add_argument("--salida", type=Path, default=SALIDA)
    argumentos = analizador.parse_args()

    for nombre in argumentos.instancias:
        ruta = argumentos.instancias_raiz / f"{nombre}.txt"
        if not ruta.exists():
            raise SystemExit(f"Falta la instancia {ruta}.")
        instancia = SCPInstance.from_orlib(ruta, known_optimum=OPTIMOS[nombre])
        for semilla in range(argumentos.semillas):
            inicio = time.perf_counter()
            medir(instancia, nombre, semilla, argumentos.evaluaciones)
            print(
                f"  {nombre} semilla {semilla}: "
                f"{time.perf_counter() - inicio:.0f}s",
                flush=True,
            )

    filas = []
    for (instancia_nombre, rotulo, semilla), serie in sorted(_acumulador.items()):
        n = serie["propuestas"]
        filas.append(
            {
                "configuracion": rotulo,
                "instancia": instancia_nombre,
                "semilla": semilla,
                "propuestas": int(n),
                "activas_medias": round(serie["activas"] / n, 1),
                "esperada": round(serie["esperada"] / n, 4),
                "real": round(serie["real"] / n, 4),
                "estado_medio": round(serie["estado"] / n, 4),
                "analitico": ANALITICO[rotulo],
            }
        )
    argumentos.salida.parent.mkdir(parents=True, exist_ok=True)
    with argumentos.salida.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)

    print(f"\n{'configuración':10s} {'refer':>7s} {'real':>8s} {'|real−ref|':>10s} "
          f"{'|estado|':>9s} {'corridas':>9s}")
    for rotulo in ANALITICO:
        sub = [f for f in filas if f["configuracion"] == rotulo]
        if not sub:
            continue
        reales = [float(f["real"]) for f in sub]
        estados = [float(f["estado_medio"]) for f in sub]
        error = statistics.fmean(abs(r - ANALITICO[rotulo]) for r in reales)
        print(f"{rotulo:10s} {ANALITICO[rotulo]:7.4f} {statistics.fmean(reales):8.4f} "
              f"{error:10.4f} {statistics.fmean(estados):9.4f} {len(sub):9d}")
    try:
        destino = argumentos.salida.resolve().relative_to(RAIZ)
    except ValueError:
        destino = argumentos.salida
    print(f"\nEscrito: {destino} ({len(filas)} filas)")


if __name__ == "__main__":
    main()
