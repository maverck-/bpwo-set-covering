#!/usr/bin/env python3

"""Deriva la Tabla 7 del informe desde las trayectorias del contraste dirigido.

La tabla mezcla dos experimentos que comparten ejecutor, semillas, presupuesto e
instancias: `results/anchor/` y `results/rules/`. Las métricas por evaluación se
calculan sobre las evaluaciones del ciclo y no sobre el presupuesto completo,
porque las diez evaluaciones de inicialización provienen de una población
aleatoria densa que el reparador poda de forma masiva y desplazaría los
promedios.

El archivo `results/rules/contraste-dirigido.csv` queda versionado. Las
trayectorias de las que se deriva no lo están, por peso, de modo que este
script es el puente verificable entre unas y otras. `tests/test_tabla_contraste.py`
compara ese CSV contra las cifras publicadas en el informe.

Uso:

    .venv/bin/python scripts/derivar_contraste_dirigido.py
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
TRAYECTORIAS = (RAIZ / "results/anchor/history.csv", RAIZ / "results/rules/history.csv")
RESULTADOS = (RAIZ / "results/anchor/results.csv", RAIZ / "results/rules/results.csv")
SALIDA = RAIZ / "results/rules/contraste-dirigido.csv"
SALIDA_ESCALAS = RAIZ / "results/rules/poda-por-escala.csv"

# Columnas de cada instancia de calibración, para la razón de poda por dimensión.
COLUMNAS = {
    "scp41": 1000, "scp61": 1000, "scp51": 2000, "scpa1": 3000, "scpb1": 3000,
    "scpc1": 4000, "scpd1": 4000, "scpnre1": 5000, "scpnrf1": 5000,
    "scpnrg1": 10000, "scpnrh1": 10000,
}

# El orden es el de la Tabla 7 y el rótulo es el que aparece en el informe.
ROTULOS = {
    "BPWO_BASE": "Referencia congelada",
    "BPWO_RALLY": "Cohesión corregida",
    "BPWO_ANCHOR": "Anclaje kappa=4, S1-STD",
    "ALPHA_S1": "Control sin ecuaciones PWO",
    "BPWO_S2_STD": "S2-STD, kappa=1",
    "BPWO_V3_COMP": "V3-COMP, kappa=1",
}

# Agentes por iteración: cada uno consume una evaluación del ciclo.
POBLACION = 10


def _booleano(valor: str) -> float:
    if valor == "True":
        return 1.0
    if valor == "False":
        return 0.0
    return float(valor)


def leer_trayectorias() -> dict[str, dict[str, list[float]]]:
    datos: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ruta in TRAYECTORIAS:
        if not ruta.exists():
            raise SystemExit(
                f"Falta {ruta}. Las trayectorias no se versionan; hay que "
                "regenerarlas ejecutando el experimento antes de derivar la tabla."
            )
        with ruta.open(encoding="utf-8", newline="") as archivo:
            for fila in csv.DictReader(archivo):
                serie = datos[fila["algorithm"]]
                serie["explotacion"].append(_booleano(fila["exploitation"]))
                serie["factibilidad_previa"].append(float(fila["initially_feasible_rate"]))
                serie["agregadas"].append(float(fila["added_columns"]) / POBLACION)
                serie["eliminadas"].append(float(fila["removed_columns"]) / POBLACION)
                serie["diversidad"].append(float(fila["binary_diversity"]))
    return datos


def derivar_por_escala() -> list[dict[str, object]]:
    """Poda por evaluación del ciclo en cada instancia, y su razón por dimensión."""

    podas: dict[tuple[str, str], list[float]] = defaultdict(list)
    for ruta in TRAYECTORIAS:
        with ruta.open(encoding="utf-8", newline="") as archivo:
            for fila in csv.DictReader(archivo):
                clave = (fila["algorithm"], fila["instance"])
                podas[clave].append(float(fila["removed_columns"]) / POBLACION)
    filas = []
    for (algoritmo, instancia), valores in sorted(podas.items()):
        n = COLUMNAS[instancia]
        media = statistics.fmean(valores)
        filas.append(
            {
                "algoritmo": algoritmo,
                "instancia": instancia,
                "columnas": n,
                "podadas_por_evaluacion": round(media, 2),
                "razon_por_dimension": round(media / n, 6),
            }
        )
    return filas


def leer_resultados() -> dict[str, dict[str, list[float]]]:
    datos: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for ruta in RESULTADOS:
        with ruta.open(encoding="utf-8", newline="") as archivo:
            for fila in csv.DictReader(archivo):
                serie = datos[fila["algorithm"]]
                serie["rpd"].append(float(fila["rpd"]))
                serie["aciertos"].append(1.0 if float(fila["rpd"]) == 0 else 0.0)
    return datos


def derivar() -> list[dict[str, object]]:
    trayectorias = leer_trayectorias()
    resultados = leer_resultados()
    filas = []
    for algoritmo, rotulo in ROTULOS.items():
        t, r = trayectorias[algoritmo], resultados[algoritmo]
        filas.append(
            {
                "algoritmo": algoritmo,
                "rotulo": rotulo,
                "corridas": len(r["rpd"]),
                "rpd_medio": round(statistics.fmean(r["rpd"]), 4),
                "aciertos": int(sum(r["aciertos"])),
                "explotacion": round(statistics.fmean(t["explotacion"]), 4),
                "factibilidad_previa": round(statistics.fmean(t["factibilidad_previa"]), 4),
                "agregadas_por_evaluacion": round(statistics.fmean(t["agregadas"]), 2),
                "eliminadas_por_evaluacion": round(statistics.fmean(t["eliminadas"]), 2),
                "diversidad_mediana": round(statistics.median(t["diversidad"]), 4),
            }
        )
    return filas


def main() -> None:
    filas = derivar()
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)
    ancho = max(len(f["rotulo"]) for f in filas)
    print(f"{'variante':{ancho}}  {'RPD':>8}  {'acier':>6}  {'expl':>7}  "
          f"{'previa':>7}  {'+col':>8}  {'-col':>9}  {'divers':>7}")
    for f in filas:
        print(f"{f['rotulo']:{ancho}}  {f['rpd_medio']:8.4f}  "
              f"{f['aciertos']:>3}/{f['corridas']:<2}  {f['explotacion']:7.4f}  "
              f"{f['factibilidad_previa']:7.4f}  {f['agregadas_por_evaluacion']:8.2f}  "
              f"{f['eliminadas_por_evaluacion']:9.2f}  {f['diversidad_mediana']:7.4f}")
    escalas = derivar_por_escala()
    with SALIDA_ESCALAS.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(escalas[0]))
        escritor.writeheader()
        escritor.writerows(escalas)
    print(f"\nEscrito: {SALIDA.relative_to(RAIZ)}")
    print(f"Escrito: {SALIDA_ESCALAS.relative_to(RAIZ)} ({len(escalas)} filas)")


if __name__ == "__main__":
    main()
