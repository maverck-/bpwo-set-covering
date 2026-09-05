#!/usr/bin/env python3

"""Extiende a las once instancias de test el proxy de conservación de la Tabla 6.

El informe estima $\\widehat C_{rep} = 1 - (\\bar A - \\bar E)/|x_g|$ sobre `scp42`
y lo compara con el valor analítico de la transferencia evaluada en el estado de
cada configuración. Este script repite ese cálculo en las once instancias y
emite las dos convenciones de denominador que conviven en 5.6, porque no dan lo
mismo:

* `ciclo`: promedia sobre las 5990 evaluaciones del ciclo, excluyendo las diez
  de inicialización. Es la convención que enuncia la definición de 5.6 y la que
  usan sus cifras en prosa, 19.51 columnas agregadas y 2.58 eliminadas.
* `total6000`: divide el total de la corrida por el presupuesto completo. Es la
  que produjo las cifras publicadas en la Tabla 6.

La inicialización repara una población aleatoria densa cuya poda escala con el
número de columnas, de modo que repartirla entre 6000 evaluaciones contamina el
promedio, y contamina más en instancias grandes. En `scp42` la diferencia queda
en 0.012 y pasa inadvertida.

Entrada: `results/final/results.csv`, versionado, y `results/final/history.csv`,
que no lo está por peso y se regenera con el experimento final.

Uso:

    .venv/bin/python scripts/derivar_proxy_conservacion.py
"""

from __future__ import annotations

import csv
import statistics
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
RESULTADOS = RAIZ / "results/final/results.csv"
TRAYECTORIA = RAIZ / "results/final/history.csv"
SALIDA = RAIZ / "results/conservacion/proxy-por-instancia.csv"
SALIDA_CORRIDA = RAIZ / "results/conservacion/proxy-por-corrida.csv"
MEDIDO = RAIZ / "results/conservacion/solapamiento-medido.csv"

# La medición directa instrumenta solo estas semillas, y las corridas son
# deterministas, de modo que el proxy admite emparejarse con ella corrida a
# corrida en lugar de compararse contra el agregado de las 31.
SEMILLAS_MEDIDAS = ("0", "1", "2")

# Agentes por iteración: cada fila de la trayectoria agrega sus diez evaluaciones.
POBLACION = 10
PRESUPUESTO = 6000

# Columnas de cada instancia de test, para acompañar la lectura por escala.
COLUMNAS = {
    "scp42": 1000, "scp62": 1000, "scp52": 2000, "scpa2": 3000, "scpb2": 3000,
    "scpc2": 4000, "scpd2": 4000, "scpnre2": 5000, "scpnrf2": 5000,
    "scpnrg2": 10000, "scpnrh2": 10000,
}

# Transferencia evaluada donde vive la variable de estado de cada configuración.
# V3(±1)=1/raíz(2) para las posiciones sincronizadas, S2(+1) bajo S2, la media de
# V3 sobre el muestreo uniforme de IID es raíz(2)-1, y la velocidad de BPSO tiende
# a cero sin alcanzarlo, de modo que su valor es una cota y no una predicción.
CONFIGURACIONES = (
    ("BPSO", "V3-ELIT", "BPSO", "velocidad", 0.0),
    ("ABLATION_IID", "V3-ELIT", "IID", "muestreo uniforme", 0.4142),
    ("BGWO", "V3-ELIT", "BGWO", "posición sincronizada", 0.7071),
    ("BPWO", "V3-ELIT", "BPWO-V3", "posición sincronizada", 0.7071),
    ("BPWO", "S2-ELIT", "BPWO-S2", "posición sincronizada", 0.7311),
)


def leer_greedy() -> dict[str, int]:
    """Tamaño de la solución greedy de cada instancia, agregadas menos podadas."""

    tamanos: dict[str, int] = {}
    with RESULTADOS.open(encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            if fila["algorithm"] == "GREEDY":
                tamanos[fila["instance"]] = (
                    int(fila["added_columns"]) - int(fila["removed_columns"])
                )
    return tamanos


def leer_totales() -> dict[tuple[tuple[str, str], str], dict[str, list[float]]]:
    datos: dict[tuple[tuple[str, str], str], dict[str, list[float]]] = defaultdict(
        lambda: {"a": [], "e": []}
    )
    with RESULTADOS.open(encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            if fila["algorithm"] == "GREEDY":
                continue
            clave = ((fila["algorithm"], fila["scheme"]), fila["instance"])
            datos[clave]["a"].append(float(fila["added_columns"]))
            datos[clave]["e"].append(float(fila["removed_columns"]))
    return datos


def leer_ciclo_por_semilla() -> dict[tuple[tuple[str, str], str, str], dict[str, list[float]]]:
    """Igual que `leer_ciclo`, pero sin agregar sobre las semillas medidas."""

    if not TRAYECTORIA.exists():
        raise SystemExit(f"Falta {TRAYECTORIA}.")
    datos: dict[tuple[tuple[str, str], str, str], dict[str, list[float]]] = defaultdict(
        lambda: {"a": [], "e": []}
    )
    with TRAYECTORIA.open(encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            if fila["seed"] not in SEMILLAS_MEDIDAS:
                continue
            clave = ((fila["algorithm"], fila["scheme"]), fila["instance"], fila["seed"])
            datos[clave]["a"].append(float(fila["added_columns"]) / POBLACION)
            datos[clave]["e"].append(float(fila["removed_columns"]) / POBLACION)
    return datos


def derivar_por_corrida() -> list[dict[str, object]]:
    """Empareja el proxy con la medición directa, corrida a corrida.

    El informe compara ambos indicadores sobre 165 corridas. `leer_ciclo` agrega
    sobre las 31 semillas del test y no permite ese emparejamiento, así que la
    comparación se deriva aquí desde la trayectoria filtrada por semilla.
    """

    greedy = leer_greedy()
    ciclo = leer_ciclo_por_semilla()
    with MEDIDO.open(encoding="utf-8", newline="") as archivo:
        medido = {
            (f["configuracion"], f["instancia"], f["semilla"]): f
            for f in csv.DictReader(archivo)
        }
    identificador = {rotulo: (alg, esq) for alg, esq, rotulo, _, _ in CONFIGURACIONES}

    filas = []
    for (rotulo, instancia, semilla), fila in sorted(medido.items()):
        serie = ciclo[(identificador[rotulo], instancia, semilla)]
        neta = statistics.fmean(serie["a"]) - statistics.fmean(serie["e"])
        proxy = 1 - neta / greedy[instancia]
        real = float(fila["real"])
        filas.append(
            {
                "configuracion": rotulo,
                "instancia": instancia,
                "semilla": int(semilla),
                "neta_ciclo": round(neta, 3),
                "x_greedy": greedy[instancia],
                "proxy_ciclo": round(proxy, 4),
                "solapamiento_real": round(real, 4),
                "diferencia": round(proxy - real, 4),
            }
        )
    return filas


def leer_ciclo() -> dict[tuple[tuple[str, str], str], dict[str, list[float]]]:
    if not TRAYECTORIA.exists():
        raise SystemExit(
            f"Falta {TRAYECTORIA}. La trayectoria no se versiona por peso; hay "
            "que regenerarla ejecutando el experimento final antes de derivar "
            "el proxy por evaluación del ciclo."
        )
    datos: dict[tuple[tuple[str, str], str], dict[str, list[float]]] = defaultdict(
        lambda: {"a": [], "e": []}
    )
    with TRAYECTORIA.open(encoding="utf-8", newline="") as archivo:
        for fila in csv.DictReader(archivo):
            clave = ((fila["algorithm"], fila["scheme"]), fila["instance"])
            datos[clave]["a"].append(float(fila["added_columns"]) / POBLACION)
            datos[clave]["e"].append(float(fila["removed_columns"]) / POBLACION)
    return datos


def derivar() -> list[dict[str, object]]:
    greedy = leer_greedy()
    totales = leer_totales()
    ciclo = leer_ciclo()
    instancias = sorted(COLUMNAS, key=lambda i: (COLUMNAS[i], i))

    filas: list[dict[str, object]] = []
    for algoritmo, esquema, rotulo, estado, analitico in CONFIGURACIONES:
        for instancia in instancias:
            xg = greedy[instancia]
            c = ciclo[((algoritmo, esquema), instancia)]
            t = totales[((algoritmo, esquema), instancia)]
            neta_ciclo = statistics.fmean(c["a"]) - statistics.fmean(c["e"])
            neta_total = (
                statistics.fmean(t["a"]) - statistics.fmean(t["e"])
            ) / PRESUPUESTO
            proxy_ciclo = 1 - neta_ciclo / xg
            filas.append(
                {
                    "configuracion": rotulo,
                    "estado": estado,
                    "instancia": instancia,
                    "columnas": COLUMNAS[instancia],
                    "x_greedy": xg,
                    "agregadas_ciclo": round(statistics.fmean(c["a"]), 2),
                    "eliminadas_ciclo": round(statistics.fmean(c["e"]), 2),
                    "neta_ciclo": round(neta_ciclo, 2),
                    "proxy_ciclo": round(proxy_ciclo, 4),
                    "neta_total6000": round(neta_total, 2),
                    "proxy_total6000": round(1 - neta_total / xg, 4),
                    "analitico": analitico,
                    "error_ciclo": round(proxy_ciclo - analitico, 4),
                }
            )
    return filas


def _pearson(x: list[float], y: list[float]) -> float:
    mx, my = statistics.fmean(x), statistics.fmean(y)
    numerador = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denominador = (
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    ) ** 0.5
    return numerador / denominador


def _spearman(x: list[float], y: list[float]) -> float:
    """Pearson sobre los rangos, con rango promedio en los empates.

    Los valores vienen redondeados a cuatro decimales y repiten bastante, de modo
    que asignar rangos por orden de aparición sesgaría el resultado.
    """

    def rangos(valores: list[float]) -> list[float]:
        orden = sorted(range(len(valores)), key=lambda i: valores[i])
        salida = [0.0] * len(valores)
        inicio = 0
        while inicio < len(orden):
            fin = inicio
            while fin + 1 < len(orden) and valores[orden[fin + 1]] == valores[orden[inicio]]:
                fin += 1
            promedio = (inicio + fin) / 2
            for posicion in range(inicio, fin + 1):
                salida[orden[posicion]] = promedio
            inicio = fin + 1
        return salida

    return _pearson(rangos(x), rangos(y))


def main() -> None:
    filas = derivar()
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    with SALIDA.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0]))
        escritor.writeheader()
        escritor.writerows(filas)

    print("Proxy de conservación por instancia, las dos convenciones.\n")
    print(f"{'configuración':10s} {'analít':>7s} {'ciclo':>17s} {'total/6000':>17s}")
    print(f"{'':10s} {'':7s} {'media':>8s} {'desv':>8s} {'media':>8s} {'desv':>8s}")
    for _, _, rotulo, _, analitico in CONFIGURACIONES:
        sub = [f for f in filas if f["configuracion"] == rotulo]
        pc = [float(f["proxy_ciclo"]) for f in sub]
        pt = [float(f["proxy_total6000"]) for f in sub]
        print(
            f"{rotulo:10s} {analitico:7.4f} {statistics.fmean(pc):8.4f} "
            f"{statistics.pstdev(pc):8.4f} {statistics.fmean(pt):8.4f} "
            f"{statistics.pstdev(pt):8.4f}"
        )
    orden = [f[2] for f in CONFIGURACIONES]
    coinciden = sum(
        1
        for i in sorted(COLUMNAS)
        if [
            float(next(f for f in filas if f["configuracion"] == r and f["instancia"] == i)["proxy_ciclo"])
            for r in orden
        ]
        == sorted(
            float(next(f for f in filas if f["configuracion"] == r and f["instancia"] == i)["proxy_ciclo"])
            for r in orden
        )
    )
    print(f"\nInstancias donde el orden {' < '.join(orden)} se cumple: {coinciden}/11")
    print(f"Escrito: {SALIDA.relative_to(RAIZ)} ({len(filas)} filas)")

    corridas = derivar_por_corrida()
    with SALIDA_CORRIDA.open("w", encoding="utf-8", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(corridas[0]))
        escritor.writeheader()
        escritor.writerows(corridas)
    proxy = [float(f["proxy_ciclo"]) for f in corridas]
    real = [float(f["solapamiento_real"]) for f in corridas]
    diferencia = [float(f["diferencia"]) for f in corridas]
    print(
        f"\nEmparejado con la medición directa en {len(corridas)} corridas: "
        f"r={_pearson(real, proxy):.4f}  rho={_spearman(real, proxy):.4f}  "
        f"MAE={statistics.fmean(abs(d) for d in diferencia):.4f}  "
        f"sesgo={statistics.fmean(diferencia):+.4f}  "
        f"máx={max(abs(d) for d in diferencia):.4f}"
    )
    print(f"Escrito: {SALIDA_CORRIDA.relative_to(RAIZ)} ({len(corridas)} filas)")


if __name__ == "__main__":
    main()
