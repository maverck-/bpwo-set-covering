# BPWO para Set Covering

Código, datos derivados y pruebas del informe *Diseño, evaluación y diagnóstico
de una variante binaria de Painted Wolf Optimization para Set Covering*.

Maverick Gayoso y Rogelio González. MII902 Optimización Estocástica, Magíster en
Ingeniería Informática, Pontificia Universidad Católica de Valparaíso, 2026.

BPWO toma Painted Wolf Optimization, una metaheurística poblacional continua, y
la adapta al Set Covering Problem mediante un estado latente sincronizado, tres
ecuaciones de movimiento vectoriales, una binarización de dos pasos y una
reparación constructiva. El aporte del trabajo no es una mejora de costo sino el
diagnóstico de por qué el movimiento continuo no se transmite a los bits.

## Algoritmo de origen

Painted Wolf Optimization fue propuesto por Saeid Sheikhi. Este trabajo lo toma
como inspiración y no como objeto de estudio: BPWO conserva sus referencias
funcionales, el rally que selecciona el modo de búsqueda y las tres ramas de
movimiento, pero la forma exacta de las ecuaciones implementadas aquí pertenece
a BPWO y se declara por completo en el informe.

- S. Sheikhi, "Painted wolf optimization: a novel nature-inspired metaheuristic
  algorithm for real-world optimization problems", *Computers, Materials &
  Continua*, vol. 87, n.º 2, 2026.
  [doi:10.32604/cmc.2026.077788](https://doi.org/10.32604/cmc.2026.077788)
- S. Sheikhi, *Painted-Wolf-Optimization*, repositorio oficial.
  [github.com/saeidsheikhi/Painted-Wolf-Optimization](https://github.com/saeidsheikhi/Painted-Wolf-Optimization)

El protocolo experimental, la binarización de dos pasos y la elección de
comparadores siguen el antecedente más próximo en el dominio binario:

- B. Crawford, F. Cisternas-Caneo, R. Soto *et al.*, "Binary Secretary Bird
  Optimization Algorithm for the Set Covering Problem", *Mathematics*, vol. 13,
  n.º 15, art. 2482, 2025.
  [doi:10.3390/math13152482](https://doi.org/10.3390/math13152482)

## Qué contiene

```text
src/bpwo/        algoritmo, comparadores, binarización, reparación e inferencia
scripts/         derivaciones que producen las tablas del informe
tests/           39 pruebas sobre lectura de instancias, binarización,
                 reparación, factibilidad, reproducibilidad y presupuesto
results/         resultados agregados y CSV derivados que sostienen las tablas
instancias/      manifiesto de las 22 instancias con su rol y su referencia
```

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[analysis]"
.venv/bin/python -m pytest tests/ -q
```

Requiere Python 3.11 o superior. El experimento se ejecutó con Python 3.13.15 y
NumPy 2.5.1 en macOS arm64; el análisis estadístico usa SciPy 1.18.1.

## Instancias

Las instancias son las de la OR-Library de Beasley y no se redistribuyen aquí.
`instancias/manifiesto.csv` declara las 22 usadas, su partición entre
calibración y test, su escala y si la referencia es el óptimo o el mejor valor
conocido. Los archivos `scpNN.txt` se descargan de la OR-Library y se dejan en
un directorio que después se pasa como `--instances-root`.

## Reproducir el experimento principal

Once instancias de test, 31 corridas por configuración y 6000 evaluaciones por
corrida:

```bash
.venv/bin/python -m bpwo.final_experiment \
    --manifest instancias/manifiesto.csv \
    --instances-root <ruta a las instancias> \
    --role test \
    --checkpoint-dir results/final/checkpoints \
    --output results/final/results.csv \
    --history-output results/final/history.csv
```

Cada corrida usa su propia semilla y su propio generador, de modo que el
resultado es determinista y no depende del orden de ejecución ni del número de
procesos.

## Derivaciones

Las trayectorias completas, `history.csv` y los puntos de control, no se
versionan por peso. Los tres scripts son el puente verificable entre esas
trayectorias y las cifras publicadas, y abortan con un mensaje explícito si
falta su insumo.

| Script | Produce | Sostiene |
|:---|:---|:---|
| `derivar_contraste_dirigido.py` | `results/rules/contraste-dirigido.csv` y `poda-por-escala.csv` | Tabla 7, contraste dirigido sobre calibración |
| `derivar_proxy_conservacion.py` | `results/conservacion/proxy-por-instancia.csv` y `proxy-por-corrida.csv` | El proxy de conservación y su emparejamiento con la medición |
| `medir_conservacion.py` | `results/conservacion/solapamiento-medido.csv` | Tabla 6, conservación medida sobre 165 corridas |

`medir_conservacion.py` envuelve la binarización para observar cuántas columnas
activas del incumbente sobreviven antes de reparar. Llama a la implementación
original y solo lee sus argumentos y su retorno, de modo que no consume el
generador y las corridas reproducen exactamente las del experimento sin
instrumentar. `results/conservacion/README.md` documenta ese contraste, sus
límites y las dos convenciones de denominador del proxy.

## Qué no está aquí

Este repositorio acompaña al informe y contiene lo necesario para verificarlo y
reproducirlo. No incluye las fuentes de composición del documento, las figuras,
los materiales de referencia ni la documentación interna del proyecto.

Dos pruebas adicionales viven junto al manuscrito y no aquí, porque comparan las
tablas publicadas celda a celda contra los CSV derivados y necesitan el texto
del informe para funcionar.
