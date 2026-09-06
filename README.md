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
instancias/      manifiestos de las instancias y sus particiones experimentales
```

[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) relaciona cada experimento del
informe con su comando, sus insumos y sus salidas versionadas.

## Instalación

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[analysis]"
.venv/bin/python -m unittest discover -s tests
```

Las pruebas usan la biblioteca estándar `unittest` y no requieren `pytest`.
El código admite Python 3.11 o superior. Para reconstruir el entorno del
experimento principal con las versiones registradas en sus CSV:

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements-reproduction.txt
.venv/bin/pip install -e . --no-deps
```

El experimento principal se ejecutó con Python 3.13.15, NumPy 2.5.1 y SciPy
1.18.1 en macOS arm64. Algunos experimentos de desarrollo registran otras
versiones en sus propias filas. `uv.lock` conserva el entorno de verificación
actual, no reemplaza ese registro histórico.

## Instancias

Las instancias son las de la OR-Library de Beasley y no se redistribuyen aquí.
También están disponibles en el
[repositorio OII-450-1-2024](https://github.com/FelipeCisternasCaneo/OII-450-1-2024/tree/master/src/problem/SCP/Instances)
usado por el proyecto. `instancias/manifiesto.csv` declara las 22 instancias de
calibración y test. `instancias/manifiesto-desarrollo.csv` registra además las
particiones de desarrollo, validación y reserva descritas en el informe. Ambos
manifiestos incluyen la escala y distinguen entre óptimo y mejor valor
conocido. Los archivos `scpNN.txt` se dejan en un directorio que después se
pasa como `--instances-root`:

```bash
INSTANCES_ROOT=/ruta/a/SCP/Instances
```

## Reproducir el experimento principal

Once instancias de test, 31 corridas por configuración y 6000 evaluaciones por
corrida:

```bash
.venv/bin/python -m bpwo.final_experiment \
    --manifest instancias/manifiesto.csv \
    --instances-root "$INSTANCES_ROOT" \
    --role test \
    --checkpoint-dir results/final/checkpoints \
    --output results/final/results.csv \
    --history-output results/final/history.csv
```

Cada corrida usa su propia semilla y su propio generador, de modo que el
resultado es determinista y no depende del orden de ejecución ni del número de
procesos.

El resto de los comandos, incluidos los que reconstruyen las Tablas 6 y 7 y
los experimentos de diagnóstico, está en
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).

## Derivaciones

Las trayectorias completas, `history.csv` y los puntos de control, no se
versionan por peso. Los tres scripts son el puente verificable entre esas
trayectorias y las cifras publicadas, y abortan con un mensaje explícito si
falta su insumo.

| Script | Produce | Sostiene |
|:---|:---|:---|
| `derivar_contraste_dirigido.py` | `results/rules/contraste-dirigido.csv` y `poda-por-escala.csv` | Tabla 7, contraste dirigido sobre calibración |
| `derivar_proxy_conservacion.py` | `results/conservacion/proxy-por-instancia.csv` y `proxy-por-corrida.csv` | Aproximación diagnóstica de la conservación y su emparejamiento con la medición |
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

Dos archivos de pruebas adicionales viven junto al manuscrito y no aquí, porque
comparan las tablas publicadas celda a celda contra los CSV derivados y
necesitan el texto del informe para funcionar.

## Cómo citar

Para citar el informe:

> M. Gayoso y R. González, “Diseño, evaluación y diagnóstico de una variante
> binaria de Painted Wolf Optimization para Set Covering”, informe de curso,
> MII902 Optimización Estocástica, Magíster en Ingeniería Informática,
> Pontificia Universidad Católica de Valparaíso, 2026.

Para citar el código y los datos derivados:

> M. Gayoso y R. González, *BPWO para Set Covering: código, datos derivados y
> pruebas*, versión 0.1.0, 2026. Disponible en:
> <https://github.com/maverck-/bpwo-set-covering>.

GitHub también puede generar la cita del software a partir de
[`CITATION.cff`](CITATION.cff).

## Licencia

El código y la documentación propios de este repositorio se distribuyen bajo la
[licencia MIT](LICENSE). Los CSV derivados contenidos en `results/` se
distribuyen bajo [Creative Commons Atribución 4.0 Internacional
(CC BY 4.0)](LICENSE-DATA.md). Las instancias externas no forman parte del
repositorio y conservan las condiciones establecidas por sus fuentes. Estas
licencias permiten reutilizar los materiales, mientras que la referencia
académica se especifica por separado en `CITATION.cff`.
