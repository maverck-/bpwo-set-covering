# Reproducción de los resultados

Este documento relaciona el código público con los experimentos y resultados
del informe. Los comandos se ejecutan desde la raíz del repositorio.

## Preparación

1. Instale el entorno descrito en `README.md`.
2. Descargue las instancias SCP y defina su directorio:

   ```bash
   INSTANCES_ROOT=/ruta/a/SCP/Instances
   ```

3. Ejecute las pruebas:

   ```bash
   .venv/bin/python -m unittest discover -s tests
   ```

Las trayectorias y los puntos de control no se versionan. Los comandos que los
generan son reanudables y escriben en directorios ignorados por Git. Los CSV
agregados que sostienen el informe sí están versionados.

## Correspondencia con el informe

| Parte del informe | Ejecución | Resultado versionado |
|:---|:---|:---|
| Experimento principal, secciones 5.1 a 5.5 | `bpwo.final_experiment` | `results/final/` |
| Conservación, sección 5.6 y Tabla 6 | `scripts/medir_conservacion.py` | `results/conservacion/solapamiento-medido.csv` |
| Proxy auxiliar de conservación | `scripts/derivar_proxy_conservacion.py` | `results/conservacion/proxy-por-*.csv` |
| Contraste dirigido, sección 5.7 y Tabla 7 | `bpwo.anchor_experiment` y `scripts/derivar_contraste_dirigido.py` | `results/anchor/` y `results/rules/` |
| Exploración posterior, sección 5.8 | `bpwo.improvement_experiment` | `results/improvement/` |
| Actualización del estado, sección 5.9 | `bpwo.state_update_experiment` | `results/state-update-*/` |

## Experimento principal

El comando usa las once instancias de test, las semillas 0 a 30, población 10
y 6000 evaluaciones por corrida. Esos son los valores predeterminados del
ejecutor.

```bash
.venv/bin/python -m bpwo.final_experiment \
    --manifest instancias/manifiesto.csv \
    --instances-root "$INSTANCES_ROOT" \
    --role test \
    --checkpoint-dir results/final/checkpoints \
    --output results/final/results.csv \
    --history-output results/final/history.csv
```

Los resúmenes descriptivos, las métricas de mecanismo y la inferencia se
reconstruyen así:

```bash
.venv/bin/python -m bpwo.analysis \
    --input results/final/results.csv \
    --summary-output results/final/summary.csv \
    --ranking-output results/final/ranking.csv \
    --history-input results/final/history.csv \
    --mechanism-output results/final/mechanism.csv

.venv/bin/python -m bpwo.inference \
    --input results/final/results.csv \
    --output results/final/inference.csv
```

## Conservación y Tabla 6

La medición directa repite tres semillas del experimento principal sin cambiar
el consumo del generador aleatorio:

```bash
.venv/bin/python scripts/medir_conservacion.py \
    --instancias-raiz "$INSTANCES_ROOT"
```

La ejecución completa toma alrededor de dos horas en el equipo utilizado. Para
una comprobación corta:

```bash
.venv/bin/python scripts/medir_conservacion.py \
    --instancias-raiz "$INSTANCES_ROOT" \
    --instancias scp62 \
    --semillas 1 \
    --evaluaciones 600
```

El proxy auxiliar requiere `results/final/history.csv`, generado por el
experimento principal:

```bash
.venv/bin/python scripts/derivar_proxy_conservacion.py
```

## Contraste dirigido y Tabla 7

Las cuatro configuraciones principales del contraste se generan con:

```bash
.venv/bin/python -m bpwo.anchor_experiment \
    --manifest instancias/manifiesto.csv \
    --instances-root "$INSTANCES_ROOT" \
    --role calibration \
    --variants BPWO_BASE BPWO_RALLY BPWO_ANCHOR ALPHA_S1 \
    --checkpoint-dir results/anchor/checkpoints \
    --output results/anchor/results.csv \
    --history-output results/anchor/history.csv
```

Las dos reglas densas se ejecutan por separado para conservar la separación
experimental del informe:

```bash
.venv/bin/python -m bpwo.anchor_experiment \
    --manifest instancias/manifiesto.csv \
    --instances-root "$INSTANCES_ROOT" \
    --role calibration \
    --variants BPWO_S2_STD BPWO_V3_COMP \
    --checkpoint-dir results/rules/checkpoints \
    --output results/rules/results.csv \
    --history-output results/rules/history.csv
```

Con ambas trayectorias disponibles, la Tabla 7 y la poda por escala se derivan
con:

```bash
.venv/bin/python scripts/derivar_contraste_dirigido.py
```

La réplica con 30 agentes y 15000 evaluaciones usa:

```bash
.venv/bin/python -m bpwo.anchor_experiment \
    --manifest instancias/manifiesto.csv \
    --instances-root "$INSTANCES_ROOT" \
    --role calibration \
    --seeds 0 1 2 3 4 \
    --population 30 \
    --evaluations 15000 \
    --variants BPWO_BASE BPWO_ANCHOR \
    --checkpoint-dir results/anchor-n30/checkpoints \
    --output results/anchor-n30/results.csv \
    --history-output results/anchor-n30/history.csv
```

## Exploración posterior de la sección 5.8

Esta etapa utiliza solo las once instancias con rol `development` del segundo
manifiesto. Las particiones `validation` y `holdout` se conservan sin ejecutar.

```bash
.venv/bin/python -m bpwo.improvement_experiment \
    --manifest instancias/manifiesto-desarrollo.csv \
    --instances-root "$INSTANCES_ROOT" \
    --role development \
    --seeds 0 1 2 \
    --evaluations 300 \
    --checkpoint-dir results/improvement/checkpoints/screening-300 \
    --output results/improvement/screening-300.csv \
    --history-output results/improvement/screening-300-history.csv

.venv/bin/python -m bpwo.improvement_experiment \
    --manifest instancias/manifiesto-desarrollo.csv \
    --instances-root "$INSTANCES_ROOT" \
    --role development \
    --seeds 0 1 2 3 4 \
    --evaluations 6000 \
    --variants BPWO_BASE BPWO_PROB \
    --checkpoint-dir results/improvement/checkpoints/confirmation-6000 \
    --output results/improvement/confirmation-6000.csv \
    --history-output results/improvement/confirmation-6000-history.csv
```

Los módulos `bpwo.analysis` y `bpwo.inference` se aplican a esos CSV con el
mismo esquema mostrado para el experimento principal. En la inferencia, la
referencia es `BPWO_BASE|PWO|V3-ELIT`.

## Actualización del estado de la sección 5.9

El cribado inicial y su control adicional usan tres semillas y 300
evaluaciones:

```bash
.venv/bin/python -m bpwo.state_update_experiment \
    --manifest instancias/manifiesto-desarrollo.csv \
    --instances-root "$INSTANCES_ROOT" \
    --role development \
    --seeds 0 1 2 \
    --evaluations 300 \
    --variants ANCHOR_GREEDY_SYNC ANCHOR_ALWAYS_SYNC \
               PWO_ALWAYS_FEEDBACK ALPHA_ALWAYS_FEEDBACK \
    --checkpoint-dir results/state-update-screening/checkpoints \
    --output results/state-update-screening/results.csv \
    --history-output results/state-update-screening/history.csv

.venv/bin/python -m bpwo.state_update_experiment \
    --manifest instancias/manifiesto-desarrollo.csv \
    --instances-root "$INSTANCES_ROOT" \
    --role development \
    --seeds 0 1 2 \
    --evaluations 300 \
    --variants ALPHA_ALWAYS_SYNC \
    --checkpoint-dir results/state-update-control-screening/checkpoints \
    --output results/state-update-control-screening/results.csv \
    --history-output results/state-update-control-screening/history.csv
```

La confirmación usa cinco semillas y 6000 evaluaciones:

```bash
.venv/bin/python -m bpwo.state_update_experiment \
    --manifest instancias/manifiesto-desarrollo.csv \
    --instances-root "$INSTANCES_ROOT" \
    --role development \
    --seeds 0 1 2 3 4 \
    --evaluations 6000 \
    --variants ANCHOR_GREEDY_SYNC ANCHOR_ALWAYS_SYNC ALPHA_ALWAYS_SYNC \
    --checkpoint-dir results/state-update-confirmation/checkpoints \
    --output results/state-update-confirmation/results.csv \
    --history-output results/state-update-confirmation/history.csv
```

## Comprobación de salidas

Cada CSV de resultados registra las versiones de Python y NumPy utilizadas.
Los tiempos dependen del equipo y no se espera que coincidan. Los costos, RPD,
semillas, presupuestos y métricas derivadas sí deben reproducir las salidas
versionadas. Después de una ejecución puede revisar las diferencias con:

```bash
git diff -- results/
```
