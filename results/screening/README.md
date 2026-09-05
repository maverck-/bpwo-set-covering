# Resultados de calibración

Esta carpeta conserva salidas crudas y resúmenes reproducibles. Ningún archivo
corresponde al conjunto de test ni al experimento final.

## Series

| Prefijo | Propósito | Semillas | Evaluaciones |
|---|---|---:|---:|
| `pwo-arquitectura` | BPWO, IID, NBPWO, BPSO y BGWO con V3-ELIT | 0, 1, 2 | 300 |
| `pwo-binarizacion` | S2-STD, V3-COMP, S2-ELIT y V3-ELIT | 0, 1, 2 | 300 |
| `pwo-calibracion-confirmatoria` | S2-ELIT y V3-ELIT con PWO e IID | 0 a 4 | 300 |
| `pwo-diversity-restart` | BPWO y BPWO-DR con S2-ELIT | 0 a 4 | 300 |
| `pwo-budget-1000` | Medición de costo en `scpnrg1` | 0 | 1000 |
| `pwo-calibration-extension` | Confirmación S2-ELIT frente a V3-ELIT en siete familias adicionales | 0 a 4 | 300 |
| `pwo-budget-6000` | Suite final completa en `scpnrg1` para medir costo | 0 | 6000 |
| `pwo-calibration-6000` | Selección definitiva S2-ELIT frente a V3-ELIT en once instancias | 0 a 4 | 6000 |

Los archivos `*-history.csv` contienen las trayectorias por iteración;
`*-summary*.csv`, los estadísticos descriptivos; `*-ranking*.csv`, los rangos
por mediana de RPD, y `*-mechanism.csv`, las métricas internas del mecanismo.

## Alcance interpretativo

Las series usan instancias de calibración y pocas semillas. Sirven para elegir
arquitectura, binarización y presupuesto. No sustentan afirmaciones de
superioridad ni reemplazan las 31 corridas del experimento final.
