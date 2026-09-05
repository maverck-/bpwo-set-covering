# Experimento final

El protocolo congelado utiliza las once instancias con sufijo `2`, semillas 0
a 30, población 10 y 6000 evaluaciones por corrida.

Suite estocástica:

1. BPWO-V3-ELIT, propuesta principal;
2. IID-V3-ELIT, ablación del movimiento PWO;
3. BPWO-S2-ELIT, ablación de binarización;
4. BPSO-V3-ELIT;
5. BGWO-V3-ELIT.

Greedy se ejecuta una vez por instancia. Los checkpoints y `history.csv` se
conservan localmente, pero están excluidos de Git por su tamaño. El consolidado
`results.csv`, los resúmenes y el análisis inferencial sí pueden versionarse.

No se debe cambiar la configuración después de iniciar estas corridas. Una
ejecución interrumpida se reanuda con el mismo comando y reutiliza los
checkpoints completos.
