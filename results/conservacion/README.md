# Conservación: medición directa y proxy auxiliar

Material de respaldo de la Tabla 6 y de la sección 5.6 del informe. La medición
directa es la evidencia principal publicada. El proxy se conserva como una
aproximación diagnóstica para resultados que no registren el solapamiento bit a
bit.

La sección 5.6 define la probabilidad de conservación como la probabilidad de
que una columna activa del incumbente siga activa en la propuesta binaria antes
de reparar. Bajo ELIT, esa probabilidad es la transferencia evaluada en el
estado propuesto. `solapamiento-medido.csv` registra directamente esa cantidad
en las once instancias de test. Los otros dos CSV permiten auditar el proxy
$\widehat C_{rep} = 1 - (\bar A - \bar E)/|x_g|$ y sus límites.

## `proxy-por-corrida.csv`

Genera: el mismo script. Una fila por configuración, instancia y semilla, 165 en
total, con el proxy y el solapamiento medido lado a lado. Es la fuente de la
comparación emparejada que cita 5.6; `proxy-por-instancia.csv` agrega sobre las
31 semillas del test y no admite ese emparejamiento.

## `proxy-por-instancia.csv`

Genera: `scripts/derivar_proxy_conservacion.py`. Deriva el proxy en las once
instancias a partir de `results/final/`, en las dos convenciones de denominador
que conviven en 5.6.

| Convención | Denominador | Relación con el informe actual |
|:---|:---|:---|
| `proxy_ciclo` | 5990 evaluaciones del ciclo | Aproximación diagnóstica comparada con la medición directa en 5.6 |
| `proxy_total6000` | 6000 evaluaciones del presupuesto | Convención histórica conservada para mostrar el efecto de incluir la inicialización |

Las dos no dan lo mismo. La inicialización repara una población aleatoria densa
cuya poda escala con el número de columnas, de modo que repartirla entre 6000
evaluaciones contamina el promedio, y contamina más en instancias grandes. En
`scp42` la diferencia queda en 0.012 y pasa inadvertida; en once instancias el
sesgo sube a +0.08 y produce valores por sobre el máximo de la curva, que es
imposible para una probabilidad de conservación.

| Configuración | Referencia de transferencia | `ciclo` | `total6000` |
|:---|---:|---:|---:|
| BPSO | 0.0000 | 0.1755 | 0.2560 |
| IID | 0.4142 | **0.4140** | 0.4939 |
| BGWO | 0.7071 | 0.6936 | 0.7732 |
| BPWO-V3 | 0.7071 | **0.7112** | 0.7906 |
| BPWO-S2 | 0.7311 | **0.7345** | 0.8139 |

Con la convención del ciclo el orden se cumple en las 11 instancias.

## `solapamiento-medido.csv`

Genera: `scripts/medir_conservacion.py`. Registra el solapamiento publicado en
5.6. Contiene una fila por configuración, instancia y semilla, 165 en total,
cada una sobre las 5990 propuestas del ciclo de esa corrida.

| Columna | Qué contiene |
|:---|:---|
| `esperada` | Media de la transferencia sobre las columnas activas del incumbente. La predicción, calculada sin mirar el resultado |
| `real` | Fracción de esas columnas que efectivamente sobrevive a la binarización |
| `estado_medio` | Media del valor absoluto de la variable de estado sobre esas mismas columnas |

Resultado, promediado sobre las once instancias:

| Configuración | Referencia | `real` | Error medio | `estado_medio` |
|:---|---:|---:|---:|---:|
| BPWO-V3 | 0.7071 | **0.7069** | **0.0006** | 0.9997 |
| BPWO-S2 | 0.7311 | **0.7300** | 0.0011 | 0.9997 |
| IID | 0.4142 | **0.4142** | **0.0006** | 0.5001 |
| BGWO | 0.7071 | 0.6888 | 0.0183 | 0.9948 |
| BPSO | cota 0 | 0.1678 | no aplica | 0.0706 a 0.8163 |

Tres lecturas:

1. `esperada` y `real` coinciden al agregarse en las 55 combinaciones de
   configuración e instancia, como exige la regla. La medición es consistente
   consigo misma.
2. `estado_medio` verifica el supuesto que sostiene todo el cálculo analítico.
   La sincronización (4) deja el estado en ±1: la media del valor absoluto es
   0.9997 en las dos variantes de BPWO. En IID es 0.5001, contra la media del
   valor absoluto de un uniforme en [−1,1], que es un medio exacto.
3. El proxy reproduce las diferencias agregadas entre configuraciones, pero solo
   en la convención del ciclo. La distinción importa y no es menor.

| Convención | Pearson | Spearman | MAE | Sesgo | Error máx |
|:---|---:|---:|---:|---:|---:|
| `proxy_ciclo` | 0.9963 | 0.9592 | 0.0138 | +0.0042 | 0.0747 |
| `proxy_total6000`, histórico | 0.9601 | 0.8013 | 0.0840 | +0.0840 | 0.2594 |

Solo la convención del ciclo aproxima adecuadamente la medición directa. La
convención histórica se conserva para documentar por qué la inicialización no
debe mezclarse con las evaluaciones del ciclo.

Además, ese 0.9963 lo empuja la separación entre configuraciones. Dentro de cada
una, la asociación entre instancias es mucho más débil, en parte porque la
conservación real casi no varía y queda poco que correlacionar:

| Configuración | r interno | MAE | Rango de `real` |
|:---|---:|---:|---:|
| BPSO | 0.964 | 0.0253 | 0.3617 |
| BGWO | 0.801 | 0.0093 | 0.0188 |
| IID | 0.526 | 0.0172 | 0.0022 |
| BPWO-V3 | 0.477 | 0.0086 | 0.0035 |
| BPWO-S2 | 0.165 | 0.0085 | 0.0014 |

El proxy es una aproximación agregada, no un sustituto de la medición directa ni
un estimador de la variación entre instancias.

## Dos precisiones sobre la referencia de transferencia

**La conservación de BGWO no vale 0.7071.** Mide 0.6888, sistemáticamente por
debajo. Su
`estado_medio` va de 0.9751 a 1.0020, contra el rango de 0.9996 a 0.9998 de
BPWO: no clava el estado con la misma firmeza. Como V3 es cóncava, el déficit se
descompone en corrimiento de la media más brecha de Jensen,

$$T(1)-\mathbb{E}[T(|z'|)] = [T(1)-T(\mathbb{E}|z'|)] + [T(\mathbb{E}|z'|)-\mathbb{E}T(|z'|)],$$

y ambos términos son computables con lo registrado, porque `esperada` es
$\mathbb{E}T(|z'|)$ y `estado_medio` es $\mathbb{E}|z'|$:

| | Déficit total | Corrimiento | Brecha de Jensen |
|:---|---:|---:|---:|
| BGWO | 0.0183 | 0.0019 | **0.0165 (89.9 %)** |

La brecha representa cerca del 90 % del déficit, lo que da cuenta algebraicamente
de que la caída proviene sobre todo de la heterogeneidad del estado y no de que
su magnitud media sea menor que uno. No es una prueba causal por ablación. En
BPWO-V3 la brecha es $5.8\times10^{-5}$, prácticamente nula, consistente con la
fuerte concentración del estado, aunque no demuestra ausencia de dispersión.

El script registra la media de $|z'|$ pero no su varianza ni sus cuantiles, de
modo que sostener el mecanismo con más detalle exige registrar una medida de
dispersión.

**La velocidad de BPSO no converge a un valor común.** Va de 0.0706 en `scp42` a
0.8163 en `scpnrf2`, un factor de doce, y su conservación la acompaña de 0.0359
a 0.3976. La referencia $T_{V3}(0)=0$ es una cota inferior y no una predicción.
El marco se sostiene igual: la correlación entre `estado_medio` y `real` en las
once instancias es 0.9986.

## Cuatro cautelas

**La unidad experimental es la corrida, no la propuesta.** Las propuestas dentro
de una corrida están relacionadas temporalmente y no son observaciones
independientes. Sirven para la comprobación mecánica entre `esperada` y `real`,
pero cualquier afirmación de estabilidad debe estimarse entre corridas, y por eso
el CSV conserva la semilla. Las dos dispersiones son muy pequeñas en BPWO-V3,
BPWO-S2 e IID, sin que una domine a la otra de forma consistente; en BGWO
predomina la variación entre instancias, y en BPSO ambas son mucho mayores:

| Configuración | Desv. entre semillas | Desv. entre instancias |
|:---|---:|---:|
| BPWO-V3 | 0.00061 | 0.00086 |
| BPWO-S2 | 0.00063 | 0.00047 |
| IID | 0.00072 | 0.00071 |
| BGWO | 0.00147 | 0.00544 |
| BPSO | 0.01104 | 0.10608 |

Las 33 corridas de BPWO-V3 caen entre 0.7044 y 0.7094, una amplitud de 0.0050
alrededor del 0.7071 predicho. Las de IID entre 0.4118 y 0.4165 alrededor de
0.4142. BPSO es el único donde ambas dispersiones son grandes, coherente con que
su estado no queda fijado.

Con tres semillas esto sigue siendo un diagnóstico, sin intervalos de confianza
ni afirmaciones de significancia.

**El signo en S2 no se midió, se infirió.** S2 no es simétrica,
$T_{S2}(1)=0.7311$ y $T_{S2}(-1)=0.2689$, de modo que una media de $|z'|$
cercana a uno no distingue entre ambas. El script no registra el signo. Bajo la
concentración observada, la transferencia media es compatible con alrededor de
0.19 % de componentes negativas en las columnas activas del alfa, resolviendo
$p=(T_{S2}(1)-\overline{T_{S2}})/(T_{S2}(1)-T_{S2}(-1))$. Es una inferencia
fuerte, no una medición.

**El proxy y la medición sí se emparejan.** Las corridas son deterministas y la
instrumentación no consume el generador, de modo que las semillas 0 a 2 de la
medición reproducen exactamente las del historial original. `proxy-por-corrida.csv`
deriva el proxy de esas mismas 165 corridas y permite compararlo par a par en vez
de contra el agregado de las 31 semillas:

| | Pearson | Spearman | MAE | Sesgo | Error máx |
|:---|---:|---:|---:|---:|---:|
| Emparejado, 165 corridas | 0.9966 | 0.9532 | 0.0141 | +0.0046 | 0.0567 |
| Agregado, 31 semillas | 0.9963 | 0.9592 | 0.0138 | +0.0042 | 0.0747 |

**El denominador del proxy acierta por compensación.** La derivación exacta bajo
ELIT es $C=|x'|/|x_\alpha|$, porque las columnas activas de la propuesta son un
subconjunto de las del alfa. El proxy usa $|x_g|$ y la carga neta de reparación,
lo que lo aproxima solo cuando $|x^{rep}|\approx|x_\alpha|\approx|x_g|$. Sobre
las 55 combinaciones, $|x_\alpha|$ queda entre 15.5 % por debajo de $|x_g|$ y su
mismo valor, con el máximo en IID sobre `scpnrh2`, 54.1 frente a 64. Sustituir
$|x_g|$ por el promedio de $|x_\alpha|$, sin reconstruir también el tamaño
reparado de las mismas propuestas, no mejora el resultado:

| Denominador | Pearson | MAE | Sesgo |
|:---|---:|---:|---:|
| $\|x_g\|$, el del informe | 0.9963 | **0.0138** | +0.0042 |
| $\|x_\alpha\|$ medido | 0.9971 | 0.0201 | −0.0201 |

El sesgo cambia de signo, lo que indica que las dos aproximaciones se compensan
parcialmente con $|x_g|$. Eso no convierte a $|x_\alpha|$ en peor denominador
conceptual: muestra que la precisión del proxy original depende en parte de una
cancelación entre errores.

## Sobre la lectura como destrucción y reconstrucción

La conservación medida admite leerse como su complemento, la fracción esperada de
columnas del alfa que se destruye antes de reparar:

| Configuración | Conservación | Destrucción efectiva |
|:---|---:|---:|
| BPWO-S2 | 0.7300 | 0.2700 |
| BPWO-V3 | 0.7069 | 0.2931 |
| BGWO | 0.6888 | 0.3112 |
| IID | 0.4142 | 0.5858 |
| BPSO | variable | variable |

El algoritmo contiene entonces un mecanismo implícito de control de la
destrucción, determinado conjuntamente por la transferencia y por la distribución
del estado. Ese control no está expuesto como un parámetro independiente del
movimiento: elegir V3 o S2 es una decisión categórica, y en BGWO y BPSO la tasa
depende además de una distribución emergente que varía entre instancias. Por eso
conviene llamarlo dinámica emergente de destrucción y reconstrucción, afín a
Large Neighborhood Search, y no afirmar que constituya formalmente un método LNS.
La regla que desacople encender de apagar, propuesta en 5.7, sí lo convertiría en
un parámetro explícito con dos tasas.

## Reproducir

```
.venv/bin/python scripts/derivar_proxy_conservacion.py
.venv/bin/python scripts/medir_conservacion.py \
    --instancias-raiz <ruta>/SCP/Instances
```

El primero necesita `results/final/history.csv`, que no se versiona por peso y
se regenera con el experimento final. El segundo necesita las instancias ORLIB,
que tampoco se versionan, y toma alrededor de dos horas, dominadas por las dos
instancias de diez mil columnas.

`medir_conservacion.py` no modifica el código de producción: envuelve
`BinarizationScheme.apply`, llama a la implementación original y solo observa
sus argumentos y su retorno, de modo que el consumo del generador aleatorio no
cambia y las corridas son las mismas que sin instrumentar.
