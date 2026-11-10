# Resultados de eficiencia (Módulo 5) — int4 (nf4)

> **Estado: esqueleto.** Las celdas numéricas se llenan con la corrida en
> **Colab T4** (`notebooks/M5_efficiency.ipynb`). No hay métricas inventadas:
> cada `(lo llena la corrida T4)` se reemplaza por el número real tras ejecutar.
> El notebook escribe `reports/efficiency_m5.csv` y
> `reports/efficiency_m5_detail.json`, y de ahí se copian los valores aquí.

## Presupuesto de pérdida aceptable (fijado ANTES de medir)

Aceptamos perder hasta **5 puntos** de tasa de acierto de dominio (dimensión 3)
en el sistema de **RAG avanzado** — bajar de **42.9 %** a no menos de **37.9 %** —
siempre que int4:

- reduzca el **pico de VRAM** por debajo de la huella fp16, y
- **no aumente** la latencia **p95** respecto de la versión fp16.

**¿Se cumplió el presupuesto?** `(lo declara la corrida T4: SÍ / NO + por qué)`

## Condiciones de la medición

- Mismo eval set congelado: `eval/eval_set.json` (14 casos, 4 adversariales).
- Mismo harness congelado: `eval/harness.py` (3 dimensiones).
- Misma máquina (Colab T4), seed 42, decodificación greedy.
- Sistema medido: `sistema_rag_avanzado` (RAG avanzado) sobre cada generador.
  Opcionalmente también `sistema_final` (agente) — ver detalle JSON.
- Latencia: por consulta, agregada con `numpy.percentile` (p50 y p95).
- Memoria: `torch.cuda.max_memory_allocated` (pico VRAM) y huella de pesos.

## Tabla antes/después

| Version | Quality (harness) | Latency p50 | Latency p95 | Memory/size |
| --- | --- | --- | --- | --- |
| Antes (M4/M3 fp16) | `(lo llena la corrida T4)` | `(lo llena la corrida T4)` | `(lo llena la corrida T4)` | `(lo llena la corrida T4)` |
| Despues (int4 nf4) | `(lo llena la corrida T4)` | `(lo llena la corrida T4)` | `(lo llena la corrida T4)` | `(lo llena la corrida T4)` |

`Quality (harness)` = tasa de acierto de dominio (dimensión 3) del RAG avanzado
sobre los 14 casos; el detalle (dim 1 similitud, dim 2 juez, gold vs
adversarial) va en `reports/efficiency_m5.csv`.

## Pruebas que no miden exactitud, sobre int4

| Prueba | Resultado (int4) |
| --- | --- |
| Invariancia (fracción de 14 respuestas idénticas bajo perturbación trivial) | `(lo llena la corrida T4)` |
| Válvula de escape dispara en adv01 y adv04 | `(lo llena la corrida T4: ambas SÍ / cuáles no)` |
| Bloque adversarial adv01..adv04 (dimensión 3) | `(lo llena la corrida T4)` |

## Lectura (se escribe tras la corrida)

`(lo llena la corrida T4: ¿bajó la VRAM? ¿cambió la latencia p50/p95? ¿cuántos`
`puntos de dimensión 3 se perdieron o ganaron? ¿se mantuvieron válvula e`
`invariancia? conclusión sobre si int4 vale la pena para este sistema.)`
