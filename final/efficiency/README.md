# Eficiencia (Módulo 5) — cuantización int4 (nf4) del generador

Una sola técnica de eficiencia, aplicada al componente más pesado y lento del
sistema, con tabla antes/después sobre el mismo eval set y la misma máquina, y
un presupuesto de pérdida aceptable fijado **antes** de medir.

## 1. Qué componente y por qué

El sistema final tiene tres piezas: recuperación (Chroma + BM25 + reranker),
generador (Qwen3-1.7B + adaptador LoRA de M1) y el agente con la herramienta
`find_exercises`. La ablación de M3 muestra que **la latencia la domina el
generador**, no la recuperación:

- HyDE, la técnica ganadora del RAG avanzado, es una **generación completa
  extra** antes de recuperar: cada consulta paga dos pasadas del generador.
- El RAG avanzado mide **42.9 %** de tasa de acierto de dominio (dimensión 3)
  con **15.2 s** de latencia mediana; el sistema agente (`sistema_final`) baja a
  **28.6 %** y sube a **38.9 s**, casi todo gasto de generación por los pasos de
  herramienta.

La recuperación (índice denso + BM25 + un cross-encoder pequeño) es barata al
lado de dos generaciones de un modelo de 1.7B. Por eso la técnica se aplica al
**generador** y no al recuperador: ahí está el cuello de botella.

## 2. La técnica: int4 (nf4) estilo QLoRA

Se carga el base `Qwen/Qwen3-1.7B` con `bitsandbytes` en 4 bits y se le adjunta
el adaptador LoRA comiteado `models/r16-all-linear`:

```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",        # nf4, no fp4
    bnb_4bit_compute_dtype=fp16,      # cómputo fp16 (T4)
    bnb_4bit_use_double_quant=True,   # doble cuantización
)
```

Es exactamente la rama `four_bit` de `scripts/macgyver.py::load_model`, que ya
usa el proyecto para entrenar/evaluar en 4 bits. El adaptador **no se fusiona**
en el base de 4 bits (un `Linear` cuantizado no puede absorber los pesos LoRA);
se aplica encima con `peft`, igual que documenta ese mismo `load_model`.

El pipeline de RAG **no cambia**: `efficiency.load_int4_generator()` devuelve un
callable `generate(prompt)->reply` con la MISMA firma que `systems.bind` espera
para `llm` (reusa `harness.make_hf_system`, con greedy, seed 42 y
`enable_thinking=False`). Así `sistema_rag_avanzado` / `sistema_final` corren sin
tocar una línea sobre el generador cuantizado.

## 3. Presupuesto de pérdida aceptable (fijado ANTES de medir)

> **Aceptamos perder hasta 5 puntos de tasa de acierto de dominio (dimensión 3)
> en el sistema de RAG avanzado — es decir, bajar de 42.9 % a no menos de
> 37.9 % — siempre que int4 reduzca el pico de VRAM por debajo de la huella
> fp16 y NO aumente la latencia p95 respecto de la versión fp16.**

Justificación de los números: 42.9 % es la dimensión 3 real del RAG avanzado en
M3; 5 puntos es ~1 caso de los 14 del eval set, el margen de ruido que ya
tolera el equipo entre corridas. El beneficio esperado (menos VRAM) debe ser
real y la latencia no debe empeorar, porque cuantizar a veces introduce overhead
de de-cuantización en modelos pequeños sobre T4. La corrida final en Colab debe
**declarar explícitamente si el presupuesto se cumplió o no** (hay una celda y
una fila en `RESULTS.md` para ello).

## 4. Pruebas que NO miden exactitud, repetidas sobre int4

La asignación exige repetir sobre la versión eficiente las pruebas que no miden
exactitud. Se corren sobre el sistema int4 (`final/efficiency/invariance.py`):

1. **Sonda de invariancia** — con greedy + seed fija, la respuesta no debería
   cambiar ante una perturbación trivial que preserva el significado (espacios
   extra, reordenar las dos frases, sinónimo de "I have"). Se reporta la
   fracción de los 14 casos cuya respuesta queda idéntica.
2. **Válvula de escape** — adv01 (par imposible) y adv04 (objeto inseguro)
   deben seguir disparando la válvula con el fraseo exacto que el parser
   congelado del harness reconoce (`mg.parse_reply(...)['refused']`).
3. **Bloque adversarial** — adv01..adv04 se re-puntúan con `harness.domain_check`
   sobre el sistema cuantizado.

## 5. Cómo se corre (Colab T4)

1. Abrir `notebooks/M5_efficiency.ipynb` en Colab con runtime **T4 GPU**.
2. La primera celda clona este repo en la rama de trabajo dentro de
   `/content/JoseJimenez-JuanYoung-AlbertoCervantes-DavidCuadros`
   (`--recurse-submodules`) e instala `final/efficiency/requirements.txt`
   (bitsandbytes + el stack de m3).
3. Ejecutar de arriba a abajo. El notebook:
   - construye el recuperador de M3 una vez (reusa `m3/rag.py` + `m3/corpus.py`);
   - mide la fila **Antes** con el generador fp16;
   - mide la fila **Después** con el generador int4 de `efficiency.py`;
   - corre invariancia + válvula + adversarial sobre la versión int4;
   - escribe `reports/efficiency_m5.csv`, `reports/efficiency_m5_detail.json` y
     la tabla antes/después en `final/efficiency/RESULTS.md`.

La tabla y los números se producen en la corrida T4 y se **pegan** en
`RESULTS.md` tras la corrida. El repo se entrega con los números en blanco
(marcados "lo llena la corrida T4"); **no se inventa ninguna métrica**.

## 6. Verificación sin GPU (local)

Sin red y sin GPU no se puede cargar el modelo de 4 bits ni correr el harness
completo; eso se verifica en Colab T4. Lo que sí corre localmente:

```sh
# plomería pura (solo numpy): percentiles, perturbación, esquema del dict
python final/efficiency/selftest.py

# sintaxis de los módulos que dependen de GPU
python -c "import ast; [ast.parse(open(f'final/efficiency/{m}').read()) \
    for m in ('efficiency.py','measure.py','invariance.py')]; print('ok')"

# validez JSON del notebook
python -c "import json; json.load(open('notebooks/M5_efficiency.ipynb')); print('ok')"
```

## 7. Archivos

| Archivo | Qué es |
| --- | --- |
| `efficiency.py` | Carga int4 (nf4) del generador + adaptador LoRA; expone `generate(prompt)->reply` compatible con `systems.bind`. |
| `measure.py` | Latencia p50/p95 (numpy), pico de VRAM, huella de pesos y calidad vía el harness congelado; escribe CSV/JSON. |
| `invariance.py` | Sonda de invariancia (perturbación pura), check de válvula y bloque adversarial sobre el sistema int4. |
| `selftest.py` | Self-test sin GPU (solo numpy): percentiles, perturbación y esquema del dict. |
| `requirements.txt` | Pines exactos: m3 + `bitsandbytes==0.44.1`. |
| `RESULTS.md` | Esqueleto de la tabla antes/después con celdas en blanco para la corrida T4. |
