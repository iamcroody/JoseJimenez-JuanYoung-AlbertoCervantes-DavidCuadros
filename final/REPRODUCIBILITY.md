# Reproducibilidad — "otra persona clona y obtiene las mismas tablas"

Este documento es el pegamento de reproducibilidad del proyecto: fija los ids y
versiones de modelos, la semilla, el modo de decodificación y el **orden de
corrida** de los notebooks, y mapea cada tabla (`reports/*.csv`) al notebook que
la produce. El requisito mínimo del PDF es que la entrega sea reproducible
(semillas y versiones de modelo fijas) y que **todas** las filas de todas las
tablas usen el mismo eval set y el mismo harness. CI guarda esa invariante en
cada push (ver `.github/workflows/harness.yml`); este archivo documenta cómo
rehacer los números desde cero.

## 0. Clonar con el submódulo del dataset

El catálogo vive en un submódulo pineado. Sin `--recurse-submodules` no hay
dataset y nada reproduce:

```sh
git clone --recurse-submodules \
  https://github.com/iamcroody/JoseJimenez-JuanYoung-AlbertoCervantes-DavidCuadros.git
cd JoseJimenez-JuanYoung-AlbertoCervantes-DavidCuadros
# si ya clonaste sin submódulos:
git submodule update --init --recursive
```

El submódulo `data/exercises-dataset` está pineado al commit
`8872272c9f78cd30889523b79e34e95cca8c5637`. Todos los splits, el eval set y los
índices RAG se derivan de ese commit exacto; cambiarlo cambia las tablas.

En Colab los notebooks clonan en
`/content/JoseJimenez-JuanYoung-AlbertoCervantes-DavidCuadros`; las rutas del
repo se simulan igual en local para que el traslado sea directo.

## 1. Determinismo: semilla y decodificación

- **Semilla única: 42.** `SEED = 42` en `eval/harness.py`, en los scripts de
  datos/entrenamiento y en el juez (`set_seed(42)`). La construcción de splits
  (`scripts/01_macgyver_data.py`) y el muestreo del probe de prompts
  (`m3/probe.py`) son deterministas bajo esta semilla.
- **Decodificación greedy** en todas partes: `do_sample=False` tanto en el
  generador (`make_hf_system`) como en el juez (`Judge._generate`). Dos corridas
  del scorecard producen los mismos números. No hay temperatura, top-p ni
  muestreo en ninguna ruta que alimente una tabla.
- **Entorno:** Python 3.12 (ver `.python-version`; `requires-python >=3.12`).
  Dependencias pineadas a versiones EXACTAS (no rangos) en `pyproject.toml` /
  `uv.lock` y en los `requirements.txt` de cada módulo.

## 2. Ids y versiones de modelos pineados

| Rol | Id del modelo | Dónde se usa |
| --- | --- | --- |
| Generador (base) | `Qwen/Qwen3-1.7B` | M1 fine-tuning, M2/M3 generación, M5 eficiencia/deploy |
| Adaptador LoRA (comiteado) | `models/r16-all-linear/` (r=16, alpha=32, dropout=0.05, all-linear; `peft==0.20.0`) | se aplica sobre Qwen3-1.7B; reproduce sin GPU el modelo afinado |
| Juez (harness M2) | `Qwen/Qwen2.5-1.5B-Instruct` | dimensión 2, rúbrica v1.0 (`eval/rubric.md`) |
| Juez (RAGAS M3) | `Qwen/Qwen2.5-3B-Instruct` (fallback `Qwen/Qwen2.5-1.5B-Instruct`) | métricas RAGAS locales |
| Embedder (similitud, dim 1) | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | dimensión 1 del harness |
| Embedder (RAG denso) | `BAAI/bge-small-en-v1.5` | índice Chroma de M3 |
| Reranker (RAG) | `cross-encoder/ms-marco-MiniLM-L-6-v2` | reordenamiento del pool de candidatos |
| Cuantización int4 (M5) | `bitsandbytes==0.44.1` (nf4, QLoRA-style) | técnica de eficiencia sobre el generador |

Pines de librerías de inferencia (idénticos entre `m3/requirements.txt`,
`final/efficiency/requirements.txt` y `final/deploy/requirements.txt` para que no
haya deriva de versiones): `transformers==5.16.1`, `peft==0.20.0`,
`sentence-transformers==6.0.1`, `chromadb==1.5.5`, `rouge_score==0.1.2`,
`opentelemetry-api/sdk==1.42.1`, `bitsandbytes==0.44.1`; el Space añade
`gradio==5.9.1`, `wandb==0.18.7` y `torch==2.5.1`.

Ningún modelo requiere una clave de pago. W&B y cualquier API externa hacen
no-op elegante sin clave (ver `final/deploy/wandb_utils.py`); las claves, cuando
existen, van como *secrets* del Space, nunca en el repo.

## 3. Orden de corrida de los notebooks → tablas

Los números de cada tabla salen de correr los notebooks en este orden. Cada
notebook clona el repo con submódulos y construye los splits/índices desde el
commit pineado, así que el **mismo eval set** (`eval/eval_set.json`, 14 casos, 4
adversariales) y el **mismo harness** (`eval/harness.py`) alimentan todas las
filas. La latencia se mide en la misma máquina (Colab T4) y se reporta p50 y p95.

| # | Notebook | Qué corre | Tablas que produce (`reports/`) |
| --- | --- | --- | --- |
| 1 | `notebooks/M1_macgyver.ipynb` | Fine-tuning LoRA (r16 all-linear) del generador, ~2 h T4; eval zero-shot vs afinado | `finetuned.json`, `finetuned_test.json`, `zeroshot.json`, `zeroshot_test.json`, `replies_finetuned_test.json` |
| 2 | `notebooks/M2_harness.ipynb` | Scorecard base (3 dimensiones) sobre el modelo afinado vs zero-shot; probes de sesgo del juez | `scorecard_baseline.csv` (+ `scorecard_baseline_detail.json`), `replies_m2.json` |
| 3a | `notebooks/M3_rag_v1.ipynb` | Primera corrida RAG (prompt v1), ~1 h T4 | `scorecard_m3_v1.csv`, `latency_m3_v1.csv`, `ragas_m3_v1.csv` (+ `*_detail.json`), `replies_m3_v1.json` |
| 3b | `notebooks/M3_rag.ipynb` | Corrida RAG final (prompt v2a): naive / advanced / final + ablaciones | `scorecard_m3.csv`, `latency_m3.csv`, `ragas_m3.csv` (+ `*_detail.json`), `replies_m3.json` |
| 4 | `notebooks/M5_efficiency.ipynb` | Cuantización int4 (nf4) del generador; antes/después de calidad + latencia; válvula + adversarial + invariancia sobre la versión cuantizada | `reports/efficiency_m5.csv` (+ `efficiency_m5_detail.json`) |
| 5 | Despliegue (`final/deploy/`) | Space Gradio en HF: componente visual, health check, logging JSON por petición (sin PII), observabilidad W&B, filtro de satisfacción de restricciones | Métricas de observabilidad en W&B (no CSV); el Space se verifica end-to-end |

Notas de mapeo:

- `reports/scorecard_m3.csv` tiene una columna por sistema
  (`zeroshot`, `lora_m1`, `rag_naive`, `rag_advanced`, las ablaciones
  `rag_adv_no_hybrid/_rerank/_hyde`, y `rag_final`): agregar un sistema es
  agregar una columna, por diseño del harness.
- El sufijo `_v1` marca la primera corrida RAG (prompt v1), conservada para la
  tabla evolutiva y la ablación de prompt v1 → v2a. El prompt v2a se eligió en
  un probe de validación (`m3/probe.py`), **nunca** sobre el eval set.
- La tabla de eficiencia (`reports/efficiency_m5.csv`) reusa el mismo harness
  (dimensión 3 + válvula/adversarial/invariancia), así que la fila "después" es
  comparable con el resto de tablas.

## 4. Verificación sin GPU y guardián de regresión

Lo que **no** necesita GPU ni descarga de pesos corre localmente y en CI:

```sh
python scripts/01_macgyver_data.py       # construye splits (seed 42) desde el submódulo
python scripts/02_score.py               # self-test del scorer (M1)
python eval/harness.py                   # self-test del harness (M2)
python m3/selftest.py                    # self-test de M3 (parser, válvula, sistemas)
python final/efficiency/selftest.py      # self-test de eficiencia (M5)
python final/deploy/selftest.py          # self-test de despliegue (M5)
python final/ci/harness_smoke.py         # humo determinista del harness (dim 3 vs stub)
python eval/build_eval_set.py            # regenera eval_set.json byte-idéntico
```

`.github/workflows/harness.yml` corre exactamente esto en cada push y pull
request (Ubuntu, Python 3.12, solo `numpy + rouge_score`), y **falla si el
puntaje del humo cae** por debajo de la línea base comiteada
(`final/ci/harness_smoke.py`, 5 casos del eval set, dimensión 3 = 5/5 sobre el
sistema stub). Esto realiza, de forma factible en CPU, el *"3 a 5 casos … falle
si el puntaje cae"* del PDF.

**El guardián de regresión de calidad sobre las salidas del modelo REAL corre en
Colab T4**, dentro de los notebooks de M2/M3/M5: CI cuida la plomería y la
integridad del eval set (la "misma vara"), no la calidad del modelo, porque esa
exige la T4 y los pesos de Qwen que un runner de CI gratuito no tiene.
