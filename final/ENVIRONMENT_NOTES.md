# Notas de entorno (Módulo 5)

Estas notas registran las restricciones del entorno local (sandbox) donde se
desarrolla el Módulo 5 y qué se pudo verificar aquí vs qué se difiere a Colab
T4. Las comparte el equipo para trabajar con los mismos hechos.

## Restricciones del sandbox local

- **Sin red.** DNS falla para `pypi.org` y `github.com`. Por tanto
  `pip install`, `uv sync` y `git submodule init/update` NO funcionan. No se
  intentan: pierden tiempo. La instalación normal (`git clone
  --recurse-submodules` + `uv sync`) solo aplica fuera del sandbox o en Colab.
- **Sin GPU.** Es macOS sin GPU. El entrenamiento LoRA completo (M1, ~2 h), el
  scorecard completo (M2, necesita modelo juez), la corrida completa de M3
  (~1 h) y RAGAS NO corren aquí. Eso es esperado y se verifica en Colab T4.
- **Submódulo del dataset no inicializado.** `data/exercises-dataset` está
  pineado al commit `8872272` pero no inicializado en este repo. Para habilitar
  los self-tests locales se copió el contenido del repo hermano ya poblado y
  pineado al mismo commit:

  ```sh
  cp -R ../models-for-exercises-dataset/data/exercises-dataset/. data/exercises-dataset/
  ```

  Tras copiar se eliminó el archivo `.git` que venía dentro de esa copia
  (apuntaba a un gitdir inexistente y rompía `git status` del repo padre). Los
  archivos copiados quedan como contenido del submódulo y NO se agregan ni se
  comitean desde el repo padre: el pin del submódulo sigue siendo `8872272` y
  `git status` del padre queda limpio.
- **Venv local.** Se usa `../models-for-exercises-dataset/.venv/bin/python`
  (Python 3.12.13), que trae `torch`, `transformers`, `datasets`,
  `rouge_score` y `numpy`. NO trae `sentence_transformers`, `chromadb`,
  `rank_bm25` ni `sklearn`; cualquier verificación que los necesite se marca
  como "verificar en Colab T4".

## Rutas estilo Colab

Todo lo que el equipo entregue debe seguir corriendo en un Colab T4 gratuito.
Los notebooks clonan el repo en
`/content/JoseJimenez-JuanYoung-AlbertoCervantes-DavidCuadros`. Simular esa
ruta al escribir código de `final/` para que el traslado a Colab sea directo.

## Resultado de los self-tests (sin GPU, venv hermano)

Comando base: `PY=../models-for-exercises-dataset/.venv/bin/python`

| Self-test | Comando | Resultado local | Nota |
| --- | --- | --- | --- |
| Construcción de splits | `$PY scripts/01_macgyver_data.py` | PASÓ | Genera `data/processed/macgyver/{train,val,test}.jsonl` + `meta.json`. Requiere correr antes que `02_score.py`. |
| Scorer (M1) | `$PY scripts/02_score.py` | PASÓ | `self-test PASSED`: constraint satisfaction 100%, refusal F1 100%. Depende de los splits anteriores. |
| Eval set (M2) | `$PY eval/build_eval_set.py` | PASÓ | Reescribe `eval/eval_set.json` **byte-idéntico** (`git diff --stat` sin cambios). 14 casos, 4 adversariales. |
| Harness (M2) | `$PY eval/harness.py` | PASÓ | `self-test PASSED`. En el venv hermano no hizo falta `sentence_transformers` para el modo self-test. |
| Self-test M3 | `$PY m3/selftest.py` | PASÓ | `self-test PASSED`. Cubre tool-call, step cap, recorder, nombres de sistemas y prompt por defecto `v2a`. |
| Validez JSON de notebooks | `$PY -c "import json; json.load(open(nb))"` | PASÓ (4/4) | `M1_macgyver`, `M2_harness`, `M3_rag`, `M3_rag_v1`. |

Observación: en este sandbox el venv hermano fue suficiente para que `eval/harness.py`
y `m3/selftest.py` pasaran en modo self-test (no requirieron
`sentence_transformers`/`chromadb`/`rank_bm25`, que sí harían falta para las
corridas completas con recuperación y reranking).

## Qué queda para Colab T4 (no verificable localmente)

- Entrenamiento LoRA completo de M1 (~2 h, GPU).
- Scorecard completo de M2 (modelo juez + similitud por embeddings).
- Corrida completa de M3 (~1 h; `chromadb` + `sentence_transformers` +
  `rank_bm25`).
- Medición de eficiencia antes/después y RAGAS.
- El Space desplegado (health check, logging JSON, observabilidad W&B).
